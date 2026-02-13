use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use clap::Parser;
use oxislam_features::keypoint::Keypoint;
use oxislam_features::matcher::brute_force::BruteForceMatcher;
use oxislam_features::matcher::distance::Hamming;
use oxislam_features::pipeline::orb::OrbPipeline;
use oxislam_features::traits::matcher::DescriptorMatcher;
use oxislam_features::traits::pipeline::FeaturePipeline;
use oxislam_geometry::Point2;
use oxislam_image::image::Image;
use oxislam_image::{ConvertTo, Gray, Rgb, gaussian_3x3};
use oxislam_viz::canvas::side_by_side;
use oxislam_viz::color::distinct_color;
use oxislam_viz::drawing::{draw_cross, draw_line};

/// Match features across video frames to demonstrate ORB stability
#[derive(Parser, Debug)]
#[command(name = "match_video")]
#[command(about = "Video feature matching stability example", long_about = None)]
struct Args {
    /// Path to input video
    video: PathBuf,

    /// Path to output video (defaults to matches.mp4 next to input)
    #[arg(short, long)]
    output: Option<PathBuf>,
}

struct VideoInfo {
    width: usize,
    height: usize,
    fps: String,
    num_frames: usize,
}

fn probe_video(path: &PathBuf) -> VideoInfo {
    let output = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
            "-of", "csv=p=0",
        ])
        .arg(path)
        .output()
        .expect("failed to run ffprobe — is ffmpeg installed?");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("ffprobe failed: {stderr}");
        std::process::exit(1);
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = text.trim().split(',').collect();
    if parts.len() < 4 {
        eprintln!("unexpected ffprobe output: {text}");
        std::process::exit(1);
    }

    VideoInfo {
        width: parts[0].parse().expect("invalid width"),
        height: parts[1].parse().expect("invalid height"),
        fps: parts[2].to_string(),
        num_frames: parts[3].parse().unwrap_or(0),
    }
}

fn rgb_image_from_bytes(bytes: &[u8], width: usize, height: usize) -> Image<Rgb<u8>> {
    let pixels: Vec<Rgb<u8>> = bytes
        .chunks_exact(3)
        .map(|c| Rgb::new(c[0], c[1], c[2]))
        .collect();
    Image::new(width, height, width, pixels)
}

fn image_to_bytes(image: &Image<Rgb<u8>>) -> &[u8] {
    let data = image.data();
    // SAFETY: Rgb<u8> is repr(C) with 3 u8 fields, no padding.
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 3) }
}

fn main() {
    let args = Args::parse();
    let output_path = args.output.unwrap_or_else(|| {
        let dir = args.video.parent().unwrap_or_else(|| std::path::Path::new("."));
        dir.join("matches.mp4")
    });

    // Probe video metadata
    let info = probe_video(&args.video);
    let (w, h) = (info.width, info.height);
    let canvas_w = 2 * w;
    let frame_bytes = w * h * 3;
    let total_frames = info.num_frames;
    println!("Input: {}x{} @ {} fps, {} frames", w, h, info.fps, total_frames);

    // Spawn decoder
    let mut decoder = Command::new("ffmpeg")
        .args([
            "-i",
        ])
        .arg(&args.video)
        .args([
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-v", "quiet",
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn ffmpeg decoder");

    // Spawn encoder
    let mut encoder = Command::new("ffmpeg")
        .args([
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", &format!("{canvas_w}x{h}"),
            "-r", &info.fps,
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
        ])
        .arg(&output_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn ffmpeg encoder");

    let decoder_stdout = decoder.stdout.as_mut().expect("no decoder stdout");
    let encoder_stdin = encoder.stdin.as_mut().expect("no encoder stdin");

    let pipeline = OrbPipeline::default();
    let matcher = BruteForceMatcher::new(Hamming).with_ratio(0.75);
    let green = Rgb::new(0u8, 255u8, 0u8);

    // Read first frame as reference
    let mut buf = vec![0u8; frame_bytes];
    decoder_stdout.read_exact(&mut buf).expect("failed to read first frame");
    let ref_rgb = rgb_image_from_bytes(&buf, w, h);
    let ref_gray: Image<Gray<f32>> = ref_rgb.view().to();
    let ref_gray = gaussian_3x3(&ref_gray.view());
    let ref_feats = pipeline.extract(&ref_gray.view());
    let ref_descs: Vec<_> = ref_feats.iter().map(|f| f.descriptor.clone()).collect();
    let ref_kps: Vec<Keypoint> = ref_feats.iter().map(|f| f.keypoint).collect();
    println!("Reference frame: {} features", ref_kps.len());

    let t0 = Instant::now();
    let mut frame_count = 0usize;
    let mut total_matches = 0usize;

    // Process frames (including first frame, re-read from buf)
    let mut frame_buf = buf;
    let mut first = true;

    loop {
        if first {
            first = false;
        } else {
            if decoder_stdout.read_exact(&mut frame_buf).is_err() {
                break;
            }
        }

        let cur_rgb = rgb_image_from_bytes(&frame_buf, w, h);
        let cur_gray: Image<Gray<f32>> = cur_rgb.view().to();
        let cur_gray = gaussian_3x3(&cur_gray.view());
        let cur_feats = pipeline.extract(&cur_gray.view());
        let cur_descs: Vec<_> = cur_feats.iter().map(|f| f.descriptor.clone()).collect();
        let cur_kps: Vec<Keypoint> = cur_feats.iter().map(|f| f.keypoint).collect();

        let matches = matcher.match_descriptors(&ref_descs, &cur_descs);

        // Build side-by-side canvas
        let mut canvas = side_by_side(&ref_rgb.view(), &cur_rgb.view());

        // Draw keypoints
        for kp in &ref_kps {
            draw_cross(&mut canvas, kp.position, green, 2);
        }
        for kp in &cur_kps {
            let shifted = Point2::new(kp.position.x + w as f32, kp.position.y);
            draw_cross(&mut canvas, shifted, green, 2);
        }

        // Draw match lines
        for m in &matches {
            let p1 = ref_kps[m.source_idx].position;
            let p2 = cur_kps[m.reference_idx].position;
            let p2_shifted = Point2::new(p2.x + w as f32, p2.y);
            draw_line(&mut canvas, p1, p2_shifted, distinct_color(m.source_idx));
        }

        // Write frame to encoder
        let raw = image_to_bytes(&canvas);
        encoder_stdin.write_all(raw).expect("failed to write frame");

        total_matches += matches.len();
        frame_count += 1;

        eprint!("\rFrame {frame_count}/{total_frames}");
    }

    // Finish encoding
    drop(encoder.stdin.take());
    let status = encoder.wait().expect("encoder failed");
    let _ = decoder.wait();

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!();
    println!("Processed {frame_count} frames in {elapsed:.1}s ({:.1} fps)", frame_count as f64 / elapsed);
    println!("Average matches per frame: {:.1}", total_matches as f64 / frame_count.max(1) as f64);
    if status.success() {
        println!("Saved to {}", output_path.display());
    } else {
        eprintln!("Warning: encoder exited with {status}");
    }
}
