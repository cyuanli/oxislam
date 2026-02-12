use std::path::PathBuf;
use std::process;
use std::time::Instant;

use clap::Parser;
use oxislam_features::descriptor::brief::{BriefExtractor128, BriefExtractor256, BriefExtractor512};
use oxislam_features::descriptor::patch::PatchExtractor;
use oxislam_features::detector::fast::FastDetector;
use oxislam_features::detector::harris::HarrisDetector;
use oxislam_features::keypoint::Keypoint;
use oxislam_features::matcher::brute_force::BruteForceMatcher;
use oxislam_features::matcher::distance::{Hamming, L2Squared};
use oxislam_features::traits::descriptor::DescriptorExtractor;
use oxislam_features::traits::detector::KeypointDetector;
use oxislam_features::traits::matcher::{DescriptorMatch, DescriptorMatcher};
use oxislam_image::image::{Image, ImageView};
use oxislam_image::{ConvertTo, Gray, gaussian_3x3};

/// Match features between two images
#[derive(Parser, Debug)]
#[command(name = "match_features")]
#[command(about = "Feature matching example", long_about = None)]
struct Args {
    /// Path to first input image
    image1: PathBuf,

    /// Path to second input image
    image2: PathBuf,

    /// Path to output image (defaults to matches.png in same directory as image1)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Detector to use: fast, harris
    #[arg(short, long, default_value = "fast")]
    detector: String,

    /// Descriptor to use: brief128, brief256, brief512, patch
    #[arg(short = 'x', long, default_value = "brief256")]
    descriptor: String,
}

fn load_gray(path: &PathBuf) -> (image::RgbImage, Image<Gray<f32>>) {
    let img = image::open(path).unwrap_or_else(|e| {
        eprintln!("Error: failed to open {}: {e}", path.display());
        process::exit(1);
    });
    let rgb = img.to_rgb8();
    let luma = img.to_luma8();
    let (w, h) = (luma.width() as usize, luma.height() as usize);
    let gray_u8: Image<Gray<u8>> = Image::from_raw(w, h, w, luma.into_raw());
    let gray_f32: Image<Gray<f32>> = gray_u8.view().to();
    (rgb, gray_f32)
}

struct MatchResult {
    keypoints1: Vec<Keypoint>,
    keypoints2: Vec<Keypoint>,
    matches: Vec<DescriptorMatch>,
}

fn detect(
    detector: &str,
    img1: &ImageView<Gray<f32>>,
    img2: &ImageView<Gray<f32>>,
) -> (Vec<Keypoint>, Vec<Keypoint>) {
    let det: Box<dyn KeypointDetector<Gray<f32>>> = match detector {
        "fast" => Box::new(FastDetector::default()),
        "harris" => Box::new(HarrisDetector::default()),
        _ => {
            eprintln!("Error: unknown detector '{detector}'");
            eprintln!("Available detectors: fast, harris");
            process::exit(1);
        }
    };
    let t0 = Instant::now();
    let kps1 = det.detect(img1);
    let kps2 = det.detect(img2);
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "Detected {} + {} keypoints in {ms:.1}ms ({})",
        kps1.len(),
        kps2.len(),
        detector,
    );
    (kps1, kps2)
}

fn describe_and_match(
    descriptor: &str,
    img1: &ImageView<Gray<f32>>,
    img2: &ImageView<Gray<f32>>,
    kps1: Vec<Keypoint>,
    kps2: Vec<Keypoint>,
) -> MatchResult {
    match descriptor {
        "brief128" => brief_pipeline(&BriefExtractor128::default(), img1, img2, kps1, kps2),
        "brief256" => brief_pipeline(&BriefExtractor256::default(), img1, img2, kps1, kps2),
        "brief512" => brief_pipeline(&BriefExtractor512::default(), img1, img2, kps1, kps2),
        "patch" => patch_pipeline(img1, img2, kps1, kps2),
        _ => {
            eprintln!("Error: unknown descriptor '{descriptor}'");
            eprintln!("Available descriptors: brief128, brief256, brief512, patch");
            process::exit(1);
        }
    }
}

fn brief_pipeline<const L: usize>(
    extractor: &oxislam_features::descriptor::brief::BriefExtractor<L>,
    img1: &ImageView<Gray<f32>>,
    img2: &ImageView<Gray<f32>>,
    kps1: Vec<Keypoint>,
    kps2: Vec<Keypoint>,
) -> MatchResult {
    let t0 = Instant::now();
    let feats1 = extractor.describe(img1, kps1);
    let feats2 = extractor.describe(img2, kps2);
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "Extracted {} + {} descriptors in {ms:.1}ms",
        feats1.len(),
        feats2.len(),
    );

    let matcher = BruteForceMatcher::new(Hamming).with_ratio(0.75);
    let descs1: Vec<_> = feats1.iter().map(|f| f.descriptor.clone()).collect();
    let descs2: Vec<_> = feats2.iter().map(|f| f.descriptor.clone()).collect();
    let t1 = Instant::now();
    let matches = matcher.match_descriptors(&descs1, &descs2);
    let ms = t1.elapsed().as_secs_f64() * 1000.0;
    println!("Found {} matches in {ms:.1}ms", matches.len());

    MatchResult {
        keypoints1: feats1.into_iter().map(|f| f.keypoint).collect(),
        keypoints2: feats2.into_iter().map(|f| f.keypoint).collect(),
        matches,
    }
}

fn patch_pipeline(
    img1: &ImageView<Gray<f32>>,
    img2: &ImageView<Gray<f32>>,
    kps1: Vec<Keypoint>,
    kps2: Vec<Keypoint>,
) -> MatchResult {
    let extractor = PatchExtractor::<7, 49>::default();
    let t0 = Instant::now();
    let feats1 = extractor.describe(img1, kps1);
    let feats2 = extractor.describe(img2, kps2);
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "Extracted {} + {} descriptors in {ms:.1}ms",
        feats1.len(),
        feats2.len(),
    );

    let matcher = BruteForceMatcher::new(L2Squared).with_ratio(0.75);
    let descs1: Vec<_> = feats1.iter().map(|f| f.descriptor.clone()).collect();
    let descs2: Vec<_> = feats2.iter().map(|f| f.descriptor.clone()).collect();
    let t1 = Instant::now();
    let matches = matcher.match_descriptors(&descs1, &descs2);
    let ms = t1.elapsed().as_secs_f64() * 1000.0;
    println!("Found {} matches in {ms:.1}ms", matches.len());

    MatchResult {
        keypoints1: feats1.into_iter().map(|f| f.keypoint).collect(),
        keypoints2: feats2.into_iter().map(|f| f.keypoint).collect(),
        matches,
    }
}

fn draw_cross(img: &mut image::RgbImage, cx: i32, cy: i32, color: image::Rgb<u8>) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let arm = 2;
    for d in -arm..=arm {
        let px = cx + d;
        let py = cy + d;
        if px >= 0 && px < w && cy >= 0 && cy < h {
            img.put_pixel(px as u32, cy as u32, color);
        }
        if cx >= 0 && cx < w && py >= 0 && py < h {
            img.put_pixel(cx as u32, py as u32, color);
        }
    }
}

fn draw_line(img: &mut image::RgbImage, x0: i32, y0: i32, x1: i32, y1: i32, color: image::Rgb<u8>) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;
    loop {
        if x >= 0 && x < w && y >= 0 && y < h {
            img.put_pixel(x as u32, y as u32, color);
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

fn match_color(index: usize) -> image::Rgb<u8> {
    let hue = ((index as f32) * 137.508) % 360.0; // golden angle
    let (r, g, b) = hue_to_rgb(hue);
    image::Rgb([r, g, b])
}

fn hue_to_rgb(hue: f32) -> (u8, u8, u8) {
    let h = hue / 60.0;
    let x = (1.0 - (h % 2.0 - 1.0).abs()) * 255.0;
    let c = 255.0;
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (r as u8, g as u8, b as u8)
}

fn main() {
    let args = Args::parse();
    let output_path = args.output.unwrap_or_else(|| {
        let dir = args.image1.parent().unwrap_or_else(|| std::path::Path::new("."));
        dir.join("matches.png")
    });

    // Load images
    let (rgb1, gray1) = load_gray(&args.image1);
    let (w1, h1) = (rgb1.width(), rgb1.height());
    println!("Image 1: {w1}x{h1} from {}", args.image1.display());

    let (rgb2, gray2) = load_gray(&args.image2);
    let (w2, h2) = (rgb2.width(), rgb2.height());
    println!("Image 2: {w2}x{h2} from {}", args.image2.display());

    // Smooth images with Gaussian 3x3
    let gray1 = gaussian_3x3(&gray1.view());
    let gray2 = gaussian_3x3(&gray2.view());

    // Detect, describe, and match
    let (kps1, kps2) = detect(&args.detector, &gray1.view(), &gray2.view());
    let result = describe_and_match(&args.descriptor, &gray1.view(), &gray2.view(), kps1, kps2);

    // Create side-by-side visualization
    let out_w = w1 + w2;
    let out_h = h1.max(h2);
    let mut canvas = image::RgbImage::new(out_w, out_h);

    // Copy images
    for y in 0..h1 {
        for x in 0..w1 {
            canvas.put_pixel(x, y, *rgb1.get_pixel(x, y));
        }
    }
    for y in 0..h2 {
        for x in 0..w2 {
            canvas.put_pixel(w1 + x, y, *rgb2.get_pixel(x, y));
        }
    }

    // Draw keypoints
    let green = image::Rgb([0u8, 255, 0]);
    for kp in &result.keypoints1 {
        draw_cross(&mut canvas, kp.position.x.round() as i32, kp.position.y.round() as i32, green);
    }
    for kp in &result.keypoints2 {
        draw_cross(
            &mut canvas,
            kp.position.x.round() as i32 + w1 as i32,
            kp.position.y.round() as i32,
            green,
        );
    }

    // Draw match lines
    for (i, m) in result.matches.iter().enumerate() {
        let p1 = &result.keypoints1[m.source_idx].position;
        let p2 = &result.keypoints2[m.reference_idx].position;
        let x0 = p1.x.round() as i32;
        let y0 = p1.y.round() as i32;
        let x1 = p2.x.round() as i32 + w1 as i32;
        let y1 = p2.y.round() as i32;
        draw_line(&mut canvas, x0, y0, x1, y1, match_color(i));
    }

    // Save output
    canvas.save(&output_path).unwrap_or_else(|e| {
        eprintln!("Error: failed to save {}: {e}", output_path.display());
        process::exit(1);
    });
    println!("Saved match visualization to {}", output_path.display());
}
