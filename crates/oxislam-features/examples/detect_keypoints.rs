use std::path::PathBuf;
use std::process;
use std::time::Instant;

use clap::Parser;
use oxislam_features::detector::fast::FastDetector;
use oxislam_features::detector::harris::HarrisDetector;
use oxislam_features::traits::detector::KeypointDetector;
use oxislam_image::image::Image;
use oxislam_image::{ConvertTo, Gray, Rgb};
use oxislam_viz::drawing::draw_cross;

/// Detect keypoints in an image
#[derive(Parser, Debug)]
#[command(name = "detect_keypoints")]
#[command(about = "Keypoint detection example", long_about = None)]
struct Args {
    /// Path to input image
    input: PathBuf,

    /// Path to output image (defaults to <input>_keypoints.<ext>)
    output: Option<PathBuf>,

    /// Detector to use
    #[arg(short, long, default_value = "fast")]
    detector: String,
}

fn create_detector(name: &str) -> Box<dyn KeypointDetector<Gray<f32>>> {
    match name {
        "fast" => Box::new(FastDetector::default()),
        "harris" => Box::new(HarrisDetector::default()),
        _ => {
            eprintln!("Error: unknown detector '{name}'");
            eprintln!();
            eprintln!("Available detectors: fast, harris");
            process::exit(1);
        }
    }
}

fn main() {
    let args = Args::parse();

    // Output path
    let output_path = args.output.unwrap_or_else(|| {
        let stem = args
            .input
            .file_stem()
            .unwrap_or_else(|| {
                eprintln!("Error: cannot determine file stem from input path");
                process::exit(1);
            })
            .to_string_lossy();
        let ext = args
            .input
            .extension()
            .map(|e| e.to_string_lossy().into_owned())
            .unwrap_or_else(|| "png".to_owned());
        args.input.with_file_name(format!("{stem}_keypoints.{ext}"))
    });

    // Load image
    let img = image::open(&args.input).unwrap_or_else(|e| {
        eprintln!("Error: failed to open {}: {e}", args.input.display());
        process::exit(1);
    });
    let (w, h) = (img.width() as usize, img.height() as usize);
    println!("Loaded image: {w}x{h} from {}", args.input.display());

    // Convert to Gray<f32>
    let luma = img.to_luma8();
    let gray_u8: Image<Gray<u8>> = luma.into();
    let gray_f32: Image<Gray<f32>> = gray_u8.view().to();

    // Detect keypoints
    let detector = create_detector(&args.detector);
    println!("Using detector: {}", args.detector);
    let t0 = Instant::now();
    let keypoints = detector.detect(&gray_f32.view());
    let detect_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("Detected {} keypoints in {detect_ms:.1}ms", keypoints.len());

    // Visualize keypoints
    let rgb = img.to_rgb8();
    let mut canvas: Image<Rgb<u8>> = rgb.into();
    let green = Rgb::new(0, 255, 0);
    for keypoint in &keypoints {
        draw_cross(&mut canvas, keypoint.position, green, 2);
    }

    // Save output
    let out: image::RgbImage = (&canvas).into();
    out.save(&output_path).unwrap_or_else(|e| {
        eprintln!("Error: failed to save {}: {e}", output_path.display());
        process::exit(1);
    });
    println!("Saved annotated image to {}", output_path.display());
}
