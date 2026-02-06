use std::path::PathBuf;
use std::process;
use std::time::Instant;

use clap::Parser;
use oxislam_features::descriptor::patch::PatchExtractor;
use oxislam_features::detector::harris::HarrisDetector;
use oxislam_features::traits::descriptor::DescriptorExtractor;
use oxislam_features::traits::detector::KeypointDetector;
use oxislam_image::image::Image;
use oxislam_image::{ConvertTo, Gray};

/// Detect features in an image
#[derive(Parser, Debug)]
#[command(name = "detect_features")]
#[command(about = "Feature detection example", long_about = None)]
struct Args {
    /// Path to input image
    input: PathBuf,

    /// Path to output image (defaults to <input>_features.<ext>)
    output: Option<PathBuf>,

    /// Detector to use
    #[arg(short, long, default_value = "harris")]
    detector: String,
}

fn create_detector(name: &str) -> Box<dyn KeypointDetector<Gray<f32>>> {
    match name {
        "harris" => Box::new(HarrisDetector::default()),
        _ => {
            eprintln!("Error: unknown detector '{name}'");
            eprintln!();
            eprintln!("Available detectors: harris");
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
        args.input.with_file_name(format!("{stem}_features.{ext}"))
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
    let raw = luma.into_raw();
    let gray_u8: Image<Gray<u8>> = Image::from_raw(w, h, w, raw);
    let gray_f32: Image<Gray<f32>> = gray_u8.view().to();

    // Detect keypoints
    let detector = create_detector(&args.detector);
    println!("Using detector: {}", args.detector);
    let t0 = Instant::now();
    let keypoints = detector.detect(&gray_f32.view());
    let detect_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("Detected {} keypoints in {detect_ms:.1}ms", keypoints.len());

    // Extract descriptors
    let extractor = PatchExtractor::<7, 49>::default();
    let kp_count = keypoints.len();
    let t1 = Instant::now();
    let features = extractor.describe(&gray_f32.view(), keypoints);
    let desc_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let discarded = kp_count - features.len();
    println!(
        "Extracted {} descriptors in {desc_ms:.1}ms ({discarded} near-border keypoints discarded)",
        features.len()
    );

    // Visualize keypoints
    let mut rgb = img.to_rgb8();
    let green = image::Rgb([0u8, 255, 0]);
    let arm: i32 = 2;
    for feat in &features {
        let cx = feat.keypoint.position.x.round() as i32;
        let cy = feat.keypoint.position.y.round() as i32;
        for d in -arm..=arm {
            let px = cx + d;
            let py = cy + d;
            if px >= 0 && (px as usize) < w && cy >= 0 && (cy as usize) < h {
                rgb.put_pixel(px as u32, cy as u32, green);
            }
            if cx >= 0 && (cx as usize) < w && py >= 0 && (py as usize) < h {
                rgb.put_pixel(cx as u32, py as u32, green);
            }
        }
    }

    // Save output
    rgb.save(&output_path).unwrap_or_else(|e| {
        eprintln!("Error: failed to save {}: {e}", output_path.display());
        process::exit(1);
    });
    println!("Saved annotated image to {}", output_path.display());
}
