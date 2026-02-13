use std::path::PathBuf;
use std::process;
use std::time::Instant;

use clap::Parser;
use oxislam_features::descriptor::brief::{
    BriefExtractor128, BriefExtractor256, BriefExtractor512,
};
use oxislam_features::descriptor::patch::PatchExtractor;
use oxislam_features::detector::fast::FastDetector;
use oxislam_features::detector::harris::HarrisDetector;
use oxislam_features::keypoint::Keypoint;
use oxislam_features::matcher::brute_force::BruteForceMatcher;
use oxislam_features::matcher::distance::{Hamming, L2Squared};
use oxislam_features::pipeline::orb::OrbPipeline;
use oxislam_features::traits::descriptor::DescriptorExtractor;
use oxislam_features::traits::detector::KeypointDetector;
use oxislam_features::traits::matcher::{DescriptorMatch, DescriptorMatcher};
use oxislam_features::traits::pipeline::FeaturePipeline;
use oxislam_geometry::Point2;
use oxislam_image::image::{Image, ImageView};
use oxislam_image::{ConvertTo, Gray, Rgb, gaussian_3x3};
use oxislam_viz::canvas::side_by_side;
use oxislam_viz::color::distinct_color;
use oxislam_viz::drawing::{draw_cross, draw_line};

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

    /// Detector to use: fast, harris, orb
    #[arg(short, long, default_value = "orb")]
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
    let gray_u8: Image<Gray<u8>> = luma.into();
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
    println!("Detected {} + {} keypoints in {ms:.1}ms ({})", kps1.len(), kps2.len(), detector);
    (kps1, kps2)
}

fn orb_pipeline(img1: &ImageView<Gray<f32>>, img2: &ImageView<Gray<f32>>) -> MatchResult {
    let pipeline = OrbPipeline::default();

    let t0 = Instant::now();
    let feats1 = pipeline.extract(img1);
    let feats2 = pipeline.extract(img2);
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("ORB pipeline: {} + {} features in {ms:.1}ms", feats1.len(), feats2.len());

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
    println!("Extracted {} + {} descriptors in {ms:.1}ms", feats1.len(), feats2.len(),);

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
    println!("Extracted {} + {} descriptors in {ms:.1}ms", feats1.len(), feats2.len(),);

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
    let result = if args.detector == "orb" {
        orb_pipeline(&gray1.view(), &gray2.view())
    } else {
        let (kps1, kps2) = detect(&args.detector, &gray1.view(), &gray2.view());
        describe_and_match(&args.descriptor, &gray1.view(), &gray2.view(), kps1, kps2)
    };

    // Create side-by-side visualization
    let ox_rgb1: Image<Rgb<u8>> = rgb1.into();
    let ox_rgb2: Image<Rgb<u8>> = rgb2.into();
    let mut canvas = side_by_side(&ox_rgb1.view(), &ox_rgb2.view());

    // Draw keypoints
    let green = Rgb::new(0, 255, 0);
    for kp in &result.keypoints1 {
        draw_cross(&mut canvas, kp.position, green, 2);
    }
    for kp in &result.keypoints2 {
        let shifted = Point2::new(kp.position.x + w1 as f32, kp.position.y);
        draw_cross(&mut canvas, shifted, green, 2);
    }

    // Draw match lines
    for (i, m) in result.matches.iter().enumerate() {
        let p1 = result.keypoints1[m.source_idx].position;
        let p2 = result.keypoints2[m.reference_idx].position;
        let p2_shifted = Point2::new(p2.x + w1 as f32, p2.y);
        draw_line(&mut canvas, p1, p2_shifted, distinct_color(i));
    }

    // Save output
    let out: image::RgbImage = (&canvas).into();
    out.save(&output_path).unwrap_or_else(|e| {
        eprintln!("Error: failed to save {}: {e}", output_path.display());
        process::exit(1);
    });
    println!("Saved match visualization to {}", output_path.display());
}
