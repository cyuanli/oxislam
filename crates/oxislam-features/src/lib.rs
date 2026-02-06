//! Feature detection and description.
//!
//! Provides keypoint detectors (Harris, etc.) and feature descriptors (patch-based, etc.).
//!
//! # Example
//!
//! ```ignore
//! use oxislam_features::detector::harris::HarrisDetector;
//! use oxislam_features::traits::detector::KeypointDetector;
//! use oxislam_image::image::Image;
//!
//! let detector = HarrisDetector::default();
//! let keypoints = detector.detect(&image.view());
//! ```

pub mod feature;
pub mod keypoint;

pub mod traits;

pub mod descriptor;
pub mod detector;
