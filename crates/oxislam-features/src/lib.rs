//! Feature detection, description, and matching.
//!
//! Provides keypoint detectors, feature descriptors, and descriptor matchers.
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
pub mod orientation;
pub mod pyramid;

pub mod traits;

pub mod descriptor;
pub mod detector;
pub mod matcher;
pub mod pipeline;
