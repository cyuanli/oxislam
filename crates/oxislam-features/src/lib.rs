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

/// Internal tracing helpers — no-ops when the `tracing` feature is disabled.
#[doc(hidden)]
pub mod trace {
    /// Create a tracing span and enter it. Returns a guard that exits the span on drop.
    /// When the `tracing` feature is disabled this is a zero-cost no-op.
    macro_rules! span {
        ($name:literal) => {{
            #[cfg(feature = "tracing")]
            let _span = tracing::info_span!($name).entered();
            #[cfg(not(feature = "tracing"))]
            let _span = ();
            _span
        }};
        ($name:literal, $($field:tt)*) => {{
            #[cfg(feature = "tracing")]
            let _span = tracing::info_span!($name, $($field)*).entered();
            #[cfg(not(feature = "tracing"))]
            let _span = ();
            _span
        }};
    }
    pub(crate) use span;
}
