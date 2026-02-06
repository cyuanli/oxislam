use oxislam_geometry::Point2;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A detected keypoint in an image.
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Keypoint {
    /// Position in the image (in pixels).
    pub position: Point2<f32>,
    /// Characteristic scale of the keypoint.
    pub scale: f32,
    /// Orientation of the keypoint (if available), in radians.
    pub orientation: Option<f32>,
    /// Detector response value (higher = more confident).
    pub response: f32,
}
