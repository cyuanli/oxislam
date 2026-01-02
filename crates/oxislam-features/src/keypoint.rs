use oxislam_geometry::Point2;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Keypoint {
    pub position: Point2<f32>,
    pub scale: f32,
    pub orientation: Option<f32>,
    pub response: f32,
}
