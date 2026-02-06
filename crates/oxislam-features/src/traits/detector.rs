use oxislam_image::image::ImageView;

use crate::keypoint::Keypoint;

/// Detects keypoints in an image.
pub trait KeypointDetector<P> {
    /// Detect keypoints in the given image.
    fn detect(&self, image: &ImageView<P>) -> Vec<Keypoint>;
}
