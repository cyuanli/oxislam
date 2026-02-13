use oxislam_geometry::Point2;
use oxislam_image::image::ImageView;

use crate::keypoint::Keypoint;
use crate::pyramid::Pyramid;

/// Detects keypoints in an image.
pub trait KeypointDetector<P> {
    /// Detect keypoints in the given image.
    fn detect(&self, image: &ImageView<P>) -> Vec<Keypoint>;

    /// Detect keypoints across all pyramid levels, mapping coordinates back to
    /// the base (level-0) scale.
    fn detect_multiscale(&self, pyramid: &Pyramid<P>) -> Vec<Keypoint> {
        let mut all = Vec::new();
        for level in 0..pyramid.num_levels() {
            let scale = pyramid.scale_at_level(level);
            let mut kps = self.detect(&pyramid.level(level).view());
            for kp in &mut kps {
                kp.position = Point2::new(kp.position.x * scale, kp.position.y * scale);
                kp.scale = scale;
            }
            all.extend(kps);
        }
        all
    }
}
