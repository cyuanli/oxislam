use oxislam_image::image::ImageView;
use oxislam_image::pixel::Gray;

use super::common::harris_response_local;
use crate::detector::fast::FastDetector;
use crate::keypoint::Keypoint;
use crate::trace::span;
use crate::traits::detector::KeypointDetector;

const DEFAULT_HARRIS_K: f32 = 0.04;

/// ORB keypoint detector (FAST + Harris scoring).
///
/// Detects corners with FAST, then re-scores them using the Harris response.
///
/// For the full ORB pipeline (multi-scale detection, orientation, truncation,
/// and BRIEF descriptors), use [`OrbPipeline`](crate::pipeline::orb::OrbPipeline).
pub struct OrbDetector {
    pub fast: FastDetector,
    pub harris_k: f32,
}

impl Default for OrbDetector {
    fn default() -> Self { Self { fast: FastDetector::default(), harris_k: DEFAULT_HARRIS_K } }
}

impl KeypointDetector<Gray<f32>> for OrbDetector {
    fn detect(&self, image: &ImageView<Gray<f32>>) -> Vec<Keypoint> {
        let mut keypoints = self.fast.detect(image);

        {
            let _s = span!("harris_rescore", keypoints = keypoints.len());
            for kp in &mut keypoints {
                let x = kp.position.x as usize;
                let y = kp.position.y as usize;
                kp.response = harris_response_local(image, x, y, self.harris_k);
            }
        }

        keypoints
    }
}

#[cfg(test)]
mod tests {
    use oxislam_image::image::Image;

    use super::*;

    /// 20x20 dark image with a bright pixel — ORB should re-score with Harris.
    #[test]
    fn orb_rescores_with_harris() {
        let mut data = vec![Gray::new(0.0f32); 20 * 20];
        data[10 * 20 + 10] = Gray::new(1.0);
        let img = Image::new(20, 20, 20, data);

        let detector = OrbDetector::default();
        let keypoints = detector.detect(&img.view());

        // Should detect the bright spot with a Harris-based response
        assert!(!keypoints.is_empty(), "ORB should detect keypoints");
    }
}
