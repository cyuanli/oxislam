use oxislam_image::Gray;
use oxislam_image::image::ImageView;

use super::common::{harris_response_map, nms, structure_tensor};
use crate::keypoint::Keypoint;
use crate::traits::detector::KeypointDetector;

const DEFAULT_K: f32 = 0.04;
const DEFAULT_ALPHA: f32 = 0.01;
const DEFAULT_MIN_THRESHOLD: f32 = 1e-6;
const MIN_IMAGE_SIZE: usize = 3;

/// Harris corner detector.
#[derive(Debug, Clone)]
pub struct HarrisDetector {
    /// Sensitivity parameter for the Harris response (det - k * trace^2).
    pub k: f32,
    /// Fraction of the maximum response used as the detection threshold.
    pub alpha: f32,
    /// Absolute minimum detection threshold.
    pub min_threshold: f32,
}

impl Default for HarrisDetector {
    fn default() -> Self {
        Self { k: DEFAULT_K, alpha: DEFAULT_ALPHA, min_threshold: DEFAULT_MIN_THRESHOLD }
    }
}

impl HarrisDetector {
    /// Create a new Harris detector with the given parameters.
    pub fn new(k: f32, alpha: f32, min_threshold: f32) -> Self {
        assert!(k > 0.0, "k must be positive, got {k}");
        assert!((0.0..=1.0).contains(&alpha), "alpha must be in 0.0..=1.0, got {alpha}");
        assert!(min_threshold >= 0.0, "min_threshold must be non-negative, got {min_threshold}");
        Self { k, alpha, min_threshold }
    }
}

impl KeypointDetector<Gray<f32>> for HarrisDetector {
    fn detect(&self, image: &ImageView<Gray<f32>>) -> Vec<Keypoint> {
        let _span =
            crate::trace::span!("harris_detect", width = image.width(), height = image.height());
        if image.width() < MIN_IMAGE_SIZE || image.height() < MIN_IMAGE_SIZE {
            crate::trace::event!(
                tracing::Level::WARN,
                width = image.width(),
                height = image.height(),
                min = MIN_IMAGE_SIZE,
                "image too small for harris detection"
            );
            return Vec::new();
        }

        let gt = structure_tensor(image);
        let response = harris_response_map(&gt.sxx.view(), &gt.syy.view(), &gt.sxy.view(), self.k);
        let max_r = response.view().iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v));
        let threshold = self.min_threshold.max(self.alpha * max_r);
        crate::trace::event!(tracing::Level::DEBUG, max_response = max_r, threshold = threshold);
        let w = response.width();
        let h = response.height();

        nms(&response, threshold, 0..w, 0..h)
    }
}

#[cfg(test)]
mod tests {
    use oxislam_image::image::Image;

    use super::*;

    #[test]
    fn harris_detects_corner() {
        // 20x20 image with a 5x5 white square at (5,5)-(9,9): single L-corner at (9,9)
        let mut data = vec![Gray::new(0.0f32); 20 * 20];
        for y in 5..10 {
            for x in 5..10 {
                data[y * 20 + x] = Gray::new(1.0);
            }
        }
        let img = Image::new(20, 20, 20, data);
        let detector = HarrisDetector::default();
        let keypoints = detector.detect(&img.view());

        assert_eq!(keypoints.len(), 4, "Harris should detect 4 corners of the square");
    }

    #[test]
    fn harris_four_corners() {
        // 30x30 image with a 10x10 white rectangle at (10,10)-(19,19)
        // This creates 4 L-corners at (10,10), (19,10), (10,19), (19,19)
        let size = 30;
        let mut data = vec![Gray::new(0.0f32); size * size];
        for y in 10..20 {
            for x in 10..20 {
                data[y * size + x] = Gray::new(1.0);
            }
        }
        let img = Image::new(size, size, size, data);

        let detector = HarrisDetector::default();
        let keypoints = detector.detect(&img.view());

        let expected = [(10.0, 10.0), (19.0, 10.0), (10.0, 19.0), (19.0, 19.0)];
        assert_eq!(keypoints.len(), expected.len(), "expected 4 corners, got {}", keypoints.len());

        for (ex, ey) in &expected {
            let found = keypoints.iter().any(|kp| {
                let dx = kp.position.x - ex;
                let dy = kp.position.y - ey;
                (dx * dx + dy * dy).sqrt() <= 1.0
            });
            assert!(found, "expected a keypoint within 1 pixel of ({ex}, {ey})");
        }
    }

    #[test]
    fn harris_too_small_image() {
        let data = vec![Gray::new(1.0f32); 2 * 2];
        let img = Image::new(2, 2, 2, data);

        let detector = HarrisDetector::default();
        let keypoints = detector.detect(&img.view());

        assert!(keypoints.is_empty(), "image smaller than 3x3 should return no keypoints");
    }
}
