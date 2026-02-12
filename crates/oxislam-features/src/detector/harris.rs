use oxislam_image::image::{Image, ImageView};
use oxislam_image::parallel::par_row_collect;
use oxislam_image::{Gray, Grid2D, gaussian_3x3, sobel};

use super::non_maximum_suppression;
use crate::keypoint::Keypoint;
use crate::traits::detector::KeypointDetector;

const DEFAULT_K: f32 = 0.04;
const DEFAULT_ALPHA: f32 = 0.01;
const DEFAULT_MIN_THRESHOLD: f32 = 1e-6;
// Minimum image size: sobel (3x3) shrinks by 2, gaussian (3x3) shrinks by 2 more
const MIN_IMAGE_SIZE: usize = 5;
// Coordinate offset from response image to original: sobel (1) + gaussian (1)
const COORD_OFFSET: f32 = 2.0;

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

    fn response_at(
        &self,
        sxx: &ImageView<Gray<f32>>,
        syy: &ImageView<Gray<f32>>,
        sxy: &ImageView<Gray<f32>>,
        x: usize,
        y: usize,
    ) -> f32 {
        let xx = sxx.get(x, y).value;
        let yy = syy.get(x, y).value;
        let xy = sxy.get(x, y).value;

        let det = xx * yy - xy * xy;
        let trace = xx + yy;

        det - self.k * trace * trace
    }

    fn response_map(
        &self,
        sxx: &ImageView<Gray<f32>>,
        syy: &ImageView<Gray<f32>>,
        sxy: &ImageView<Gray<f32>>,
    ) -> Grid2D<f32> {
        let w = sxx.width();
        let h = sxx.height();

        let data = par_row_collect(w, h, |x, y| self.response_at(sxx, syy, sxy, x, y));

        Grid2D::new(w, h, w, data)
    }

    fn compute_gradient_tensors(
        image: &ImageView<Gray<f32>>,
    ) -> (Image<Gray<f32>>, Image<Gray<f32>>, Image<Gray<f32>>) {
        let (ix, iy) = sobel(image);
        let ix2 = &ix * &ix;
        let iy2 = &iy * &iy;
        let ixiy = &ix * &iy;
        let sxx = gaussian_3x3(&ix2.view());
        let syy = gaussian_3x3(&iy2.view());
        let sxy = gaussian_3x3(&ixiy.view());
        (sxx, syy, sxy)
    }
}

impl KeypointDetector<Gray<f32>> for HarrisDetector {
    fn detect(&self, image: &ImageView<Gray<f32>>) -> Vec<Keypoint> {
        if image.width() < MIN_IMAGE_SIZE || image.height() < MIN_IMAGE_SIZE {
            return Vec::new();
        }

        let (sxx, syy, sxy) = Self::compute_gradient_tensors(image);
        let response = self.response_map(&sxx.view(), &syy.view(), &sxy.view());
        let max_r = response.view().iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v));
        let threshold = self.min_threshold.max(self.alpha * max_r);
        let w = response.width();
        let h = response.height();

        non_maximum_suppression(&response, threshold, 0..w, 0..h, COORD_OFFSET)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corner_image() -> Image<Gray<f32>> {
        // 10x10 image: 5x5 white square in top-left, rest black
        let mut data = vec![Gray::new(0.0f32); 10 * 10];
        for y in 0..5 {
            for x in 0..5 {
                data[y * 10 + x] = Gray::new(1.0);
            }
        }
        Image::new(10, 10, 10, data)
    }

    #[test]
    fn harris_detects_corner() {
        let img = corner_image();
        let detector = HarrisDetector::default();
        let keypoints = detector.detect(&img.view());

        assert_eq!(keypoints.len(), 1, "Harris should detect exactly one keypoint");

        let kp = &keypoints[0];
        let dx = kp.position.x - 4.0;
        let dy = kp.position.y - 4.0;
        let dist = (dx * dx + dy * dy).sqrt();

        assert!(dist <= 1.0, "Expected keypoint within 1 pixel of (4, 4), got distance {dist}");
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
        let data = vec![Gray::new(1.0f32); 4 * 4];
        let img = Image::new(4, 4, 4, data);

        let detector = HarrisDetector::default();
        let keypoints = detector.detect(&img.view());

        assert!(keypoints.is_empty(), "image smaller than 5x5 should return no keypoints");
    }
}
