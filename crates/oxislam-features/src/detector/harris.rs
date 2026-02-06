use oxislam_geometry::Point2;
use oxislam_image::image::{Image, ImageView};
use oxislam_image::parallel::{par_flat_map, par_row_collect};
use oxislam_image::{Gray, gaussian_3x3, sobel};

use crate::keypoint::Keypoint;
use crate::traits::detector::KeypointDetector;

const DEFAULT_K: f32 = 0.04;
const DEFAULT_ALPHA: f32 = 0.01;
const DEFAULT_MIN_THRESHOLD: f32 = 1e-6;
// Minimum image size: sobel (3x3) shrinks by 2, gaussian (3x3) shrinks by 2 more
const MIN_IMAGE_SIZE: usize = 5;
// Coordinate offset from response image to original: sobel (1) + gaussian (1)
const COORD_OFFSET: f32 = 2.0;

#[derive(Debug, Clone)]
pub struct HarrisDetector {
    pub k: f32,
    pub alpha: f32,
    pub min_threshold: f32,
}

impl HarrisDetector {
    pub fn new(k: f32, alpha: f32, min_threshold: f32) -> Self { Self { k, alpha, min_threshold } }
}

impl Default for HarrisDetector {
    fn default() -> Self {
        Self { k: DEFAULT_K, alpha: DEFAULT_ALPHA, min_threshold: DEFAULT_MIN_THRESHOLD }
    }
}

impl HarrisDetector {
    fn response_at(
        &self,
        sxx: &ImageView<Gray<f32>>,
        syy: &ImageView<Gray<f32>>,
        sxy: &ImageView<Gray<f32>>,
        x: usize,
        y: usize,
    ) -> Gray<f32> {
        let xx = sxx.get(x, y).value;
        let yy = syy.get(x, y).value;
        let xy = sxy.get(x, y).value;

        let det = xx * yy - xy * xy;
        let trace = xx + yy;

        Gray::new(det - self.k * trace * trace)
    }

    fn response_map(
        &self,
        sxx: &ImageView<Gray<f32>>,
        syy: &ImageView<Gray<f32>>,
        sxy: &ImageView<Gray<f32>>,
    ) -> Image<Gray<f32>> {
        let w = sxx.width();
        let h = sxx.height();

        let data = par_row_collect(w, h, |x, y| self.response_at(sxx, syy, sxy, x, y));

        Image::new(w, h, w, data)
    }
}

impl KeypointDetector<Gray<f32>> for HarrisDetector {
    fn detect(&self, image: &ImageView<Gray<f32>>) -> Vec<Keypoint> {
        if image.width() < MIN_IMAGE_SIZE || image.height() < MIN_IMAGE_SIZE {
            return Vec::new();
        }

        let (ix, iy) = sobel(image);
        let ix2 = &ix * &ix;
        let iy2 = &iy * &iy;
        let ixiy = &ix * &iy;
        let sxx = gaussian_3x3(&ix2.view());
        let syy = gaussian_3x3(&iy2.view());
        let sxy = gaussian_3x3(&ixiy.view());

        let response = self.response_map(&sxx.view(), &syy.view(), &sxy.view());
        let max_r = response.view().pixels().map(|p| p.value).fold(f32::NEG_INFINITY, f32::max);
        let threshold = self.min_threshold.max(self.alpha * max_r);

        let w = response.width();
        let h = response.height();

        let is_local_max = |x: usize, y: usize, r: f32| -> bool {
            (x == 0 || y == 0 || r > response.get(x - 1, y - 1).value)
                && (y == 0 || r > response.get(x, y - 1).value)
                && (x == w - 1 || y == 0 || r > response.get(x + 1, y - 1).value)
                && (x == 0 || r > response.get(x - 1, y).value)
                && (x == w - 1 || r > response.get(x + 1, y).value)
                && (x == 0 || y == h - 1 || r > response.get(x - 1, y + 1).value)
                && (y == h - 1 || r > response.get(x, y + 1).value)
                && (x == w - 1 || y == h - 1 || r > response.get(x + 1, y + 1).value)
        };

        let extract_row = |y: usize| -> Vec<Keypoint> {
            (0..w)
                .filter_map(|x| {
                    let r = response.get(x, y).value;
                    (r > threshold && is_local_max(x, y, r)).then(|| Keypoint {
                        position: Point2::new(x as f32 + COORD_OFFSET, y as f32 + COORD_OFFSET),
                        scale: 1.0,
                        orientation: None,
                        response: r,
                    })
                })
                .collect()
        };

        par_flat_map(0..h, extract_row)
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
}
