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
