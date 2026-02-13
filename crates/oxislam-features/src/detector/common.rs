use std::ops::Range;

use oxislam_geometry::Point2;
use oxislam_image::image::{Image, ImageView};
use oxislam_image::parallel::{par_flat_map, par_row_collect};
use oxislam_image::{Gray, Grid2D, gaussian_3x3, sobel};

use crate::keypoint::Keypoint;

/// Smoothed gradient tensor components (Sxx, Syy, Sxy).
pub(crate) struct GradientTensors<T> {
    pub sxx: Image<Gray<T>>,
    pub syy: Image<Gray<T>>,
    pub sxy: Image<Gray<T>>,
}

/// Suppress non-maximum responses in a 3x3 neighborhood above a threshold.
pub(crate) fn non_maximum_suppression(
    response: &Grid2D<f32>,
    threshold: f32,
    x_range: Range<usize>,
    y_range: Range<usize>,
) -> Vec<Keypoint> {
    let w = response.width();
    let h = response.height();

    let is_local_max = |x: usize, y: usize, r: f32| -> bool {
        (x == 0 || y == 0 || r > *response.get(x - 1, y - 1))
            && (y == 0 || r > *response.get(x, y - 1))
            && (x == w - 1 || y == 0 || r > *response.get(x + 1, y - 1))
            && (x == 0 || r > *response.get(x - 1, y))
            && (x == w - 1 || r > *response.get(x + 1, y))
            && (x == 0 || y == h - 1 || r > *response.get(x - 1, y + 1))
            && (y == h - 1 || r > *response.get(x, y + 1))
            && (x == w - 1 || y == h - 1 || r > *response.get(x + 1, y + 1))
    };

    let extract_row = |y: usize| -> Vec<Keypoint> {
        x_range
            .clone()
            .filter_map(|x| {
                let r = *response.get(x, y);
                (r > threshold && is_local_max(x, y, r)).then(|| Keypoint {
                    position: Point2::new(x as f32, y as f32),
                    scale: 1.0,
                    orientation: None,
                    response: r,
                })
            })
            .collect()
    };

    par_flat_map(y_range, extract_row)
}

/// Compute smoothed gradient tensor components (Sxx, Syy, Sxy) for an image.
///
/// Applies Sobel to get gradients, squares/cross-multiplies, then smooths
/// each component with a 3x3 Gaussian.
pub(crate) fn gradient_tensors(image: &ImageView<Gray<f32>>) -> GradientTensors<f32> {
    let (ix, iy) = sobel(image);
    let ix2 = &ix * &ix;
    let iy2 = &iy * &iy;
    let ixiy = &ix * &iy;
    GradientTensors {
        sxx: gaussian_3x3(&ix2.view()),
        syy: gaussian_3x3(&iy2.view()),
        sxy: gaussian_3x3(&ixiy.view()),
    }
}

/// Compute the Harris corner response from structure tensor components.
///
/// Response = det - k * trace², where det = sxx*syy - sxy² and trace = sxx + syy.
pub(crate) fn harris_response(sxx: f32, syy: f32, sxy: f32, k: f32) -> f32 {
    let det = sxx * syy - sxy * sxy;
    let trace = sxx + syy;
    det - k * trace * trace
}

/// Compute a full Harris response map from gradient tensor images.
pub(crate) fn harris_response_map(
    sxx: &ImageView<Gray<f32>>,
    syy: &ImageView<Gray<f32>>,
    sxy: &ImageView<Gray<f32>>,
    k: f32,
) -> Grid2D<f32> {
    let w = sxx.width();
    let h = sxx.height();
    let data = par_row_collect(w, h, |x, y| {
        harris_response(sxx.get(x, y).value, syy.get(x, y).value, sxy.get(x, y).value, k)
    });
    Grid2D::new(w, h, w, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response_grid(w: usize, h: usize, data: &[f32]) -> Grid2D<f32> {
        Grid2D::new(w, h, w, data.to_vec())
    }

    #[test]
    fn single_peak() {
        #[rustfmt::skip]
        let grid = response_grid(3, 3, &[
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
        ]);

        let kps = non_maximum_suppression(&grid, 0.0, 0..3, 0..3);
        assert_eq!(kps.len(), 1);
        assert_eq!(kps[0].position, Point2::new(1.0, 1.0));
    }

    #[test]
    fn adjacent_peaks_higher_wins() {
        #[rustfmt::skip]
        let grid = response_grid(5, 1, &[
            0.0, 2.0, 5.0, 3.0, 0.0,
        ]);

        let kps = non_maximum_suppression(&grid, 0.0, 0..5, 0..1);
        assert_eq!(kps.len(), 1);
        assert_eq!(kps[0].position.x, 2.0);
    }

    #[test]
    fn peak_at_corner() {
        #[rustfmt::skip]
        let grid = response_grid(3, 3, &[
            5.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]);

        let kps = non_maximum_suppression(&grid, 0.0, 0..3, 0..3);
        assert_eq!(kps.len(), 1);
        assert_eq!(kps[0].position, Point2::new(0.0, 0.0));
    }

    #[test]
    fn threshold_filters_weak_responses() {
        #[rustfmt::skip]
        let grid = response_grid(3, 1, &[0.0, 0.5, 0.0]);

        let kps = non_maximum_suppression(&grid, 1.0, 0..3, 0..1);
        assert!(kps.is_empty());
    }
}
