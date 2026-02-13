use std::ops::Range;

use oxislam_geometry::Point2;
use oxislam_image::filter::{GAUSSIAN_3X3, SOBEL_X, SOBEL_Y};
use oxislam_image::image::{Image, ImageView};
use oxislam_image::parallel::{par_flat_map, par_row_collect};
use oxislam_image::{Gray, Grid2D, gaussian, sobel};

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
    let _span = crate::trace::span!("nms");
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
    let _span = crate::trace::span!("gradient_tensors");
    let (ix, iy) = sobel(image);
    let ix2 = &ix * &ix;
    let iy2 = &iy * &iy;
    let ixiy = &ix * &iy;
    GradientTensors {
        sxx: gaussian::<3>(&ix2.view()),
        syy: gaussian::<3>(&iy2.view()),
        sxy: gaussian::<3>(&ixiy.view()),
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

/// Compute Harris response locally at a single pixel using a 5×5 neighborhood.
///
/// Equivalent to computing full-image Sobel → square/cross → Gaussian smoothing,
/// but only at the single pixel (cx, cy).
///
/// # Panics
///
/// Panics if (cx, cy) is within 2 pixels of any image border.
pub(crate) fn harris_response_local(
    image: &ImageView<Gray<f32>>,
    cx: usize,
    cy: usize,
    k: f32,
) -> f32 {
    assert!(cx >= 2 && cy >= 2, "harris_response_local: cx={cx}, cy={cy} must be >= 2");
    assert!(
        cx < image.width() - 2 && cy < image.height() - 2,
        "harris_response_local: ({cx}, {cy}) too close to border for {}x{} image",
        image.width(),
        image.height(),
    );

    // Read 5×5 patch centered at (cx, cy).
    let mut patch = [[0.0f32; 5]; 5];
    for py in 0..5 {
        for px in 0..5 {
            patch[py][px] = image.get(cx + px - 2, cy + py - 2).value;
        }
    }

    let mut sxx = 0.0f32;
    let mut syy = 0.0f32;
    let mut sxy = 0.0f32;

    for gy in 0..3usize {
        for gx in 0..3usize {
            // Sobel at patch position (1 + gx, 1 + gy).
            let mut ix = 0.0f32;
            let mut iy = 0.0f32;
            for sy in 0..3usize {
                for sx in 0..3usize {
                    let p = patch[gy + sy][gx + sx];
                    ix += SOBEL_X[sy][sx] * p;
                    iy += SOBEL_Y[sy][sx] * p;
                }
            }
            let w = GAUSSIAN_3X3[gy][gx];
            sxx += w * ix * ix;
            syy += w * iy * iy;
            sxy += w * ix * iy;
        }
    }

    harris_response(sxx, syy, sxy, k)
}

/// Compute a full Harris response map from gradient tensor images.
pub(crate) fn harris_response_map(
    sxx: &ImageView<Gray<f32>>,
    syy: &ImageView<Gray<f32>>,
    sxy: &ImageView<Gray<f32>>,
    k: f32,
) -> Grid2D<f32> {
    let _span = crate::trace::span!("harris_response_map");
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

    #[test]
    fn harris_local_matches_full_image() {
        use oxislam_image::image::Image;

        // Create a 20×20 test image with varied content.
        let w = 20usize;
        let h = 20usize;
        let data: Vec<Gray<f32>> = (0..w * h)
            .map(|i| {
                let x = i % w;
                let y = i / w;
                Gray::new(((x * 7 + y * 13 + x * y) % 256) as f32 / 255.0)
            })
            .collect();
        let img = Image::new(w, h, w, data);
        let view = img.view();

        let gt = gradient_tensors(&view);
        let k = 0.04f32;

        // Compare at all interior pixels where the local version is valid (2px border).
        for y in 2..h - 2 {
            for x in 2..w - 2 {
                let full = harris_response(
                    gt.sxx.get(x, y).value,
                    gt.syy.get(x, y).value,
                    gt.sxy.get(x, y).value,
                    k,
                );
                let local = harris_response_local(&view, x, y, k);
                assert!(
                    (full - local).abs() < 1e-4,
                    "Mismatch at ({x}, {y}): full={full}, local={local}",
                );
            }
        }
    }
}
