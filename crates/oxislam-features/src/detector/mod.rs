use std::ops::Range;

use oxislam_geometry::Point2;
use oxislam_image::Gray;
use oxislam_image::image::Image;
use oxislam_image::parallel::par_flat_map;

use crate::keypoint::Keypoint;

pub mod fast;
pub mod harris;

pub use fast::FastDetector;
pub use harris::HarrisDetector;

/// Suppress non-maximum responses in a 3x3 neighborhood above a threshold.
fn non_maximum_suppression(
    response: &Image<Gray<f32>>,
    threshold: f32,
    x_range: Range<usize>,
    y_range: Range<usize>,
    coord_offset: f32,
) -> Vec<Keypoint> {
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
        x_range
            .clone()
            .filter_map(|x| {
                let r = response.get(x, y).value;
                (r > threshold && is_local_max(x, y, r)).then(|| Keypoint {
                    position: Point2::new(x as f32 + coord_offset, y as f32 + coord_offset),
                    scale: 1.0,
                    orientation: None,
                    response: r,
                })
            })
            .collect()
    };

    par_flat_map(y_range, extract_row)
}
