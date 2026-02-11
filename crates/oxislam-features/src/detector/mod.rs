use std::ops::Range;

use oxislam_geometry::Point2;
use oxislam_image::Grid2D;
use oxislam_image::parallel::par_flat_map;

use crate::keypoint::Keypoint;

pub mod fast;
pub mod harris;

pub use fast::FastDetector;
pub use harris::HarrisDetector;

/// Suppress non-maximum responses in a 3x3 neighborhood above a threshold.
fn non_maximum_suppression(
    response: &Grid2D<f32>,
    threshold: f32,
    x_range: Range<usize>,
    y_range: Range<usize>,
    coord_offset: f32,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a response Grid2D from a row-major slice.
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

        let kps = non_maximum_suppression(&grid, 0.0, 0..3, 0..3, 0.0);
        assert_eq!(kps.len(), 1);
        assert_eq!(kps[0].position, Point2::new(1.0, 1.0));
    }

    #[test]
    fn adjacent_peaks_higher_wins() {
        #[rustfmt::skip]
        let grid = response_grid(5, 1, &[
            0.0, 2.0, 5.0, 3.0, 0.0,
        ]);

        let kps = non_maximum_suppression(&grid, 0.0, 0..5, 0..1, 0.0);
        assert_eq!(kps.len(), 1);
        assert_eq!(kps[0].position.x, 2.0);
    }

    #[test]
    fn two_separated_peaks() {
        #[rustfmt::skip]
        let grid = response_grid(7, 1, &[
            0.0, 5.0, 0.0, 0.0, 0.0, 3.0, 0.0,
        ]);

        let kps = non_maximum_suppression(&grid, 0.0, 0..7, 0..1, 0.0);
        assert_eq!(kps.len(), 2);
    }

    #[test]
    fn peak_at_corner() {
        #[rustfmt::skip]
        let grid = response_grid(3, 3, &[
            5.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]);

        let kps = non_maximum_suppression(&grid, 0.0, 0..3, 0..3, 0.0);
        assert_eq!(kps.len(), 1);
        assert_eq!(kps[0].position, Point2::new(0.0, 0.0));
    }

    #[test]
    fn coord_offset_applied() {
        #[rustfmt::skip]
        let grid = response_grid(1, 1, &[1.0]);

        let kps = non_maximum_suppression(&grid, 0.0, 0..1, 0..1, 2.5);
        assert_eq!(kps.len(), 1);
        assert_eq!(kps[0].position, Point2::new(2.5, 2.5));
    }

    #[test]
    fn threshold_filters_weak_responses() {
        #[rustfmt::skip]
        let grid = response_grid(3, 1, &[0.0, 0.5, 0.0]);

        let kps = non_maximum_suppression(&grid, 1.0, 0..3, 0..1, 0.0);
        assert!(kps.is_empty());
    }
}
