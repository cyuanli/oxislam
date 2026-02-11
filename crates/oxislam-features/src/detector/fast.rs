use oxislam_image::Grid2D;
use oxislam_image::image::ImageView;
use oxislam_image::parallel::par_row_collect;
use oxislam_image::Gray;

use super::non_maximum_suppression;
use crate::keypoint::Keypoint;
use crate::traits::detector::KeypointDetector;

const DEFAULT_N: u8 = 9;
const DEFAULT_THRESHOLD: f32 = 0.08;
// Minimum image size: Circle of radius 3
const MIN_IMAGE_SIZE: usize = 7;

#[rustfmt::skip]
const CIRCLE_OFFSETS: [(isize, isize); 16] = [
    ( 0, -3), ( 1, -3), ( 2, -2), ( 3, -1),
    ( 3,  0), ( 3,  1), ( 2,  2), ( 1,  3),
    ( 0,  3), (-1,  3), (-2,  2), (-3,  1),
    (-3,  0), (-3, -1), (-2, -2), (-1, -3),
];

/// FAST (Features from Accelerated Segment Test) corner detector.
#[derive(Debug, Clone)]
pub struct FastDetector {
    /// Number of contiguous pixels on the Bresenham circle that must all be
    /// brighter or darker than the center (1..=16).
    pub n: u8,
    /// Intensity difference threshold (0.0..=1.0).
    pub threshold: f32,
}

impl Default for FastDetector {
    fn default() -> Self { Self { n: DEFAULT_N, threshold: DEFAULT_THRESHOLD } }
}

impl FastDetector {
    /// Create a new FAST detector with the given parameters.
    pub fn new(n: u8, threshold: f32) -> Self {
        assert!((1..=16).contains(&n), "n must be in 1..=16, got {n}");
        assert!(
            (0.0..=1.0).contains(&threshold),
            "threshold must be in 0.0..=1.0, got {threshold}"
        );
        Self { n, threshold }
    }

    fn sample_circle(image: &ImageView<Gray<f32>>, x: usize, y: usize) -> [f32; 16] {
        let x = x as isize;
        let y = y as isize;
        let mut circle = [0.0f32; 16];
        for (i, &(dx, dy)) in CIRCLE_OFFSETS.iter().enumerate() {
            circle[i] = image.get((x + dx) as usize, (y + dy) as usize).value;
        }
        circle
    }

    fn quick_reject(&self, circle: &[f32; 16], high: f32, low: f32) -> bool {
        if self.n >= 8 && (low..=high).contains(&circle[0]) && (low..=high).contains(&circle[8]) {
            return true;
        }

        let min_cardinal = self.n as usize / 4;

        let brighter = [circle[0] > high, circle[4] > high, circle[8] > high, circle[12] > high];
        let darker = [circle[0] < low, circle[4] < low, circle[8] < low, circle[12] < low];

        let n_bright: usize = brighter.iter().filter(|&&b| b).count();
        let n_dark: usize = darker.iter().filter(|&&b| b).count();

        n_bright < min_cardinal && n_dark < min_cardinal
    }

    fn arc_score<F>(&self, circle: &[f32; 16], mut diff: F) -> f32
    where
        F: FnMut(f32) -> f32,
    {
        let mut max_score: f32 = 0.0;
        let mut curr_score: f32 = 0.0;
        let mut run_len: usize = 0;

        for i in 0..32 {
            if run_len >= 16 {
                break;
            }
            let val = circle[i % 16];
            let d = diff(val);

            if d > 0.0 {
                curr_score += d;
                run_len += 1;
                max_score = max_score.max(curr_score);
            } else {
                curr_score = 0.0;
                run_len = 0;
            }
        }

        max_score
    }

    fn response_at(&self, image: &ImageView<Gray<f32>>, x: usize, y: usize) -> f32 {
        if x < 3 || y < 3 || x >= image.width() - 3 || y >= image.height() - 3 {
            return 0.0;
        }

        let center = image.get(x, y).value;
        let circle = Self::sample_circle(image, x, y);

        let high = center + self.threshold;
        let low = center - self.threshold;

        if self.quick_reject(&circle, high, low) {
            return 0.0;
        }

        let mut brighter: u8 = 0;
        let mut darker: u8 = 0;

        for i in 0..(16 + self.n as usize) {
            let val = circle[i % 16];

            if val > high {
                brighter += 1;
                darker = 0;
            } else if val < low {
                darker += 1;
                brighter = 0;
            } else {
                brighter = 0;
                darker = 0;
            }

            if brighter >= self.n {
                return self.arc_score(&circle, |v| v - high);
            }
            if darker >= self.n {
                return self.arc_score(&circle, |v| low - v);
            }
        }

        0.0
    }

    fn response_map(&self, image: &ImageView<Gray<f32>>) -> Grid2D<f32> {
        let w = image.width();
        let h = image.height();

        let data = par_row_collect(w, h, |x, y| self.response_at(image, x, y));

        Grid2D::new(w, h, w, data)
    }
}

impl KeypointDetector<Gray<f32>> for FastDetector {
    fn detect(&self, image: &ImageView<Gray<f32>>) -> Vec<Keypoint> {
        if image.width() < MIN_IMAGE_SIZE || image.height() < MIN_IMAGE_SIZE {
            return Vec::new();
        }

        let response = self.response_map(image);
        let w = response.width();
        let h = response.height();
        non_maximum_suppression(&response, 0.0, 3..w - 3, 3..h - 3, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use oxislam_image::image::Image;

    use super::*;

    #[test]
    fn fast_detects_bright_spot() {
        // 20x20 dark image with a single bright pixel at (10,10).
        // The radius-3 circle lies entirely in the dark region,
        // giving 16 contiguous darker arc pixels.
        let mut data = vec![Gray::new(0.0f32); 20 * 20];
        data[10 * 20 + 10] = Gray::new(1.0);
        let img = Image::new(20, 20, 20, data);

        let detector = FastDetector::default();
        let keypoints = detector.detect(&img.view());

        assert_eq!(keypoints.len(), 1, "FAST should detect exactly one keypoint");
        assert!(
            (keypoints[0].position.x - 10.0).abs() < f32::EPSILON
                && (keypoints[0].position.y - 10.0).abs() < f32::EPSILON,
            "expected keypoint at (10, 10)"
        );
    }

    #[test]
    fn fast_no_detections_on_uniform() {
        let data = vec![Gray::new(0.5f32); 20 * 20];
        let img = Image::new(20, 20, 20, data);

        let detector = FastDetector::default();
        let keypoints = detector.detect(&img.view());

        assert!(keypoints.is_empty(), "uniform image should produce no keypoints");
    }

    #[test]
    fn fast_detects_arc_wrapping_around_circle() {
        // Place a bright pixel at (10,10) on a dark background,
        // then darken some of the circle positions so the contiguous bright arc
        // must wrap from index 15 back to index 0 to reach n=9.
        //
        // Circle offsets (index → relative position):
        //   0:( 0,-3)  1:( 1,-3)  2:( 2,-2)  3:( 3,-1)
        //   4:( 3, 0)  5:( 3, 1)  6:( 2, 2)  7:( 1, 3)
        //   8:( 0, 3)  9:(-1, 3) 10:(-2, 2) 11:(-3, 1)
        //  12:(-3, 0) 13:(-3,-1) 14:(-2,-2) 15:(-1,-3)
        //
        // We want indices 12..=15 and 0..=4 to be dark (= 9 contiguous),
        // while indices 5..=11 match the center (= no contrast, break the arc).
        let center = (10usize, 10usize);
        let center_val: f32 = 0.5;
        let dark: f32 = 0.0;

        let mut data = vec![Gray::new(center_val); 20 * 20];
        data[center.1 * 20 + center.0] = Gray::new(center_val);

        // Make indices 12..=15, 0..=4 dark (contrast from center)
        for &i in &[12, 13, 14, 15, 0, 1, 2, 3, 4] {
            let (dx, dy) = CIRCLE_OFFSETS[i];
            let px = (center.0 as isize + dx) as usize;
            let py = (center.1 as isize + dy) as usize;
            data[py * 20 + px] = Gray::new(dark);
        }

        let img = Image::new(20, 20, 20, data);
        let detector = FastDetector::default(); // n=9, threshold=0.08

        let keypoints = detector.detect(&img.view());
        let found = keypoints.iter().any(|kp| {
            (kp.position.x - 10.0).abs() < f32::EPSILON
                && (kp.position.y - 10.0).abs() < f32::EPSILON
        });
        assert!(found, "wrapping arc at (10,10) should be detected");
    }

    #[test]
    fn fast_too_small_image() {
        let data = vec![Gray::new(1.0f32); 6 * 6];
        let img = Image::new(6, 6, 6, data);

        let detector = FastDetector::default();
        let keypoints = detector.detect(&img.view());

        assert!(keypoints.is_empty(), "image smaller than 7x7 should return no keypoints");
    }
}
