use oxislam_image::image::ImageView;
use oxislam_image::pixel::Gray;

use super::common::{gradient_tensors, harris_response};
use crate::detector::fast::FastDetector;
use crate::keypoint::Keypoint;
use crate::orientation::intensity_centroid;
use crate::traits::detector::KeypointDetector;

const DEFAULT_ORIENTATION_RADIUS: usize = 15;
const DEFAULT_HARRIS_K: f32 = 0.04;
const DEFAULT_MAX_KEYPOINTS: usize = 500;

/// ORB keypoint detector (FAST + Harris scoring + orientation).
///
/// Detects corners with FAST, scores them using the Harris response,
/// keeps the top `max_keypoints`, then computes orientation via intensity centroid.
///
/// For multi-scale detection, build a [`Pyramid`](crate::pyramid::Pyramid)
/// externally and call [`detect_multiscale`](KeypointDetector::detect_multiscale).
pub struct OrbDetector {
    pub fast: FastDetector,
    pub orientation_radius: usize,
    pub harris_k: f32,
    pub max_keypoints: usize,
}

impl Default for OrbDetector {
    fn default() -> Self {
        Self {
            fast: FastDetector::default(),
            orientation_radius: DEFAULT_ORIENTATION_RADIUS,
            harris_k: DEFAULT_HARRIS_K,
            max_keypoints: DEFAULT_MAX_KEYPOINTS,
        }
    }
}

impl KeypointDetector<Gray<f32>> for OrbDetector {
    fn detect(&self, image: &ImageView<Gray<f32>>) -> Vec<Keypoint> {
        let mut keypoints = self.fast.detect(image);

        let gt = gradient_tensors(image);
        for kp in &mut keypoints {
            let x = kp.position.x as usize;
            let y = kp.position.y as usize;
            kp.response = harris_response(
                gt.sxx.get(x, y).value,
                gt.syy.get(x, y).value,
                gt.sxy.get(x, y).value,
                self.harris_k,
            );
        }

        keypoints.sort_unstable_by(|a, b| b.response.total_cmp(&a.response));
        keypoints.truncate(self.max_keypoints);

        intensity_centroid(image, &mut keypoints, self.orientation_radius);
        keypoints.retain(|kp| kp.orientation.is_some());
        keypoints
    }
}

#[cfg(test)]
mod tests {
    use oxislam_image::image::Image;

    use super::*;

    /// 20x20 dark image with a bright pixel — ORB should re-score with Harris and assign orientation.
    #[test]
    fn orb_rescores_and_orients() {
        let mut data = vec![Gray::new(0.0f32); 20 * 20];
        data[10 * 20 + 10] = Gray::new(1.0);
        let img = Image::new(20, 20, 20, data);

        let detector = OrbDetector::default();
        let keypoints = detector.detect(&img.view());

        for kp in &keypoints {
            assert!(kp.orientation.is_some(), "all ORB keypoints must have orientation");
        }
    }

    #[test]
    fn orb_truncates_to_max_keypoints() {
        // Dense texture: checkerboard produces many FAST corners.
        let size = 64;
        let data: Vec<_> = (0..size * size)
            .map(|i| {
                let x = i % size;
                let y = i / size;
                Gray::new(if (x + y) % 2 == 0 { 1.0f32 } else { 0.0 })
            })
            .collect();
        let img = Image::new(size, size, size, data);

        let mut detector = OrbDetector::default();
        detector.max_keypoints = 5;
        let keypoints = detector.detect(&img.view());

        assert!(keypoints.len() <= 5, "should truncate to max_keypoints");
    }
}
