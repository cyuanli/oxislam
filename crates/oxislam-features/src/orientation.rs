use oxislam_image::image::ImageView;
use oxislam_image::pixel::Gray;

use crate::keypoint::Keypoint;

/// Rotate an integer offset by the given sin/cos angle.
#[inline]
pub fn rotate((dx, dy): (isize, isize), sin: f32, cos: f32) -> (isize, isize) {
    let rx = (dx as f32 * cos - dy as f32 * sin).round() as isize;
    let ry = (dx as f32 * sin + dy as f32 * cos).round() as isize;
    (rx, ry)
}

/// Compute keypoint orientations using the intensity centroid method within a circular `radius`.
pub fn intensity_centroid(image: &ImageView<Gray<f32>>, keypoints: &mut [Keypoint], radius: usize) {
    let r = radius as isize;
    let r_sq = (radius * radius) as isize;

    for kp in keypoints.iter_mut() {
        let cx = kp.position.x.round() as isize;
        let cy = kp.position.y.round() as isize;

        if cx - r < 0
            || cy - r < 0
            || cx + r >= image.width() as isize
            || cy + r >= image.height() as isize
        {
            continue;
        }

        let mut m10: f32 = 0.0;
        let mut m01: f32 = 0.0;

        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy > r_sq {
                    continue;
                }
                let val = image.get((cx + dx) as usize, (cy + dy) as usize).value;
                m10 += dx as f32 * val;
                m01 += dy as f32 * val;
            }
        }

        kp.orientation = Some(m01.atan2(m10));
    }
}

#[cfg(test)]
mod tests {
    use oxislam_geometry::Point2;
    use oxislam_image::image::Image;

    use super::*;

    #[test]
    fn bright_pixel_right_of_center_gives_zero_angle() {
        // 11x11 image, keypoint at center (5,5), bright pixel at (7,5) → m10 > 0, m01 = 0 → angle ≈ 0
        let mut data = vec![Gray::new(0.0f32); 11 * 11];
        data[5 * 11 + 7] = Gray::new(1.0);
        let img = Image::new(11, 11, 11, data);

        let mut kps = vec![Keypoint {
            position: Point2::new(5.0, 5.0),
            scale: 1.0,
            orientation: None,
            response: 1.0,
        }];
        intensity_centroid(&img.view(), &mut kps, 3);

        let angle = kps[0].orientation.expect("should have orientation");
        assert!(angle.abs() < 1e-6, "expected angle ≈ 0, got {angle}");
    }

    #[test]
    fn bright_pixel_below_center_gives_half_pi() {
        // Bright pixel at (5,7) → m10 = 0, m01 > 0 → angle ≈ π/2
        let mut data = vec![Gray::new(0.0f32); 11 * 11];
        data[7 * 11 + 5] = Gray::new(1.0);
        let img = Image::new(11, 11, 11, data);

        let mut kps = vec![Keypoint {
            position: Point2::new(5.0, 5.0),
            scale: 1.0,
            orientation: None,
            response: 1.0,
        }];
        intensity_centroid(&img.view(), &mut kps, 3);

        let angle = kps[0].orientation.expect("should have orientation");
        assert!(
            (angle - std::f32::consts::FRAC_PI_2).abs() < 1e-6,
            "expected angle ≈ π/2, got {angle}"
        );
    }

    #[test]
    fn border_keypoint_skipped() {
        let data = vec![Gray::new(1.0f32); 11 * 11];
        let img = Image::new(11, 11, 11, data);

        let mut kps = vec![Keypoint {
            position: Point2::new(1.0, 1.0),
            scale: 1.0,
            orientation: None,
            response: 1.0,
        }];
        intensity_centroid(&img.view(), &mut kps, 3);

        assert!(kps[0].orientation.is_none(), "keypoint too close to border should be skipped");
    }
}
