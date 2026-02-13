use oxislam_geometry::Point2;
use oxislam_image::image::ImageView;
use oxislam_image::pixel::Gray;

use crate::descriptor::brief::{BriefDescriptor256, BriefExtractor256};
use crate::detector::orb::OrbDetector;
use crate::feature::Feature;
use crate::orientation::intensity_centroid;
use crate::pyramid::Pyramid;
use crate::trace::{event, span};
use crate::traits::descriptor::DescriptorExtractor;
use crate::traits::detector::KeypointDetector;
use crate::traits::pipeline::FeaturePipeline;

const DEFAULT_MAX_KEYPOINTS: usize = 500;
const DEFAULT_NUM_LEVELS: usize = 8;
const DEFAULT_SCALE_FACTOR: f32 = 1.2;
const DEFAULT_ORIENTATION_RADIUS: usize = 15;

/// Full ORB feature pipeline: multi-scale FAST + Harris → sort → truncate →
/// intensity centroid orientation → rBRIEF descriptors.
pub struct OrbPipeline {
    pub detector: OrbDetector,
    pub extractor: BriefExtractor256,
    pub max_keypoints: usize,
    pub num_levels: usize,
    pub scale_factor: f32,
    pub orientation_radius: usize,
}

impl Default for OrbPipeline {
    fn default() -> Self {
        Self {
            detector: OrbDetector::default(),
            extractor: BriefExtractor256::default(),
            max_keypoints: DEFAULT_MAX_KEYPOINTS,
            num_levels: DEFAULT_NUM_LEVELS,
            scale_factor: DEFAULT_SCALE_FACTOR,
            orientation_radius: DEFAULT_ORIENTATION_RADIUS,
        }
    }
}

impl FeaturePipeline<Gray<f32>, BriefDescriptor256> for OrbPipeline {
    fn extract(&self, image: &ImageView<Gray<f32>>) -> Vec<Feature<BriefDescriptor256>> {
        let _span = span!("orb_extract");

        let pyramid = Pyramid::build(image, self.num_levels, self.scale_factor);

        // Detect and orient at each pyramid level, then rescale to base coordinates
        let mut keypoints = Vec::new();
        for level in 0..pyramid.num_levels() {
            let level_view = pyramid.level(level).view();

            let mut kps = {
                let _s = span!("detect", level = level);
                self.detector.detect(&level_view)
            };

            {
                let _s = span!("orient", level = level);
                intensity_centroid(&level_view, &mut kps, self.orientation_radius);
                kps.retain(|kp| kp.orientation.is_some());
            }
            event!(tracing::Level::DEBUG, level_keypoints = kps.len(), level = level);

            let scale = pyramid.scale_at_level(level);
            for kp in &mut kps {
                kp.position = Point2::new(kp.position.x * scale, kp.position.y * scale);
                kp.scale = scale;
            }
            keypoints.extend(kps);
        }
        if keypoints.is_empty() {
            event!(tracing::Level::WARN, "no keypoints detected across any pyramid level");
        }
        event!(tracing::Level::INFO, total_keypoints = keypoints.len());

        // Sort by response (descending) and truncate
        {
            let _s = span!("sort_truncate");
            keypoints.sort_unstable_by(|a, b| b.response.total_cmp(&a.response));
            keypoints.truncate(self.max_keypoints);
        }
        event!(tracing::Level::INFO, after_truncate = keypoints.len(), max = self.max_keypoints);

        // Extract BRIEF descriptors at the appropriate pyramid scale
        self.extractor.describe_at_scale(&pyramid, keypoints)
    }
}

#[cfg(test)]
mod tests {
    use oxislam_image::image::Image;

    use super::*;

    fn checkerboard(size: usize) -> Image<Gray<f32>> {
        let data: Vec<_> = (0..size * size)
            .map(|i| {
                let x = i % size;
                let y = i / size;
                Gray::new(if (x + y) % 2 == 0 { 1.0f32 } else { 0.0 })
            })
            .collect();
        Image::new(size, size, size, data)
    }

    #[test]
    fn truncates_to_max_keypoints() {
        let img = checkerboard(64);
        let pipeline = OrbPipeline { max_keypoints: 5, ..OrbPipeline::default() };
        let features = pipeline.extract(&img.view());

        assert!(features.len() <= 5, "should truncate to max_keypoints");
    }

    #[test]
    fn all_features_have_orientation() {
        let img = checkerboard(128);
        let pipeline = OrbPipeline::default();
        let features = pipeline.extract(&img.view());

        assert!(!features.is_empty());
        for (i, f) in features.iter().enumerate() {
            assert!(f.keypoint.orientation.is_some(), "feature {i} missing orientation");
        }
    }

    #[test]
    fn multiscale_produces_multiple_scales() {
        let img = checkerboard(128);
        let pipeline = OrbPipeline { num_levels: 4, ..OrbPipeline::default() };
        let features = pipeline.extract(&img.view());

        let distinct_scales: std::collections::HashSet<u32> =
            features.iter().map(|f| f.keypoint.scale.to_bits()).collect();
        assert!(
            distinct_scales.len() > 1,
            "expected features at multiple scales, got {}",
            distinct_scales.len()
        );
    }

    #[test]
    fn positions_are_in_base_image_coordinates() {
        let size = 128;
        let img = checkerboard(size);
        let pipeline = OrbPipeline { num_levels: 4, ..OrbPipeline::default() };
        let features = pipeline.extract(&img.view());

        for f in &features {
            let pos = &f.keypoint.position;
            assert!(
                pos.x >= 0.0 && pos.x < size as f32 && pos.y >= 0.0 && pos.y < size as f32,
                "position ({}, {}) outside base image bounds",
                pos.x,
                pos.y
            );
        }
    }
}
