use oxislam_geometry::Point2;
use oxislam_image::image::{Image, ImageView};
use oxislam_image::parallel::{MaybeSend, MaybeSync, par_filter_map};

use crate::feature::Feature;
use crate::keypoint::Keypoint;
use crate::pyramid::Pyramid;

/// A descriptor backed by packed binary data (`[u64]`).
pub trait BinaryDescriptor: MaybeSend {
    fn bits(&self) -> &[u64];
}

/// A descriptor backed by floating-point data (`[f32]`).
pub trait FloatDescriptor: MaybeSend {
    fn values(&self) -> &[f32];
}

/// Extracts descriptors for keypoints.
pub trait DescriptorExtractor<P: MaybeSync + Copy, D: MaybeSend>: MaybeSync {
    /// Extract a descriptor for a single keypoint. Returns `None` if the keypoint is near image boundaries.
    fn describe_one(&self, image: &ImageView<P>, keypoint: &Keypoint) -> Option<D>;

    /// Optional preprocessing applied once per image before descriptor extraction (e.g. Gaussian blur).
    fn preprocess(&self, _image: &ImageView<P>) -> Option<Image<P>> { None }

    /// Extract descriptors for multiple keypoints in parallel.
    fn describe(&self, image: &ImageView<P>, keypoints: Vec<Keypoint>) -> Vec<Feature<D>> {
        let _span = crate::trace::span!("describe", keypoints = keypoints.len());
        let preprocessed = self.preprocess(image);
        let view;
        let image = match &preprocessed {
            Some(img) => {
                view = img.view();
                &view
            }
            None => image,
        };
        par_filter_map(keypoints, |kp| self.describe_one(image, &kp).map(|d| Feature::new(kp, d)))
    }

    /// Extract descriptors for multi-scale keypoints, applying preprocessing once per pyramid level.
    fn describe_at_scale(&self, pyramid: &Pyramid<P>, keypoints: Vec<Keypoint>) -> Vec<Feature<D>> {
        let _span = crate::trace::span!(
            "describe_at_scale",
            keypoints = keypoints.len(),
            levels = pyramid.num_levels()
        );
        let preprocessed: Vec<Option<Image<P>>> =
            (0..pyramid.num_levels()).map(|l| self.preprocess(&pyramid.level(l).view())).collect();

        par_filter_map(keypoints, |kp| {
            let level = pyramid.level_for_scale(kp.scale);
            let scale = pyramid.scale_at_level(level);

            let image = match &preprocessed[level] {
                Some(img) => img.view(),
                None => pyramid.level(level).view(),
            };

            let local_kp = Keypoint {
                position: Point2::new(kp.position.x / scale, kp.position.y / scale),
                ..kp
            };

            self.describe_one(&image, &local_kp).map(|d| Feature::new(kp, d))
        })
    }
}
