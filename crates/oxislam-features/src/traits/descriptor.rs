use oxislam_image::image::ImageView;
use oxislam_image::parallel::{MaybeSend, MaybeSync, par_filter_map};

use crate::feature::Feature;
use crate::keypoint::Keypoint;

/// A descriptor backed by packed binary data (`[u64]`).
pub trait BinaryDescriptor: MaybeSend {
    fn bits(&self) -> &[u64];
}

/// A descriptor backed by floating-point data (`[f32]`).
pub trait FloatDescriptor: MaybeSend {
    fn values(&self) -> &[f32];
}

/// Extracts descriptors for keypoints.
pub trait DescriptorExtractor<P: MaybeSync, D: MaybeSend>: MaybeSync {
    /// Extract a descriptor for a single keypoint. Returns `None` if the keypoint is near image boundaries.
    fn describe_one(&self, image: &ImageView<P>, keypoint: &Keypoint) -> Option<D>;

    /// Extract descriptors for multiple keypoints in parallel.
    fn describe(&self, image: &ImageView<P>, keypoints: Vec<Keypoint>) -> Vec<Feature<D>> {
        par_filter_map(keypoints, |kp| self.describe_one(image, &kp).map(|d| Feature::new(kp, d)))
    }
}
