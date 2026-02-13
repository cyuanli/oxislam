use oxislam_image::image::ImageView;

use crate::feature::Feature;

/// An opinionated detect-and-describe pipeline for a specific algorithm.
///
/// Unlike using [`KeypointDetector`](super::detector::KeypointDetector) and
/// [`DescriptorExtractor`](super::descriptor::DescriptorExtractor) separately,
/// a `FeaturePipeline` encapsulates the full recipe (e.g. multi-scale detection,
/// response sorting, truncation, orientation, and descriptor extraction) behind
/// a single `extract()` call.
pub trait FeaturePipeline<P, D> {
    /// Run the full pipeline: detect keypoints, compute orientations, and
    /// extract descriptors from the given image.
    fn extract(&self, image: &ImageView<P>) -> Vec<Feature<D>>;
}
