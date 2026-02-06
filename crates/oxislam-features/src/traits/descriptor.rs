use oxislam_image::image::ImageView;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::feature::Feature;
use crate::keypoint::Keypoint;

#[cfg(not(feature = "rayon"))]
pub trait DescriptorExtractor<P, D> {
    fn describe_one(&self, image: &ImageView<P>, keypoint: &Keypoint) -> Option<D>;

    fn describe(&self, image: &ImageView<P>, keypoints: Vec<Keypoint>) -> Vec<Feature<D>> {
        keypoints
            .into_iter()
            .filter_map(|kp| self.describe_one(image, &kp).map(|d| Feature::new(kp, d)))
            .collect()
    }
}

#[cfg(feature = "rayon")]
pub trait DescriptorExtractor<P, D>: Sync
where
    P: Sync,
    D: Send,
{
    fn describe_one(&self, image: &ImageView<P>, keypoint: &Keypoint) -> Option<D>;

    fn describe(&self, image: &ImageView<P>, keypoints: Vec<Keypoint>) -> Vec<Feature<D>> {
        keypoints
            .into_par_iter()
            .filter_map(|kp| self.describe_one(image, &kp).map(|d| Feature::new(kp, d)))
            .collect()
    }
}
