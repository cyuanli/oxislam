use oxislam_image::image::ImageView;
use oxislam_image::parallel::{MaybeSend, MaybeSync, par_filter_map};

use crate::feature::Feature;
use crate::keypoint::Keypoint;

pub trait DescriptorExtractor<P: MaybeSync, D: MaybeSend>: MaybeSync {
    fn describe_one(&self, image: &ImageView<P>, keypoint: &Keypoint) -> Option<D>;

    fn describe(&self, image: &ImageView<P>, keypoints: Vec<Keypoint>) -> Vec<Feature<D>> {
        par_filter_map(keypoints, |kp| self.describe_one(image, &kp).map(|d| Feature::new(kp, d)))
    }
}
