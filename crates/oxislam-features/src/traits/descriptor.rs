use oxislam_image::image::ImageView;

use crate::keypoint::Keypoint;

pub trait DescriptorExtractor<P, D> {
    fn describe(&self, image: &ImageView<P>, keypoints: &[Keypoint]) -> Vec<D>;
}
