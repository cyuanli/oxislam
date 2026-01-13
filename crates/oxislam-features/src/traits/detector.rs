use oxislam_image::image::ImageView;

use crate::keypoint::Keypoint;

pub trait KeypointDetector<P> {
    fn detect(&self, image: &ImageView<P>) -> Vec<Keypoint>;
}
