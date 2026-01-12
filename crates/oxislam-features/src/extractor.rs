use crate::feature::Feature;
use oxislam_image::image::ImageView;

pub trait FeatureExtractor<P, D> {
    fn extract(&self, image: &ImageView<P>) -> Vec<Feature<D>>;
}
