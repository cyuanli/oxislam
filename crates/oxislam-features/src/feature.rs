use crate::keypoint::Keypoint;

pub struct Feature<D> {
    pub keypoint: Keypoint,
    pub descriptor: D,
}
