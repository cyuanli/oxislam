use crate::keypoint::Keypoint;

#[derive(Debug, Clone)]
pub struct Feature<D> {
    pub keypoint: Keypoint,
    pub descriptor: D,
}

impl<D> Feature<D> {
    #[inline]
    pub fn new(keypoint: Keypoint, descriptor: D) -> Self {
        Self { keypoint, descriptor }
    }
}
