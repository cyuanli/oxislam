use crate::keypoint::Keypoint;

/// A detected feature (keypoint + descriptor).
#[derive(Debug, Clone)]
pub struct Feature<D> {
    /// The detected keypoint.
    pub keypoint: Keypoint,
    /// The descriptor for the keypoint.
    pub descriptor: D,
}

impl<D> Feature<D> {
    /// Create a new feature.
    #[inline]
    pub fn new(keypoint: Keypoint, descriptor: D) -> Self { Self { keypoint, descriptor } }
}
