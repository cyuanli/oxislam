use oxislam_image::parallel::MaybeSend;

/// Distance metric between two descriptors.
pub trait Distance<D>: MaybeSend {
    fn distance(&self, a: &D, b: &D) -> f32;
}

/// A match between two descriptors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DescriptorMatch {
    /// Index of the descriptor in the source set.
    pub source_idx: usize,
    /// Index of the descriptor in the reference set.
    pub reference_idx: usize,
    /// Distance between the two descriptors (lower is better).
    pub distance: f32,
}

/// Matches descriptors from a source set against a reference set.
pub trait DescriptorMatcher<D> {
    /// Find the best match in `reference` for each descriptor in `source`.
    fn match_descriptors(&self, source: &[D], reference: &[D]) -> Vec<DescriptorMatch>;
}
