use crate::traits::descriptor::{BinaryDescriptor, FloatDescriptor};
use crate::traits::matcher::Distance;

/// Hamming distance: number of differing bits.
pub struct Hamming;

impl<D: BinaryDescriptor> Distance<D> for Hamming {
    #[inline]
    fn distance(&self, a: &D, b: &D) -> f32 {
        let mut dist = 0u32;
        for (wa, wb) in a.bits().iter().zip(b.bits()) {
            dist += (wa ^ wb).count_ones();
        }
        dist as f32
    }
}

/// Squared Euclidean (L2²) distance.
pub struct L2Squared;

impl<D: FloatDescriptor> Distance<D> for L2Squared {
    #[inline]
    fn distance(&self, a: &D, b: &D) -> f32 {
        let mut dist = 0.0f32;
        for (va, vb) in a.values().iter().zip(b.values()) {
            let d = va - vb;
            dist += d * d;
        }
        dist
    }
}
