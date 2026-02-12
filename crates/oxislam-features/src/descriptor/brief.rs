use oxislam_image::Gray;
use oxislam_image::image::ImageView;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Normal};

use crate::keypoint::Keypoint;
use crate::traits::descriptor::{BinaryDescriptor, DescriptorExtractor};

const DEFAULT_PATCH_SIZE: usize = 49;
const DEFAULT_SEED: u64 = 0xB21EF;

/// A single binary intensity test: compares pixel at `t1` against pixel at `t2`.
#[derive(Debug, Clone, Copy)]
pub struct BinaryTest {
    pub t1: (isize, isize),
    pub t2: (isize, isize),
}

/// BRIEF (Binary Robust Independent Elementary Features) descriptor extractor.
///
/// Produces binary descriptors of `L * 64` bits by comparing pairs of pixel
/// intensities within a patch around each keypoint. Test locations are sampled
/// from an isotropic Gaussian distribution.
#[derive(Debug, Clone)]
pub struct BriefExtractor<const L: usize> {
    patch_size: usize,
    binary_tests: Vec<BinaryTest>,
}

impl<const L: usize> BriefExtractor<L> {
    #[inline]
    /// Creates a new extractor with Gaussian-sampled test locations.
    /// `patch_size` must be odd and >= 3.
    pub fn new(patch_size: usize, seed: u64) -> Self {
        assert!(patch_size >= 3, "Patch size must be at least 3");
        assert!(patch_size % 2 == 1, "Patch size must be odd");
        Self { patch_size, binary_tests: Self::sample_binary_tests(patch_size, seed) }
    }

    #[inline]
    /// Creates an extractor with pre-defined binary tests.
    pub fn with_tests(patch_size: usize, binary_tests: Vec<BinaryTest>) -> Self {
        assert!(patch_size >= 3, "Patch size must be at least 3");
        assert!(patch_size % 2 == 1, "Patch size must be odd");
        Self { patch_size, binary_tests }
    }

    fn sample_binary_tests(patch_size: usize, seed: u64) -> Vec<BinaryTest> {
        let num_bits: usize = L * 64;
        let mut tests = Vec::with_capacity(num_bits);

        let mut rng = SmallRng::seed_from_u64(seed);
        let gaussian = Normal::new(0.0, (patch_size - 1) as f32 / 6.0).unwrap();
        let mut sample = || {
            (gaussian.sample(&mut rng) as isize)
                .clamp(-(patch_size as isize) / 2, (patch_size as isize) / 2 - 1)
        };

        for _ in 0..num_bits {
            let test = BinaryTest { t1: (sample(), sample()), t2: (sample(), sample()) };
            tests.push(test);
        }

        tests
    }

    #[inline]
    fn build(&self, view: ImageView<Gray<f32>>) -> BriefDescriptor<L> {
        let mut data = [0u64; L];
        let center = (self.patch_size / 2) as isize;
        for (idx, &test) in self.binary_tests.iter().enumerate() {
            let p1 = view.get((center + test.t1.0) as usize, (center + test.t1.1) as usize);
            let p2 = view.get((center + test.t2.0) as usize, (center + test.t2.1) as usize);
            if p1.value > p2.value {
                data[idx / 64] |= 1u64 << (idx % 64);
            }
        }
        BriefDescriptor::new(data)
    }
}

impl<const L: usize> Default for BriefExtractor<L> {
    #[inline]
    fn default() -> Self { Self::new(DEFAULT_PATCH_SIZE, DEFAULT_SEED) }
}

/// A binary descriptor stored as `L` packed `u64` words (`L * 64` bits).
#[derive(Debug, Clone, PartialEq)]
pub struct BriefDescriptor<const L: usize> {
    data: [u64; L],
}

impl<const L: usize> BriefDescriptor<L> {
    #[inline]
    pub fn new(data: [u64; L]) -> Self { Self { data } }
}

impl<const L: usize> BinaryDescriptor for BriefDescriptor<L> {
    #[inline]
    fn bits(&self) -> &[u64] { &self.data }
}

impl<const L: usize> DescriptorExtractor<Gray<f32>, BriefDescriptor<L>> for BriefExtractor<L> {
    fn describe_one(
        &self,
        image: &ImageView<Gray<f32>>,
        kp: &Keypoint,
    ) -> Option<BriefDescriptor<L>> {
        let patch = image.patch(kp.position.x, kp.position.y, self.patch_size)?;
        Some(self.build(patch))
    }
}

pub type BriefExtractor128 = BriefExtractor<2>;
pub type BriefExtractor256 = BriefExtractor<4>;
pub type BriefExtractor512 = BriefExtractor<8>;

pub type BriefDescriptor128 = BriefDescriptor<2>;
pub type BriefDescriptor256 = BriefDescriptor<4>;
pub type BriefDescriptor512 = BriefDescriptor<8>;

#[cfg(test)]
mod tests {
    use oxislam_geometry::Point2;
    use oxislam_image::image::Image;

    use super::*;

    fn kp_at(x: f32, y: f32) -> Keypoint {
        Keypoint { position: Point2::new(x, y), scale: 1.0, orientation: None, response: 1.0 }
    }

    #[test]
    fn same_seed_produces_same_tests() {
        let a = BriefExtractor128::new(49, 42);
        let b = BriefExtractor128::new(49, 42);
        assert_eq!(a.binary_tests.len(), b.binary_tests.len());
        for (ta, tb) in a.binary_tests.iter().zip(&b.binary_tests) {
            assert_eq!(ta.t1, tb.t1);
            assert_eq!(ta.t2, tb.t2);
        }
    }

    #[test]
    fn boundary_keypoint_returns_none() {
        let data: Vec<Gray<f32>> = vec![Gray::new(0.5); 25];
        let img = Image::new(5, 5, 5, data);
        let extractor = BriefExtractor128::new(5, DEFAULT_SEED);
        // Keypoint at corner — not enough room for patch
        assert!(extractor.describe_one(&img.view(), &kp_at(0.0, 0.0)).is_none());
    }

    #[test]
    fn uniform_image_produces_zero_descriptor() {
        let size = 51;
        let data: Vec<Gray<f32>> = vec![Gray::new(0.5); size * size];
        let img = Image::new(size, size, size, data);
        let extractor = BriefExtractor128::new(5, DEFAULT_SEED);
        let center = (size / 2) as f32;
        let desc = extractor.describe_one(&img.view(), &kp_at(center, center)).unwrap();
        // All pixels equal → no comparison is ever "greater than" → all bits zero
        assert_eq!(desc.data, [0u64; 2]);
    }

    #[test]
    fn descriptor_changes_with_different_content() {
        let size = 51;
        let data_a: Vec<Gray<f32>> = (0..size * size).map(|i| Gray::new(i as f32)).collect();
        let data_b: Vec<Gray<f32>> =
            (0..size * size).map(|i| Gray::new((size * size - 1 - i) as f32)).collect();
        let img_a = Image::new(size, size, size, data_a);
        let img_b = Image::new(size, size, size, data_b);
        let extractor = BriefExtractor128::new(5, DEFAULT_SEED);
        let center = (size / 2) as f32;
        let desc_a = extractor.describe_one(&img_a.view(), &kp_at(center, center)).unwrap();
        let desc_b = extractor.describe_one(&img_b.view(), &kp_at(center, center)).unwrap();
        assert_ne!(desc_a, desc_b);
    }
}
