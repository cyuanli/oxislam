use oxislam_image::Gray;
use oxislam_image::image::ImageView;

use crate::keypoint::Keypoint;
use crate::traits::descriptor::DescriptorExtractor;

#[derive(Debug, Clone)]
pub struct PatchExtractor<const N: usize, const L: usize> {
    normalize: bool,
}

impl<const N: usize, const L: usize> PatchExtractor<N, L> {
    #[inline]
    pub fn new(normalize: bool) -> Self {
        assert!(N % 2 == 1, "Patch size must be odd");
        assert!(N >= 3, "Patch size must be at least 3");
        assert!(L == N * N, "L must equal N * N");
        Self { normalize }
    }

    #[inline]
    pub const fn patch_size(&self) -> usize { N }

    #[inline]
    pub const fn descriptor_length(&self) -> usize { L }

    #[inline]
    fn build<P, F>(&self, view: ImageView<P>, to_f32: F) -> PatchDescriptor<L>
    where
        F: Fn(&P) -> f32,
    {
        let mut data = [0.0f32; L];
        for (i, pixel) in view.pixels().enumerate() {
            data[i] = to_f32(pixel);
        }

        if self.normalize {
            Self::normalize(&mut data);
        }

        PatchDescriptor::new(data)
    }

    #[inline]
    fn normalize(data: &mut [f32; L]) {
        let n = L as f32;
        let mean: f32 = data.iter().sum::<f32>() / n;
        let variance: f32 = data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let std_dev = variance.sqrt();

        if std_dev > 1e-10 {
            for v in data {
                *v = (*v - mean) / std_dev;
            }
        } else {
            data.fill(0.0);
        }
    }
}

impl<const N: usize, const L: usize> Default for PatchExtractor<N, L> {
    #[inline]
    fn default() -> Self { Self::new(true) }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PatchDescriptor<const L: usize> {
    data: [f32; L],
}

impl<const L: usize> PatchDescriptor<L> {
    #[inline]
    pub fn new(data: [f32; L]) -> Self { Self { data } }
}

impl<const N: usize, const L: usize> DescriptorExtractor<Gray<f32>, PatchDescriptor<L>>
    for PatchExtractor<N, L>
{
    fn describe_one(
        &self,
        image: &ImageView<Gray<f32>>,
        kp: &Keypoint,
    ) -> Option<PatchDescriptor<L>> {
        let patch = image.patch(kp.position.x, kp.position.y, N)?;
        Some(self.build(patch, |p| p.value))
    }
}

#[cfg(test)]
mod tests {
    use oxislam_geometry::Point2;
    use oxislam_image::image::Image;

    use super::*;

    fn kp_at(x: f32, y: f32) -> Keypoint {
        Keypoint { position: Point2::new(x, y), scale: 1.0, orientation: None, response: 1.0 }
    }

    #[test]
    fn normalize_uniform_patch_produces_zeros() {
        let data: Vec<Gray<f32>> = vec![Gray::new(0.5); 9];
        let img = Image::new(3, 3, 3, data);
        let extractor = PatchExtractor::<3, 9>::new(true);
        let desc = extractor.describe_one(&img.view(), &kp_at(1.0, 1.0)).unwrap();

        for &v in desc.data.iter() {
            assert!(v.is_finite(), "descriptor element should be finite, got {v}");
            assert_eq!(v, 0.0, "uniform patch should normalize to all zeros");
        }
    }

    #[test]
    fn normalize_known_patch() {
        let data: Vec<Gray<f32>> = (1..=9).map(|v| Gray::new(v as f32)).collect();
        let img = Image::new(3, 3, 3, data);
        let extractor = PatchExtractor::<3, 9>::new(true);
        let desc = extractor.describe_one(&img.view(), &kp_at(1.0, 1.0)).unwrap();
        let n = desc.data.len() as f32;
        let mean: f32 = desc.data.iter().sum::<f32>() / n;
        let variance: f32 = desc.data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let std_dev = variance.sqrt();

        assert!(mean.abs() < 1e-5, "mean should be ~0, got {mean}");
        assert!((std_dev - 1.0).abs() < 1e-5, "std_dev should be ~1, got {std_dev}");
    }
}
