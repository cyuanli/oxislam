use oxislam_image::Gray;
use oxislam_image::image::ImageView;

use crate::keypoint::Keypoint;
use crate::traits::descriptor::DescriptorExtractor;
use crate::traits::metric::FloatDescriptor;

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
    fn build<P, F>(&self, view: ImageView<P>, to_f32: F) -> Patch<L>
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

        Patch::new(data)
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
pub struct Patch<const L: usize> {
    data: [f32; L],
}

impl<const L: usize> Patch<L> {
    #[inline]
    pub fn new(data: [f32; L]) -> Self { Self { data } }
}

impl<const L: usize> FloatDescriptor for Patch<L> {
    #[inline]
    fn data(&self) -> &[f32] { &self.data }
}

impl<const N: usize, const L: usize> DescriptorExtractor<Gray<f32>, Patch<L>>
    for PatchExtractor<N, L>
{
    fn describe_one(&self, image: &ImageView<Gray<f32>>, kp: &Keypoint) -> Option<Patch<L>> {
        let patch = image.patch(kp.position.x, kp.position.y, N)?;
        Some(self.build(patch, |p| p.value))
    }
}
