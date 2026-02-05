use super::kernel::{Kernel, apply_kernel};
use crate::image::{Image, ImageView};
use crate::pixel::Gray;

#[rustfmt::skip]
const GAUSSIAN_3X3: Kernel<3> = [
    [1.0/16.0, 2.0/16.0, 1.0/16.0],
    [2.0/16.0, 4.0/16.0, 2.0/16.0],
    [1.0/16.0, 2.0/16.0, 1.0/16.0],
];

#[rustfmt::skip]
const GAUSSIAN_5X5: Kernel<5> = [
    [1.0/256.0,  4.0/256.0,  6.0/256.0,  4.0/256.0, 1.0/256.0],
    [4.0/256.0, 16.0/256.0, 24.0/256.0, 16.0/256.0, 4.0/256.0],
    [6.0/256.0, 24.0/256.0, 36.0/256.0, 24.0/256.0, 6.0/256.0],
    [4.0/256.0, 16.0/256.0, 24.0/256.0, 16.0/256.0, 4.0/256.0],
    [1.0/256.0,  4.0/256.0,  6.0/256.0,  4.0/256.0, 1.0/256.0],
];

pub fn gaussian_3x3(image: &ImageView<Gray<f32>>) -> Image<Gray<f32>> {
    apply_kernel(image, &GAUSSIAN_3X3)
}

pub fn gaussian_5x5(image: &ImageView<Gray<f32>>) -> Image<Gray<f32>> {
    apply_kernel(image, &GAUSSIAN_5X5)
}
