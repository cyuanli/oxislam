use super::kernel::{Kernel, apply_kernel};
use crate::image::{Image, ImageView};
use crate::pixel::Gray;

#[rustfmt::skip]
pub const SOBEL_X: Kernel<3> = [
    [-1.0, 0.0, 1.0],
    [-2.0, 0.0, 2.0],
    [-1.0, 0.0, 1.0],
];

#[rustfmt::skip]
pub const SOBEL_Y: Kernel<3> = [
    [-1.0, -2.0, -1.0],
    [ 0.0,  0.0,  0.0],
    [ 1.0,  2.0,  1.0],
];

pub fn sobel(image: &ImageView<Gray<f32>>) -> (Image<Gray<f32>>, Image<Gray<f32>>) {
    let ix = apply_kernel(image, &SOBEL_X);
    let iy = apply_kernel(image, &SOBEL_Y);
    (ix, iy)
}
