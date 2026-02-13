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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sobel_gradient_direction() {
        // [0, 0, 1, 1]
        // [0, 0, 1, 1]
        // [0, 0, 1, 1]
        let data: Vec<Gray<f32>> =
            (0..3).flat_map(|_| [0.0, 0.0, 1.0, 1.0]).map(Gray::new).collect();
        let img = Image::new(4, 3, 4, data);

        let (ix, iy) = sobel(&img.view());
        // Same dimensions as input.
        assert_eq!(ix.width(), 4);
        assert_eq!(ix.height(), 3);
        // Center pixel (1,1) has strong horizontal gradient, no vertical.
        assert_eq!(ix.get(1, 1).value, 4.0);
        assert_eq!(iy.get(1, 1).value, 0.0);
    }
}
