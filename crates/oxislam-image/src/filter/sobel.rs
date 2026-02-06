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
        // Output is 2x1
        assert_eq!(ix.width(), 2);
        assert_eq!(ix.height(), 1);
        assert_eq!(ix.get(0, 0).value, 4.0); // strong horizontal gradient
        assert_eq!(iy.get(0, 0).value, 0.0); // no vertical gradient
    }
}
