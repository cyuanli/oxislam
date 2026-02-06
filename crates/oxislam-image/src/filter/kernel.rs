use crate::image::{Image, ImageView};
use crate::parallel::par_row_collect;
use crate::pixel::Gray;

pub type Kernel<const N: usize> = [[f32; N]; N];

fn compute_pixel<const N: usize>(
    image: &ImageView<Gray<f32>>,
    kernel: &Kernel<N>,
    x: usize,
    y: usize,
) -> Gray<f32> {
    let mut sum = 0.0;
    for (ky, kernel_row) in kernel.iter().enumerate() {
        for (kx, kernel_val) in kernel_row.iter().enumerate() {
            sum += image.get(x + kx, y + ky).value * kernel_val;
        }
    }
    Gray::new(sum)
}

pub fn apply_kernel<const N: usize>(
    image: &ImageView<Gray<f32>>,
    kernel: &Kernel<N>,
) -> Image<Gray<f32>> {
    let w = image.width();
    let h = image.height();

    assert!(w >= N && h >= N, "Image must be at least {N}x{N}");

    let out_w = w - (N - 1);
    let out_h = h - (N - 1);

    let data = par_row_collect(out_w, out_h, |x, y| compute_pixel(image, kernel, x, y));

    Image::new(out_w, out_h, out_w, data)
}
