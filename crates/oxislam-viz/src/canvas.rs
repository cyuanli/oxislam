use oxislam_image::Rgb;
use oxislam_image::image::{Image, ImageView};

/// Create a new image by placing `img1` and `img2` side by side horizontally.
///
/// The output height is the maximum of the two input heights.
/// Pixels beyond a shorter image's height are left black.
pub fn side_by_side(img1: &ImageView<Rgb<u8>>, img2: &ImageView<Rgb<u8>>) -> Image<Rgb<u8>> {
    let (w1, h1) = (img1.width(), img1.height());
    let (w2, h2) = (img2.width(), img2.height());
    let out_w = w1 + w2;
    let out_h = h1.max(h2);

    let mut canvas = Image::filled(out_w, out_h, Rgb::new(0, 0, 0));

    for y in 0..h1 {
        for x in 0..w1 {
            *canvas.get_mut(x, y) = *img1.get(x, y);
        }
    }
    for y in 0..h2 {
        for x in 0..w2 {
            *canvas.get_mut(w1 + x, y) = *img2.get(x, y);
        }
    }

    canvas
}
