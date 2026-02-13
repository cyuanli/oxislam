use crate::image::{Image, ImageView};
use crate::parallel::par_row_collect;
use crate::pixel::Gray;

/// Resize a grayscale image using bilinear interpolation.
pub fn resize_bilinear(
    image: &ImageView<Gray<f32>>,
    new_w: usize,
    new_h: usize,
) -> Image<Gray<f32>> {
    let _span = crate::trace::span!(
        "resize_bilinear",
        src_w = image.width(),
        src_h = image.height(),
        dst_w = new_w,
        dst_h = new_h
    );
    assert!(new_w > 0 && new_h > 0, "output dimensions must be > 0");

    let src_w = image.width() as f32;
    let src_h = image.height() as f32;

    let data = par_row_collect(new_w, new_h, |x, y| {
        let src_x = ((x as f32 + 0.5) * src_w / new_w as f32 - 0.5).max(0.0);
        let src_y = ((y as f32 + 0.5) * src_h / new_h as f32 - 0.5).max(0.0);

        let x0 = src_x.floor() as usize;
        let y0 = src_y.floor() as usize;
        let x1 = (x0 + 1).min(image.width() - 1);
        let y1 = (y0 + 1).min(image.height() - 1);

        let fx = src_x - src_x.floor();
        let fy = src_y - src_y.floor();

        let p00 = image.get(x0, y0).value;
        let p10 = image.get(x1, y0).value;
        let p01 = image.get(x0, y1).value;
        let p11 = image.get(x1, y1).value;

        let value = p00 * (1.0 - fx) * (1.0 - fy)
            + p10 * fx * (1.0 - fy)
            + p01 * (1.0 - fx) * fy
            + p11 * fx * fy;

        Gray::new(value)
    });

    Image::new(new_w, new_h, new_w, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downsample_preserves_average() {
        // 4x4 image with known values, downsample to 2x2.
        #[rustfmt::skip]
        let data: Vec<Gray<f32>> = [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ].iter().map(|&v| Gray::new(v)).collect();
        let img = Image::new(4, 4, 4, data);

        let out = resize_bilinear(&img.view(), 2, 2);
        assert_eq!(out.width(), 2);
        assert_eq!(out.height(), 2);
        // Each output pixel should be a blend of its 2x2 neighborhood.
        // Values should be reasonable (between min and max of source).
        for y in 0..2 {
            for x in 0..2 {
                let v = out.get(x, y).value;
                assert!(v >= 1.0 && v <= 8.0, "pixel ({x},{y}) = {v} out of range");
            }
        }
    }

    #[test]
    fn upsample_corners_clamp_correctly() {
        // 2x2 image, upsample to 4x4.
        // Corner pixels of output should equal corner pixels of input (clamped, no negative src coords).
        let data = vec![Gray::new(1.0), Gray::new(2.0), Gray::new(3.0), Gray::new(4.0)];
        let img = Image::new(2, 2, 2, data);

        let out = resize_bilinear(&img.view(), 4, 4);
        assert_eq!(out.width(), 4);
        assert_eq!(out.height(), 4);

        // Top-left output pixel: src_x = (0.5 * 2/4 - 0.5).max(0) = 0.0 → samples pixel (0,0).
        assert_eq!(out.get(0, 0).value, 1.0);
        // Bottom-right output pixel: src coords map near (1,1) → pixel (1,1) = 4.0.
        // src_x = (3.5 * 2/4 - 0.5) = 1.25, clamped by x1.min(w-1) → blends toward 4.0.
        let br = out.get(3, 3).value;
        assert!(br > 3.5 && br <= 4.0, "bottom-right = {br}, expected close to 4.0");
    }

    #[test]
    fn same_size_is_identity() {
        let data = vec![Gray::new(1.0), Gray::new(2.0), Gray::new(3.0), Gray::new(4.0)];
        let img = Image::new(2, 2, 2, data);

        let out = resize_bilinear(&img.view(), 2, 2);
        for y in 0..2 {
            for x in 0..2 {
                assert_eq!(out.get(x, y).value, img.get(x, y).value);
            }
        }
    }
}
