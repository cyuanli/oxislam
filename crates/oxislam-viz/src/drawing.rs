use oxislam_geometry::Point2;
use oxislam_image::Rgb;
use oxislam_image::image::Image;

/// Draw a cross marker centered at `point`. `arm` is the half-length in pixels.
pub fn draw_cross(image: &mut Image<Rgb<u8>>, point: Point2<f32>, color: Rgb<u8>, arm: i32) {
    let (w, h) = (image.width() as i32, image.height() as i32);
    let cx = point.x.round() as i32;
    let cy = point.y.round() as i32;
    for d in -arm..=arm {
        let px = cx + d;
        let py = cy + d;
        if px >= 0 && px < w && cy >= 0 && cy < h {
            *image.get_mut(px as usize, cy as usize) = color;
        }
        if cx >= 0 && cx < w && py >= 0 && py < h {
            *image.get_mut(cx as usize, py as usize) = color;
        }
    }
}

/// Draw a line from `p0` to `p1` using Bresenham's algorithm.
pub fn draw_line(image: &mut Image<Rgb<u8>>, p0: Point2<f32>, p1: Point2<f32>, color: Rgb<u8>) {
    let (w, h) = (image.width() as i32, image.height() as i32);
    let x0 = p0.x.round() as i32;
    let y0 = p0.y.round() as i32;
    let x1 = p1.x.round() as i32;
    let y1 = p1.y.round() as i32;

    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;
    loop {
        if x >= 0 && x < w && y >= 0 && y < h {
            *image.get_mut(x as usize, y as usize) = color;
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}
