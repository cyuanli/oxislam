use oxislam_image::Rgb;

/// Return a visually distinct color for the given index using golden-angle hue rotation.
pub fn distinct_color(index: usize) -> Rgb<u8> {
    let hue = ((index as f32) * 137.508) % 360.0;
    hsv_to_rgb(hue, 1.0, 1.0)
}

/// Convert HSV to an RGB color.
///
/// * `h` — hue in degrees (0..360)
/// * `s` — saturation (0..1)
/// * `v` — value (0..1)
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Rgb<u8> {
    let c = v * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());
    let (r1, g1, b1) = match h_prime as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    Rgb::new(((r1 + m) * 255.0) as u8, ((g1 + m) * 255.0) as u8, ((b1 + m) * 255.0) as u8)
}
