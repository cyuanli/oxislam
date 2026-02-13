use oxislam_image::filter::resize_bilinear;
use oxislam_image::image::{Image, ImageView};
use oxislam_image::pixel::Gray;

/// Gaussian image pyramid for multi-scale feature extraction.
pub struct Pyramid<P> {
    levels: Vec<Image<P>>,
    scale_factor: f32,
}

impl Pyramid<Gray<f32>> {
    /// Build a pyramid by repeatedly downscaling `image` by `scale_factor`.
    pub fn build(image: &ImageView<Gray<f32>>, num_levels: usize, scale_factor: f32) -> Self {
        let _span = crate::trace::span!("pyramid_build", num_levels = num_levels, scale_factor = %scale_factor);
        assert!(num_levels >= 1);
        assert!(scale_factor > 1.0);

        let mut levels = Vec::with_capacity(num_levels);

        levels.push(image.to_owned());

        let base_w = image.width() as f32;
        let base_h = image.height() as f32;

        for level in 1..num_levels {
            let scale = scale_factor.powi(level as i32);
            let new_w = (base_w / scale).round() as usize;
            let new_h = (base_h / scale).round() as usize;

            if new_w < 3 || new_h < 3 {
                crate::trace::event!(
                    tracing::Level::WARN,
                    requested = num_levels,
                    actual = levels.len(),
                    "pyramid truncated: level too small"
                );
                break;
            }

            levels.push(resize_bilinear(&levels.last().unwrap().view(), new_w, new_h));
        }

        Pyramid { levels, scale_factor }
    }
}

impl<P> Pyramid<P> {
    pub fn num_levels(&self) -> usize { self.levels.len() }

    pub fn scale_factor(&self) -> f32 { self.scale_factor }

    pub fn level(&self, index: usize) -> &Image<P> { &self.levels[index] }

    pub fn scale_at_level(&self, level: usize) -> f32 { self.scale_factor.powi(level as i32) }

    /// Return the pyramid level closest to the given scale.
    pub fn level_for_scale(&self, scale: f32) -> usize {
        let level = (scale.ln() / self.scale_factor.ln()).round() as usize;
        level.min(self.levels.len() - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_image(w: usize, h: usize) -> Image<Gray<f32>> {
        Image::new(w, h, w, vec![Gray::new(0.5); w * h])
    }

    #[test]
    fn pyramid_levels_and_dimensions() {
        let img = test_image(100, 100);
        let pyr = Pyramid::build(&img.view(), 4, 1.2);

        assert_eq!(pyr.num_levels(), 4);
        // Level 0 is the original.
        assert_eq!(pyr.level(0).width(), 100);
        // Each level shrinks.
        for l in 1..pyr.num_levels() {
            assert!(pyr.level(l).width() < pyr.level(l - 1).width());
            assert!(pyr.level(l).height() < pyr.level(l - 1).height());
        }
    }

    #[test]
    fn pyramid_stops_early_when_too_small() {
        let img = test_image(10, 10);
        // factor 2.0: level 1 = 5x5, level 2 = 3x3 (round(10/4)=3), level 3 = round(10/8)=1 < 3 → stop
        let pyr = Pyramid::build(&img.view(), 8, 2.0);

        assert!(pyr.num_levels() < 8, "should stop before 8 levels, got {}", pyr.num_levels());
        assert!(pyr.num_levels() >= 2);
        let last = pyr.level(pyr.num_levels() - 1);
        assert!(last.width() >= 3 && last.height() >= 3);
    }
}
