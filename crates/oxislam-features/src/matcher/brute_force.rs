use oxislam_image::parallel::{MaybeSend, MaybeSync, par_filter_map};

use crate::traits::matcher::{DescriptorMatch, DescriptorMatcher, Distance};

const DEFAULT_RATIO: f32 = 0.75;

/// Brute-force descriptor matcher.
///
/// Compares every query descriptor against every training descriptor,
/// returning the best match for each query. Applies Lowe's ratio test
/// to reject ambiguous matches (default ratio: 0.75).
pub struct BruteForceMatcher<Dist> {
    distance: Dist,
    ratio: f32,
}

impl<Dist> BruteForceMatcher<Dist> {
    pub fn new(distance: Dist) -> Self { Self { distance, ratio: DEFAULT_RATIO } }

    /// Set the ratio test threshold (0.0–1.0). A match is rejected when
    /// `best_distance >= ratio * second_best_distance`. Lower values are
    /// more selective. Set to 1.0 to disable.
    pub fn with_ratio(mut self, ratio: f32) -> Self {
        assert!((0.0..=1.0).contains(&ratio), "ratio must be in 0.0..=1.0, got {ratio}");
        self.ratio = ratio;
        self
    }
}

impl<D: MaybeSend + MaybeSync, Dist: Distance<D> + MaybeSync> DescriptorMatcher<D>
    for BruteForceMatcher<Dist>
{
    fn match_descriptors(&self, source: &[D], reference: &[D]) -> Vec<DescriptorMatch> {
        par_filter_map(source.iter().enumerate(), |(si, sd): (usize, &D)| {
            let (mut best, mut second_best) = (f32::MAX, f32::MAX);
            let mut best_ri = 0;

            for (ri, rd) in reference.iter().enumerate() {
                let d = self.distance.distance(sd, rd);
                if d < best {
                    second_best = best;
                    best = d;
                    best_ri = ri;
                } else if d < second_best {
                    second_best = d;
                }
            }

            if best >= self.ratio * second_best {
                return None;
            }

            Some(DescriptorMatch { source_idx: si, reference_idx: best_ri, distance: best })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::brief::BriefDescriptor;
    use crate::matcher::distance::Hamming;

    fn desc(bits: [u64; 1]) -> BriefDescriptor<1> { BriefDescriptor::new(bits) }

    #[test]
    fn finds_closest_match() {
        let matcher = BruteForceMatcher::new(Hamming).with_ratio(1.0);
        let source = [desc([0b0000])];
        let reference = [desc([0b1111]), desc([0b0001]), desc([0b0111])];

        let matches = matcher.match_descriptors(&source, &reference);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].reference_idx, 1);
        assert_eq!(matches[0].distance, 1.0);
    }

    #[test]
    fn ratio_test_rejects_ambiguous() {
        let matcher = BruteForceMatcher::new(Hamming);
        // best=1 bit, second_best=2 bits → ratio 0.5 < 0.75, passes
        let source = [desc([0b0000])];
        let clear = [desc([0b1111]), desc([0b0001]), desc([0b0011])];
        assert_eq!(matcher.match_descriptors(&source, &clear).len(), 1);

        // best=3 bits, second_best=4 bits → ratio 0.75, rejected (>=)
        let ambiguous = [desc([0b0111]), desc([0b1111])];
        let source = [desc([0b0000])];
        assert_eq!(matcher.match_descriptors(&source, &ambiguous).len(), 0);
    }
}
