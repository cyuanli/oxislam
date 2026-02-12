//! Descriptor matching algorithms and distance metrics.

pub mod brute_force;
pub mod distance;

pub use brute_force::BruteForceMatcher;
pub use distance::{Hamming, L2Squared};
