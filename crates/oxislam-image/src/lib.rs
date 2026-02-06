//! Image processing utilities.
//!
//! Provides image types, filtering operations, pixel formats, and parallel processing utilities.

pub mod filter;
pub mod image;
pub mod parallel;
pub mod pixel;

pub use filter::{Kernel, apply_kernel, gaussian_3x3, gaussian_5x5, sobel};
pub use image::ConvertTo;
pub use parallel::{MaybeSend, MaybeSync};
pub use pixel::{Gray, Rgb};
