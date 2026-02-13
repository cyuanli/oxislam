//! Image processing utilities.
//!
//! Provides image types, filtering operations, pixel formats, and parallel processing utilities.
//!
//! Enable the `image` feature for `From`/`Into` conversions with the [`image`] crate.

pub mod filter;
pub mod grid;
pub mod image;
pub mod parallel;
pub mod pixel;

pub use filter::{Kernel, apply_kernel, gaussian, resize_bilinear, sobel};
pub use grid::{Grid2D, Grid2DView, Grid2DViewMut};
pub use image::ConvertTo;
pub use parallel::{MaybeSend, MaybeSync};
pub use pixel::{Gray, Rgb};
