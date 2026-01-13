#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gray<T> {
    pub value: T,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rgb<T> {
    pub r: T,
    pub g: T,
    pub b: T,
}

impl<T> Gray<T> {
    #[inline]
    pub const fn new(value: T) -> Self { Self { value } }
}

impl<T> Rgb<T> {
    #[inline]
    pub const fn new(r: T, g: T, b: T) -> Self { Self { r, g, b } }
}

pub trait ToGrayscale {
    fn to_grayscale(&self) -> f32;
}

impl ToGrayscale for Gray<f32> {
    #[inline]
    fn to_grayscale(&self) -> f32 { self.value }
}

impl ToGrayscale for Rgb<f32> {
    #[inline]
    fn to_grayscale(&self) -> f32 { 0.299 * self.r + 0.587 * self.g + 0.114 * self.b }
}

impl ToGrayscale for Gray<u8> {
    #[inline]
    fn to_grayscale(&self) -> f32 { self.value as f32 / 255.0 }
}

impl ToGrayscale for Rgb<u8> {
    #[inline]
    fn to_grayscale(&self) -> f32 {
        let gray_u8 = (self.r as u32 * 77 + self.g as u32 * 150 + self.b as u32 * 29) >> 8;
        gray_u8 as f32 / 255.0
    }
}
