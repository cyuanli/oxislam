#[repr(transparent)]
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
