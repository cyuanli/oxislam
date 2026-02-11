use crate::grid::{Grid2D, Grid2DView, Grid2DViewMut};
use crate::pixel::Gray;

#[derive(Debug)]
pub struct Image<P> {
    grid: Grid2D<P>,
}

#[derive(Debug)]
pub struct ImageView<'a, P> {
    grid: Grid2DView<'a, P>,
}

#[derive(Debug)]
pub struct ImageViewMut<'a, P> {
    grid: Grid2DViewMut<'a, P>,
}

// --- Image<P> ---

impl<P> Image<P> {
    pub fn new(width: usize, height: usize, stride: usize, data: Vec<P>) -> Self {
        Self { grid: Grid2D::new(width, height, stride, data) }
    }

    #[inline]
    pub fn width(&self) -> usize { self.grid.width() }

    #[inline]
    pub fn height(&self) -> usize { self.grid.height() }

    #[inline]
    pub fn stride(&self) -> usize { self.grid.stride() }

    #[inline]
    pub fn data(&self) -> &[P] { self.grid.data() }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [P] { self.grid.data_mut() }

    #[inline]
    pub fn index(&self, x: usize, y: usize) -> usize { self.grid.index(x, y) }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> &P { self.grid.get(x, y) }

    #[inline]
    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut P { self.grid.get_mut(x, y) }

    pub fn view(&self) -> ImageView<'_, P> { ImageView { grid: self.grid.view() } }

    pub fn view_mut(&mut self) -> ImageViewMut<'_, P> {
        ImageViewMut { grid: self.grid.view_mut() }
    }
}

impl<P: Clone> Image<P> {
    pub fn filled(width: usize, height: usize, value: P) -> Self {
        Self { grid: Grid2D::filled(width, height, value) }
    }
}

// --- ImageView<P> ---

impl<'a, P> ImageView<'a, P> {
    pub fn new(data: &'a [P], width: usize, height: usize, stride: usize) -> Self {
        Self { grid: Grid2DView::new(data, width, height, stride) }
    }

    #[inline]
    pub fn width(&self) -> usize { self.grid.width() }

    #[inline]
    pub fn height(&self) -> usize { self.grid.height() }

    #[inline]
    pub fn stride(&self) -> usize { self.grid.stride() }

    #[inline]
    pub fn data(&self) -> &'a [P] { self.grid.data() }

    #[inline]
    pub fn index(&self, x: usize, y: usize) -> usize { self.grid.index(x, y) }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> &P { self.grid.get(x, y) }

    pub fn subview(
        &self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> Option<ImageView<'a, P>> {
        self.grid.subview(x, y, width, height).map(|g| ImageView { grid: g })
    }

    pub fn pixels(&self) -> impl Iterator<Item = &P> { self.grid.iter() }

    pub fn rows(&self) -> impl Iterator<Item = &[P]> { self.grid.rows() }

    pub fn patch(&self, cx: f32, cy: f32, size: usize) -> Option<ImageView<'a, P>> {
        let half = (size / 2) as isize;
        let cx = cx.round() as isize;
        let cy = cy.round() as isize;

        let x = cx - half;
        let y = cy - half;

        if x < 0 || y < 0 {
            return None;
        }

        self.subview(x as usize, y as usize, size, size)
    }
}

// --- ImageViewMut<P> ---

impl<'a, P> ImageViewMut<'a, P> {
    pub fn new(data: &'a mut [P], width: usize, height: usize, stride: usize) -> Self {
        Self { grid: Grid2DViewMut::new(data, width, height, stride) }
    }

    #[inline]
    pub fn width(&self) -> usize { self.grid.width() }

    #[inline]
    pub fn height(&self) -> usize { self.grid.height() }

    #[inline]
    pub fn stride(&self) -> usize { self.grid.stride() }

    #[inline]
    pub fn data(&self) -> &[P] { self.grid.data() }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [P] { self.grid.data_mut() }

    #[inline]
    pub fn index(&self, x: usize, y: usize) -> usize { self.grid.index(x, y) }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> &P { self.grid.get(x, y) }

    #[inline]
    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut P { self.grid.get_mut(x, y) }
}

// --- Raw pixel conversion (Image-specific) ---

pub trait RawPixel: Copy + 'static {}
impl RawPixel for u8 {}
impl RawPixel for f32 {}

impl<T: RawPixel> Image<Gray<T>> {
    pub fn from_raw(width: usize, height: usize, stride: usize, data: Vec<T>) -> Self {
        assert!(width > 0);
        assert!(height > 0);
        assert!(stride >= width);
        assert!(data.len() >= stride * height);

        // SAFETY: Gray<T> is repr(transparent), so its layout is identical to T.
        // We're just reinterpreting the vector's memory to change the type.
        let gray_data = unsafe {
            let ptr = data.as_ptr() as *const Gray<T>;
            let len = data.len();
            let cap = data.capacity();
            std::mem::forget(data);
            Vec::from_raw_parts(ptr as *mut Gray<T>, len, cap)
        };

        Self { grid: Grid2D::new(width, height, stride, gray_data) }
    }

    pub fn into_raw(self) -> (usize, usize, usize, Vec<T>) {
        let width = self.grid.width();
        let height = self.grid.height();
        let stride = self.grid.stride();
        let data = self.grid.into_data();

        // SAFETY: Gray<T> is repr(transparent), so its layout is identical to T.
        // We're reinterpreting back to the original raw type.
        let raw_data = unsafe {
            let ptr = data.as_ptr() as *const T;
            let len = data.len();
            let cap = data.capacity();
            std::mem::forget(data);
            Vec::from_raw_parts(ptr as *mut T, len, cap)
        };

        (width, height, stride, raw_data)
    }

    pub fn as_raw(&self) -> &[T] {
        let data = self.grid.data();
        // SAFETY: Gray<T> is repr(transparent), so a &[Gray<T>] can be safely reinterpreted as &[T].
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, data.len()) }
    }

    pub fn as_raw_mut(&mut self) -> &mut [T] {
        let data = self.grid.data_mut();
        // SAFETY: Gray<T> is repr(transparent), so &mut [Gray<T>] can be safely reinterpreted as &mut [T].
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut T, data.len()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_raw_into_raw_roundtrip() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let img = Image::<Gray<f32>>::from_raw(3, 2, 3, data.clone());

        assert_eq!(img.get(0, 0).value, 1.0);
        assert_eq!(img.get(1, 0).value, 2.0);
        assert_eq!(img.get(2, 0).value, 3.0);
        assert_eq!(img.get(0, 1).value, 4.0);
        assert_eq!(img.get(1, 1).value, 5.0);
        assert_eq!(img.get(2, 1).value, 6.0);
        assert_eq!(img.as_raw(), &data[..]);

        let (_w, _h, _s, raw) = img.into_raw();

        assert_eq!(raw, data);
    }

}
