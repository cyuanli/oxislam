/// A 2D grid of values with stride-based storage.
///
/// This is the generic storage layer used by `Image<P>` and also suitable
/// for non-image 2D data such as response maps (`Grid2D<f32>`).
#[derive(Debug)]
pub struct Grid2D<T> {
    width: usize,
    height: usize,
    stride: usize,
    data: Vec<T>,
}

/// An immutable view into a [`Grid2D`].
#[derive(Debug)]
pub struct Grid2DView<'a, T> {
    width: usize,
    height: usize,
    stride: usize,
    data: &'a [T],
}

/// A mutable view into a [`Grid2D`].
#[derive(Debug)]
pub struct Grid2DViewMut<'a, T> {
    width: usize,
    height: usize,
    stride: usize,
    data: &'a mut [T],
}

impl<T> Grid2D<T> {
    pub fn new(width: usize, height: usize, stride: usize, data: Vec<T>) -> Self {
        assert!(width > 0);
        assert!(height > 0);
        assert!(stride >= width);
        assert!(data.len() >= stride * height);

        Self { width, height, stride, data }
    }

    #[inline]
    pub fn width(&self) -> usize { self.width }

    #[inline]
    pub fn height(&self) -> usize { self.height }

    #[inline]
    pub fn stride(&self) -> usize { self.stride }

    #[inline]
    pub fn data(&self) -> &[T] { &self.data }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] { &mut self.data }

    #[inline]
    pub fn index(&self, x: usize, y: usize) -> usize {
        debug_assert!(x < self.width);
        debug_assert!(y < self.height);
        y * self.stride + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> &T {
        let idx = self.index(x, y);
        &self.data[idx]
    }

    #[inline]
    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut T {
        let idx = self.index(x, y);
        &mut self.data[idx]
    }

    pub fn into_data(self) -> Vec<T> { self.data }

    pub fn view(&self) -> Grid2DView<'_, T> {
        Grid2DView { width: self.width, height: self.height, stride: self.stride, data: &self.data }
    }

    pub fn view_mut(&mut self) -> Grid2DViewMut<'_, T> {
        Grid2DViewMut {
            width: self.width,
            height: self.height,
            stride: self.stride,
            data: &mut self.data,
        }
    }
}

impl<T: Clone> Grid2D<T> {
    pub fn filled(width: usize, height: usize, value: T) -> Self {
        let stride = width;
        let data = vec![value; width * height];

        Self::new(width, height, stride, data)
    }
}

impl<'a, T> Grid2DView<'a, T> {
    pub fn new(data: &'a [T], width: usize, height: usize, stride: usize) -> Self {
        assert!(width > 0);
        assert!(height > 0);
        assert!(stride >= width);
        assert!(data.len() >= stride * height);

        Self { width, height, stride, data }
    }

    #[inline]
    pub fn width(&self) -> usize { self.width }

    #[inline]
    pub fn height(&self) -> usize { self.height }

    #[inline]
    pub fn stride(&self) -> usize { self.stride }

    #[inline]
    pub fn data(&self) -> &'a [T] { self.data }

    #[inline]
    pub fn index(&self, x: usize, y: usize) -> usize {
        debug_assert!(x < self.width);
        debug_assert!(y < self.height);
        y * self.stride + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> &T {
        let idx = self.index(x, y);
        &self.data[idx]
    }

    pub fn subview(
        &self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> Option<Grid2DView<'a, T>> {
        if width == 0 || height == 0 {
            return None;
        }
        if x + width > self.width || y + height > self.height {
            return None;
        }

        let offset = y * self.stride + x;
        Some(Grid2DView { width, height, stride: self.stride, data: &self.data[offset..] })
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        (0..self.height).flat_map(move |y| {
            let row_start = y * self.stride;
            (0..self.width).map(move |x| &self.data[row_start + x])
        })
    }

    pub fn rows(&self) -> impl Iterator<Item = &[T]> {
        (0..self.height).map(move |y| {
            let start = y * self.stride;
            &self.data[start..start + self.width]
        })
    }

}

impl<'a, T> Grid2DViewMut<'a, T> {
    pub fn new(data: &'a mut [T], width: usize, height: usize, stride: usize) -> Self {
        assert!(width > 0);
        assert!(height > 0);
        assert!(stride >= width);
        assert!(data.len() >= stride * height);

        Self { width, height, stride, data }
    }

    #[inline]
    pub fn width(&self) -> usize { self.width }

    #[inline]
    pub fn height(&self) -> usize { self.height }

    #[inline]
    pub fn stride(&self) -> usize { self.stride }

    #[inline]
    pub fn data(&self) -> &[T] { self.data }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] { self.data }

    #[inline]
    pub fn index(&self, x: usize, y: usize) -> usize {
        debug_assert!(x < self.width);
        debug_assert!(y < self.height);
        y * self.stride + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> &T {
        let idx = self.index(x, y);
        &self.data[idx]
    }

    #[inline]
    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut T {
        let idx = self.index(x, y);
        &mut self.data[idx]
    }
}
