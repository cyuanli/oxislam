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

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: 4x3 grid with stride=5 (one padding element per row).
    ///
    ///  row0: [ 1,  2,  3,  4, 0]
    ///  row1: [ 5,  6,  7,  8, 0]
    ///  row2: [ 9, 10, 11, 12, 0]
    fn padded_grid() -> Grid2D<i32> {
        let mut data = Vec::with_capacity(5 * 3);
        for row in 0..3 {
            for col in 0..4 {
                data.push(row * 4 + col + 1);
            }
            data.push(0); // padding
        }
        Grid2D::new(4, 3, 5, data)
    }

    #[test]
    fn get_with_stride() {
        let grid = padded_grid();

        // First row
        assert_eq!(*grid.get(0, 0), 1);
        assert_eq!(*grid.get(3, 0), 4);
        // Second row — skips the padding element in memory
        assert_eq!(*grid.get(0, 1), 5);
        assert_eq!(*grid.get(3, 1), 8);
        // Last element
        assert_eq!(*grid.get(3, 2), 12);
    }

    #[test]
    fn subview_with_stride() {
        let grid = padded_grid();
        let view = grid.view();

        let sub = view.subview(1, 1, 2, 2).unwrap();
        assert_eq!(sub.stride(), 5); // inherits parent stride
        assert_eq!(*sub.get(0, 0), 6);
        assert_eq!(*sub.get(1, 0), 7);
        assert_eq!(*sub.get(0, 1), 10);
        assert_eq!(*sub.get(1, 1), 11);
    }

    #[test]
    fn subview_out_of_bounds_returns_none() {
        let grid = padded_grid(); // 4x3
        let view = grid.view();

        assert!(view.subview(3, 0, 2, 1).is_none()); // x+w > width
        assert!(view.subview(0, 2, 1, 2).is_none()); // y+h > height
        assert!(view.subview(0, 0, 0, 1).is_none()); // zero width
    }

    #[test]
    fn iter_skips_padding() {
        let grid = padded_grid();
        let view = grid.view();

        let values: Vec<i32> = view.iter().copied().collect();
        assert_eq!(values, (1..=12).collect::<Vec<_>>());
    }
}
