use crate::pixel::Gray;

#[derive(Debug)]
pub struct Image<P> {
    width: usize,
    height: usize,
    stride: usize,
    data: Vec<P>,
}

#[derive(Debug)]
pub struct ImageView<'a, P> {
    width: usize,
    height: usize,
    stride: usize,
    data: &'a [P],
}

#[derive(Debug)]
pub struct ImageViewMut<'a, P> {
    width: usize,
    height: usize,
    stride: usize,
    data: &'a mut [P],
}

impl<P> Image<P> {
    pub fn new(width: usize, height: usize, stride: usize, data: Vec<P>) -> Self {
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
    pub fn data(&self) -> &[P] { &self.data }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [P] { &mut self.data }

    #[inline]
    pub fn index(&self, x: usize, y: usize) -> usize {
        debug_assert!(x < self.width);
        debug_assert!(y < self.height);
        y * self.stride + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> &P {
        let idx = self.index(x, y);
        &self.data[idx]
    }

    #[inline]
    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut P {
        let idx = self.index(x, y);
        &mut self.data[idx]
    }

    pub fn view(&self) -> ImageView<'_, P> {
        ImageView { width: self.width, height: self.height, stride: self.stride, data: &self.data }
    }

    pub fn view_mut(&mut self) -> ImageViewMut<'_, P> {
        ImageViewMut {
            width: self.width,
            height: self.height,
            stride: self.stride,
            data: &mut self.data,
        }
    }
}

impl<'a, P> ImageView<'a, P> {
    pub fn new(data: &'a [P], width: usize, height: usize, stride: usize) -> Self {
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
    pub fn data(&self) -> &'a [P] { self.data }

    #[inline]
    pub fn index(&self, x: usize, y: usize) -> usize {
        debug_assert!(x < self.width);
        debug_assert!(y < self.height);
        y * self.stride + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> &P {
        let idx = self.index(x, y);
        &self.data[idx]
    }

    pub fn subview(
        &self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> Option<ImageView<'a, P>> {
        if width == 0 || height == 0 {
            return None;
        }
        if x + width > self.width || y + height > self.height {
            return None;
        }

        let offset = y * self.stride + x;
        Some(ImageView { width, height, stride: self.stride, data: &self.data[offset..] })
    }

    pub fn pixels(&self) -> impl Iterator<Item = &P> {
        (0..self.height).flat_map(move |y| {
            let row_start = y * self.stride;
            (0..self.width).map(move |x| &self.data[row_start + x])
        })
    }

    pub fn rows(&self) -> impl Iterator<Item = &[P]> {
        (0..self.height).map(move |y| {
            let start = y * self.stride;
            &self.data[start..start + self.width]
        })
    }

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

impl<'a, P> ImageViewMut<'a, P> {
    pub fn new(data: &'a mut [P], width: usize, height: usize, stride: usize) -> Self {
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
    pub fn data(&self) -> &[P] { self.data }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [P] { self.data }

    #[inline]
    pub fn index(&self, x: usize, y: usize) -> usize {
        debug_assert!(x < self.width);
        debug_assert!(y < self.height);
        y * self.stride + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> &P {
        let idx = self.index(x, y);
        &self.data[idx]
    }

    #[inline]
    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut P {
        let idx = self.index(x, y);
        &mut self.data[idx]
    }
}

impl<P: Clone> Image<P> {
    pub fn filled(width: usize, height: usize, value: P) -> Self {
        let stride = width;
        let data = vec![value; width * height];

        Self::new(width, height, stride, data)
    }
}

pub trait RawPixel: Copy + 'static {}
impl RawPixel for u8 {}
impl RawPixel for f32 {}

impl<T: RawPixel> Image<Gray<T>> {
    pub fn from_raw(width: usize, height: usize, stride: usize, data: Vec<T>) -> Self {
        assert!(width > 0);
        assert!(height > 0);
        assert!(stride >= width);
        assert!(data.len() >= stride * height);

        let gray_data = unsafe {
            let ptr = data.as_ptr() as *const Gray<T>;
            let len = data.len();
            let cap = data.capacity();
            std::mem::forget(data);
            Vec::from_raw_parts(ptr as *mut Gray<T>, len, cap)
        };

        Self { width, height, stride, data: gray_data }
    }

    pub fn into_raw(self) -> (usize, usize, usize, Vec<T>) {
        let width = self.width;
        let height = self.height;
        let stride = self.stride;

        let raw_data = unsafe {
            let ptr = self.data.as_ptr() as *const T;
            let len = self.data.len();
            let cap = self.data.capacity();
            std::mem::forget(self.data);
            Vec::from_raw_parts(ptr as *mut T, len, cap)
        };

        (width, height, stride, raw_data)
    }

    pub fn as_raw(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const T, self.data.len()) }
    }

    pub fn as_raw_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut T, self.data.len()) }
    }
}
