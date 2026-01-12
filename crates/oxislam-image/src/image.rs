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
    #[inline]
    pub fn width(&self) -> usize { self.width }
    #[inline]
    pub fn height(&self) -> usize { self.height }
    #[inline]
    pub fn stride(&self) -> usize { self.stride }
    #[inline]
    pub fn data(&self) -> &'a [P] { self.data }
}

impl<'a, P> ImageViewMut<'a, P> {
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
}

impl<P: Clone> Image<P> {
    pub fn filled(width: usize, height: usize, value: P) -> Self {
        let stride = width;
        let data = vec![value; width * height];

        Self::new(width, height, stride, data)
    }
}
