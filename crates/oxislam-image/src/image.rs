pub struct Image<P> {
    pub width: u32,
    pub height: u32,
    pub data: Vec<P>,
}

pub struct ImageView<'a, P> {
    pub width: u32,
    pub height: u32,
    pub data: &'a [P],
}

pub struct ImageViewMut<'a, P> {
    pub width: u32,
    pub height: u32,
    pub data: &'a mut [P],
}
