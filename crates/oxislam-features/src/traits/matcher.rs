#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Match {
    pub index: usize,
    pub distance: f32,
}

impl Match {
    pub fn new(index: usize, distance: f32) -> Self { Self { index, distance } }
}

pub trait Matcher<D> {
    fn add(&mut self, descriptors: &[D]);

    fn clear(&mut self);

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool { self.len() == 0 }

    fn knn(&self, query: &D, k: usize) -> Vec<Match>;

    fn radius(&self, query: &D, max_distance: f32) -> Vec<Match>;

    fn find_match(&self, query: &D) -> Option<Match> { self.knn(query, 1).into_iter().next() }
}
