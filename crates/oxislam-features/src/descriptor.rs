#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum Descriptor {
    Binary(Vec<u8>),
    Float(Vec<f32>),
}

impl Descriptor {
    pub fn kind(&self) -> DescriptorKind {
        match self {
            Descriptor::Binary(_) => DescriptorKind::Binary,
            Descriptor::Float(_) => DescriptorKind::Float,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Descriptor::Binary(v) => v.len(),
            Descriptor::Float(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_binary(&self) -> Option<&[u8]> {
        match self {
            Descriptor::Binary(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<&[f32]> {
        match self {
            Descriptor::Float(v) => Some(v),
            _ => None,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DescriptorKind {
    Binary,
    Float,
}
