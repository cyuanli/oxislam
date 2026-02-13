//! Shared utilities for the oxislam workspace.

#[cfg(feature = "tracing")]
#[doc(hidden)]
pub use tracing as __tracing;

#[doc(hidden)]
#[macro_export]
macro_rules! __span {
    ($name:literal) => {{
        #[cfg(feature = "tracing")]
        let _span = $crate::__tracing::info_span!($name).entered();
        #[cfg(not(feature = "tracing"))]
        let _span = ();
        _span
    }};
    ($name:literal, $($field:tt)*) => {{
        #[cfg(feature = "tracing")]
        let _span = $crate::__tracing::info_span!($name, $($field)*).entered();
        #[cfg(not(feature = "tracing"))]
        let _span = ();
        _span
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __event {
    ($level:expr, $($arg:tt)*) => {{
        #[cfg(feature = "tracing")]
        $crate::__tracing::event!($level, severity = %$level, $($arg)*);
    }};
}

#[doc(hidden)]
pub mod trace {
    pub use crate::__event as event;
    pub use crate::__span as span;
}
