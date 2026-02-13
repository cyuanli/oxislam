//! Shared utilities for the oxislam workspace.

#[cfg(feature = "tracing")]
#[doc(hidden)]
pub use tracing as __tracing;

/// Create a tracing span and enter it. Returns a guard that exits the span on drop.
/// When the `tracing` feature is disabled this is a zero-cost no-op.
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

/// Emit a tracing event at the given level.
/// When the `tracing` feature is disabled this is a zero-cost no-op.
///
/// Usage: `event!(Level::INFO, count = 42)` or `event!(Level::DEBUG, "message")`
#[doc(hidden)]
#[macro_export]
macro_rules! __event {
    ($level:expr, $($arg:tt)*) => {{
        #[cfg(feature = "tracing")]
        $crate::__tracing::event!($level, severity = %$level, $($arg)*);
    }};
}

/// Internal tracing helpers — no-ops when the `tracing` feature is disabled.
#[doc(hidden)]
pub mod trace {
    pub use crate::__event as event;
    pub use crate::__span as span;
}
