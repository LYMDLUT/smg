//! JSON as models write it: values that are still arriving.
//!
//! Tool-call arguments stream in as a JSON prefix that grows with every chunk. [`partial`] parses
//! such a prefix into the value it determines so far and says how many bytes it understood, which
//! is what an argument stream needs to emit prefix-stable fragments. The assembler that turns a
//! growing prefix into fragments builds on it in a later change.

pub mod partial;

pub use partial::{is_complete, PartialJson, PartialJsonError};
