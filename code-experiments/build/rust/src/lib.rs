use std::ffi::CStr;

mod log_level;
pub use log_level::LogLevel;

mod random;
pub use random::RandomState;

pub mod suite;
pub use suite::Suite;

pub mod problem;
pub use problem::Problem;

pub mod observer;
pub use observer::Observer;

/// COCO’s version.
pub fn version() -> &'static str {
    unsafe { CStr::from_ptr(coco_sys::coco_version.as_ptr()) }
        .to_str()
        .unwrap()
}
