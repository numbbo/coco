use std::ffi::CStr;

/// COCO's log levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// Log only errors.
    Error,
    /// Log only warnings and errors.
    Warning,
    /// Log everything except debug messages.
    Info,
    /// Log everyhting.
    Debug,
}

impl Default for LogLevel {
    fn default() -> Self {
        LogLevel::Info
    }
}

impl LogLevel {
    /// Sets COCO's log level.
    pub fn set(self) {
        let level = CStr::from_bytes_with_nul(match self {
            LogLevel::Error => b"error\0",
            LogLevel::Warning => b"warning\0",
            LogLevel::Info => b"info\0",
            LogLevel::Debug => b"debug\0",
        })
        .unwrap();

        unsafe {
            coco_sys::coco_set_log_level(level.as_ptr());
        }
    }
}
