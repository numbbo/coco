use coco_sys::coco_random_state_t;

/// COCO specific random number generator.
pub struct RandomState {
    pub(crate) inner: *mut coco_random_state_t,
}

impl RandomState {
    /// Creates a new random number state using the given seed.
    pub fn new(seed: u32) -> Self {
        let inner = unsafe { coco_sys::coco_random_new(seed) };

        RandomState { inner }
    }

    /// Generates an approximately normal random number.
    pub fn normal(&mut self) -> f64 {
        unsafe { coco_sys::coco_random_normal(self.inner) }
    }

    /// Returns one uniform [0, 1) random value.
    pub fn uniform(&mut self) -> f64 {
        unsafe { coco_sys::coco_random_uniform(self.inner) }
    }
}

impl Drop for RandomState {
    fn drop(&mut self) {
        unsafe {
            coco_sys::coco_random_free(self.inner);
        }
    }
}
