use coco_sys::coco_suite_t;
use std::{ffi::CString, ptr};

use crate::{observer::Observer, problem::Problem};

/// Index of a [`Problem`] in a [`Suite`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProblemIdx(pub(crate) usize);

/// Index of a function in a [`Suite`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FunctionIdx(usize);

/// Index of an instance in a [`Suite`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InstanceIdx(usize);

/// Index of a dimension in a [`Suite`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimensionIdx(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Name {
    Bbob,
    BbobBiobj,
    BbobBiobjExt,
    BbobLargescale,
    BbobConstrained,
    BbobMixint,
    BbobBiobjMixint,
    Toy,
}

impl Name {
    fn as_str(&self) -> &'static str {
        match self {
            Name::Bbob => "bbob",
            Name::BbobBiobj => "bbob-biobj",
            Name::BbobBiobjExt => "bbob-biobj-ext",
            Name::BbobLargescale => "bbob-largescale",
            Name::BbobConstrained => "bbob-constrained",
            Name::BbobMixint => "bbob-mixint",
            Name::BbobBiobjMixint => "bbob-biobj-mixint",
            Name::Toy => "toy",
        }
    }
}

/// A COCO suite
pub struct Suite {
    pub(crate) inner: *mut coco_suite_t,
    name: CString,
    instance: CString,
    options: CString,
}

impl Clone for Suite {
    fn clone(&self) -> Self {
        Suite::new_raw(
            self.name.clone(),
            self.instance.clone(),
            self.options.clone(),
        )
        .unwrap()
    }
}

unsafe impl Send for Suite {}

impl Suite {
    /// Instantiates the specified COCO suite.
    ///
    /// # suite_instance
    /// A string used for defining the suite instances. Two ways are supported:
    /// - "year: YEAR", where YEAR is the year of the BBOB workshop, includes the instances (to be) used in that
    /// year's workshop;
    /// - "instances: VALUES", where VALUES are instance numbers from 1 on written as a comma-separated list or a
    /// range m-n.
    ///
    /// # suite_options
    /// A string of pairs "key: value" used to filter the suite (especially useful for
    /// parallelizing the experiments). Supported options:
    /// - "dimensions: LIST", where LIST is the list of dimensions to keep in the suite (range-style syntax is
    /// not allowed here),
    /// - "dimension_indices: VALUES", where VALUES is a list or a range of dimension indices (starting from 1) to keep
    /// in the suite, and
    /// - "function_indices: VALUES", where VALUES is a list or a range of function indices (starting from 1) to keep
    /// in the suite, and
    /// - "instance_indices: VALUES", where VALUES is a list or a range of instance indices (starting from 1) to keep
    /// in the suite.
    pub fn new(name: Name, instance: &str, options: &str) -> Option<Suite> {
        let name = CString::new(name.as_str()).unwrap();
        let instance = CString::new(instance).unwrap();
        let options = CString::new(options).unwrap();

        Self::new_raw(name, instance, options)
    }

    fn new_raw(name: CString, instance: CString, options: CString) -> Option<Suite> {
        let inner =
            unsafe { coco_sys::coco_suite(name.as_ptr(), instance.as_ptr(), options.as_ptr()) };

        if inner.is_null() {
            None
        } else {
            Some(Suite {
                inner,
                name,
                instance,
                options,
            })
        }
    }

    /// Returns the function number in the suite in position function_idx (counting from 0).
    pub fn function_from_function_index(&self, function_idx: FunctionIdx) -> usize {
        unsafe { coco_sys::coco_suite_get_function_from_function_index(self.inner, function_idx.0) }
    }

    /// Returns the dimension number in the suite in position dimension_idx (counting from 0).
    pub fn dimension_from_dimension_index(&self, dimension_idx: DimensionIdx) -> usize {
        unsafe {
            coco_sys::coco_suite_get_dimension_from_dimension_index(self.inner, dimension_idx.0)
        }
    }

    /// Returns the instance number in the suite in position instance_idx (counting from 0).
    pub fn instance_from_instance_index(&self, instance_idx: InstanceIdx) -> usize {
        unsafe { coco_sys::coco_suite_get_instance_from_instance_index(self.inner, instance_idx.0) }
    }

    /// Returns the next problem or `None` when the suite completed.
    pub fn next_problem<'s>(&'s mut self, observer: Option<&mut Observer>) -> Option<Problem<'s>> {
        let observer = observer.map(|o| o.inner).unwrap_or(ptr::null_mut());
        let inner = unsafe { coco_sys::coco_suite_get_next_problem(self.inner, observer) };

        if inner.is_null() {
            return None;
        }

        unsafe {
            coco_sys::coco_suite_forget_current_problem(self.inner);
        }

        Some(Problem::new(inner, self))
    }

    /// Returns the problem of the suite defined by problem_idx.
    pub fn problem(&mut self, problem_idx: ProblemIdx) -> Option<Problem> {
        let inner = unsafe { coco_sys::coco_suite_get_problem(self.inner, problem_idx.0) };

        if inner.is_null() {
            return None;
        }

        Some(Problem::new(inner, self))
    }

    /// Returns the problem for the given function, dimension and instance.
    ///
    /// While a suite can contain multiple problems with equal function, dimension and instance, this
    /// function always returns the first problem in the suite with the given function, dimension and instance
    /// values. If the given values don't correspond to a problem, the function returns `None`.
    pub fn problem_by_function_dimension_instance(
        &mut self,
        function: usize,
        dimension: usize,
        instance: usize,
    ) -> Option<Problem> {
        let inner = unsafe {
            coco_sys::coco_suite_get_problem_by_function_dimension_instance(
                self.inner,
                function as usize,
                dimension as usize,
                instance as usize,
            )
        };

        if inner.is_null() {
            return None;
        }

        Some(Problem::new(inner, self))
    }

    /// Returns the total number of problems in the suite.
    pub fn number_of_problems(&self) -> usize {
        unsafe {
            coco_sys::coco_suite_get_number_of_problems(self.inner)
                .try_into()
                .unwrap()
        }
    }
}

impl Drop for Suite {
    fn drop(&mut self) {
        unsafe {
            coco_sys::coco_suite_free(self.inner);
        }
    }
}
