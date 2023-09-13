use coco_sys::coco_problem_t;
use std::{ffi::CStr, marker::PhantomData, ops::RangeInclusive};

use crate::{
    suite::{self, Suite},
    Observer,
};

/// A specific problem instance.
///
/// Instances can be optained using [Suite::next_problem]
/// and [Suite::problem_by_function_dimension_instance].
pub struct Problem<'suite> {
    pub(crate) inner: *mut coco_problem_t,
    _phantom: PhantomData<&'suite Suite>,
}

unsafe impl Send for Problem<'_> {}

impl<'suite> Problem<'suite> {
    pub(crate) fn new(inner: *mut coco_problem_t, _suite: &'suite Suite) -> Self {
        Problem {
            inner,
            _phantom: PhantomData,
        }
    }
}

impl Problem<'_> {
    /// Returns the ID of the problem.
    ///
    /// For the `toy` suite this is
    /// - `{function-name}_d{dimension}`
    ///
    /// For `bbob` it is
    /// - bbob_f{function-index}_i{instance}_d{dimension}
    pub fn id(&self) -> &str {
        unsafe {
            CStr::from_ptr(coco_sys::coco_problem_get_id(self.inner))
                .to_str()
                .unwrap()
        }
    }

    /// Returns the name of the problem.
    pub fn name(&self) -> &str {
        unsafe {
            CStr::from_ptr(coco_sys::coco_problem_get_name(self.inner))
                .to_str()
                .unwrap()
        }
    }

    /// Returns the type of the problem.
    pub fn typ(&self) -> &str {
        unsafe {
            CStr::from_ptr(coco_sys::coco_problem_get_type(self.inner))
                .to_str()
                .unwrap()
        }
    }

    /// Adds an observer to the given problem.
    pub fn add_observer(&mut self, observer: &Observer) {
        // The Python bindings also mutate the problem instead of returning a new one.
        self.inner = unsafe { coco_sys::coco_problem_add_observer(self.inner, observer.inner) };

        assert!(!self.inner.is_null())
    }

    /// Removes an observer to the given problem.
    pub fn remove_observer(&mut self, observer: &Observer) {
        // The Python bindings also mutate the problem instead of returning a new one.
        self.inner = unsafe { coco_sys::coco_problem_remove_observer(self.inner, observer.inner) };

        assert!(!self.inner.is_null())
    }

    /// Returns the problem index of the problem in its current suite.
    pub fn suite_index(&self) -> suite::ProblemIdx {
        let idx = unsafe { coco_sys::coco_problem_get_suite_dep_index(self.inner) };

        suite::ProblemIdx(idx)
    }

    /// Evaluates the problem at `x` and returns the result in `y`.
    ///
    /// The length of `x` must match [Problem::dimension] and the
    /// length of `y` must match [Problem::number_of_objectives].
    pub fn evaluate_function(&mut self, x: &[f64], y: &mut [f64]) {
        assert_eq!(self.dimension(), x.len());
        assert_eq!(self.number_of_objectives(), y.len());

        unsafe {
            coco_sys::coco_evaluate_function(self.inner, x.as_ptr(), y.as_mut_ptr());
        }
    }

    /// Evaluates the problem constraints in point x and save the result in y.
    ///
    /// The length of `x` must match [Problem::dimension] and the
    /// length of `y` must match [Problem::number_of_constraints].
    pub fn evaluate_constraint(&mut self, x: &[f64], y: &mut [f64]) {
        assert_eq!(self.dimension(), x.len());
        assert_eq!(self.number_of_constraints(), y.len());

        unsafe {
            coco_sys::coco_evaluate_constraint(self.inner, x.as_ptr(), y.as_mut_ptr());
        }
    }

    /// Returns true if a previous evaluation hit the target value.
    pub fn final_target_hit(&self) -> bool {
        unsafe { coco_sys::coco_problem_final_target_hit(self.inner) == 1 }
    }

    /// Returns the optimal function value + delta of the problem
    pub fn final_target_value(&self) -> f64 {
        unsafe { coco_sys::coco_problem_get_final_target_fvalue1(self.inner) }
    }

    /// Returns the optimal function value of the problem
    ///
    /// To check whether the target has been reached use [[Problem::final_target_value]]
    /// or [[Problem::final_target_hit]] instead.
    pub fn best_value(&self) -> f64 {
        unsafe { coco_sys::coco_problem_get_best_value(self.inner) }
    }

    /// Returns the best observed value for the first objective.
    pub fn best_observed_value(&self) -> f64 {
        unsafe { coco_sys::coco_problem_get_best_observed_fvalue1(self.inner) }
    }

    /// Returns the dimension of the problem.
    pub fn dimension(&self) -> usize {
        unsafe {
            coco_sys::coco_problem_get_dimension(self.inner)
                .try_into()
                .unwrap()
        }
    }

    /// Returns the number of objectives of the problem.
    pub fn number_of_objectives(&self) -> usize {
        unsafe {
            coco_sys::coco_problem_get_number_of_objectives(self.inner)
                .try_into()
                .unwrap()
        }
    }

    /// Returns the number of constraints of the problem.
    pub fn number_of_constraints(&self) -> usize {
        unsafe {
            coco_sys::coco_problem_get_number_of_constraints(self.inner)
                .try_into()
                .unwrap()
        }
    }

    /// Returns the numver of integer variables of the problem.
    ///
    /// The first `n` variables will be integers then.
    /// Returns `0` if all variables are continuous.
    pub fn number_of_integer_variables(&self) -> usize {
        unsafe {
            coco_sys::coco_problem_get_number_of_integer_variables(self.inner)
                .try_into()
                .unwrap()
        }
    }

    /// Returns the upper and lover bounds of the problem.
    pub fn ranges_of_interest(&self) -> Vec<RangeInclusive<f64>> {
        let dimension = self.dimension() as isize;
        unsafe {
            let smallest = coco_sys::coco_problem_get_smallest_values_of_interest(self.inner);
            let largest = coco_sys::coco_problem_get_largest_values_of_interest(self.inner);

            (0..dimension)
                .into_iter()
                .map(|i| (*smallest.offset(i))..=(*largest.offset(i)))
                .collect()
        }
    }

    /// Returns how often this instance has been evaluated.
    pub fn evaluations(&self) -> u64 {
        unsafe {
            #[allow(clippy::useless_conversion)]
            coco_sys::coco_problem_get_evaluations(self.inner)
                .try_into()
                .unwrap()
        }
    }

    /// Returns how often this instances constrants have been evaluated.
    pub fn evaluations_constraints(&self) -> u64 {
        unsafe {
            #[allow(clippy::useless_conversion)]
            coco_sys::coco_problem_get_evaluations_constraints(self.inner)
                .try_into()
                .unwrap()
        }
    }

    /// Writes a feasible initial solution into `x`.
    ///
    /// If the problem does not provide a specific solution,
    /// it will be the center of the problem's region of interest.
    pub fn initial_solution(&self, x: &mut [f64]) {
        assert_eq!(self.dimension(), x.len());
        unsafe {
            coco_sys::coco_problem_get_initial_solution(self.inner, x.as_mut_ptr());
        }
    }
}

impl Drop for Problem<'_> {
    fn drop(&mut self) {
        unsafe {
            coco_sys::coco_problem_free(self.inner);
        }
    }
}
