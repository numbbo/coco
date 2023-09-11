use coco_sys::{coco_observer_t, coco_problem_t, coco_random_state_t, coco_suite_t};
use std::{
    ffi::{CStr, CString},
    marker::PhantomData,
    ops::RangeInclusive,
    ptr,
};

pub enum LogLevel {
    Error,
    Warning,
    Info,
    Debug,
}
impl Default for LogLevel {
    fn default() -> Self {
        LogLevel::Info
    }
}
impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Error => "error",
            LogLevel::Warning => "warning",
            LogLevel::Info => "info",
            LogLevel::Debug => "debug",
        }
    }
}

/// Sets COCO's log level.
pub fn set_log_level(level: LogLevel) {
    let level = CString::new(level.as_str()).unwrap();
    unsafe {
        coco_sys::coco_set_log_level(level.as_ptr());
    }
}

/// A COCO suite
pub struct Suite {
    inner: *mut coco_suite_t,
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

pub enum SuiteName {
    Bbob,
    BbobBiobj,
    BbobBiobjExt,
    BbobLargescale,
    BbobConstrained,
    BbobMixint,
    BbobBiobjMixint,
    Toy,
}
impl SuiteName {
    pub fn as_str(&self) -> &'static str {
        match self {
            SuiteName::Bbob => "bbob",
            SuiteName::BbobBiobj => "bbob-biobj",
            SuiteName::BbobBiobjExt => "bbob-biobj-ext",
            SuiteName::BbobLargescale => "bbob-largescale",
            SuiteName::BbobConstrained => "bbob-constrained",
            SuiteName::BbobMixint => "bbob-mixint",
            SuiteName::BbobBiobjMixint => "bbob-biobj-mixint",
            SuiteName::Toy => "toy",
        }
    }
}

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
    pub fn new(name: SuiteName, instance: &str, options: &str) -> Option<Suite> {
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

    pub fn function_from_function_index(&self, function_idx: usize) -> usize {
        unsafe { coco_sys::coco_suite_get_function_from_function_index(self.inner, function_idx) }
    }

    pub fn dimension_from_dimension_index(&self, dimension_idx: usize) -> usize {
        unsafe {
            coco_sys::coco_suite_get_dimension_from_dimension_index(self.inner, dimension_idx)
        }
    }

    pub fn instance_from_instance_index(&self, instance_idx: usize) -> usize {
        unsafe { coco_sys::coco_suite_get_instance_from_instance_index(self.inner, instance_idx) }
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

        let mut function = 0;
        let mut dimension = 0;
        let mut instance = 0;

        unsafe {
            let suite_index = coco_sys::coco_problem_get_suite_dep_index(inner);

            coco_sys::coco_suite_decode_problem_index(
                self.inner,
                suite_index,
                &mut function,
                &mut dimension,
                &mut instance,
            );
        }

        Some(Problem {
            inner,
            function_idx: function as usize,
            dimension_idx: dimension as usize,
            instance_idx: instance as usize,
            _phantom: PhantomData,
        })
    }

    pub fn problem(&mut self, problem_idx: usize) -> Option<Problem> {
        let inner = unsafe { coco_sys::coco_suite_get_problem(self.inner, problem_idx) };

        if inner.is_null() {
            return None;
        }

        let mut function = 0;
        let mut dimension = 0;
        let mut instance = 0;

        unsafe {
            let suite_index = coco_sys::coco_problem_get_suite_dep_index(inner);

            coco_sys::coco_suite_decode_problem_index(
                self.inner,
                suite_index,
                &mut function,
                &mut dimension,
                &mut instance,
            );
        }

        Some(Problem {
            inner,
            function_idx: function as usize,
            dimension_idx: dimension as usize,
            instance_idx: instance as usize,
            _phantom: PhantomData,
        })
    }

    /// Returns the problem for the given function, dimension and instance.
    ///
    /// **Important:** This is different from [[Suite::problem_by_function_dimension_instance_index]].
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
        let problem = unsafe {
            coco_sys::coco_suite_get_problem_by_function_dimension_instance(
                self.inner,
                function as usize,
                dimension as usize,
                instance as usize,
            )
        };

        if problem.is_null() {
            return None;
        }

        Some(Problem {
            inner: problem,
            function_idx: function,
            dimension_idx: dimension,
            instance_idx: instance,
            _phantom: PhantomData,
        })
    }

    /// Returns the problem for the given function, dimension and instance index.
    pub fn problem_by_function_dimension_instance_index(
        &mut self,
        function_idx: usize,
        dimension_idx: usize,
        instance_idx: usize,
    ) -> Option<Problem> {
        let problem_index = unsafe {
            coco_sys::coco_suite_encode_problem_index(
                self.inner,
                function_idx.try_into().unwrap(),
                dimension_idx.try_into().unwrap(),
                instance_idx.try_into().unwrap(),
            )
        };

        let problem = unsafe { coco_sys::coco_suite_get_problem(self.inner, problem_index) };

        if problem.is_null() {
            return None;
        }

        Some(Problem {
            inner: problem,
            function_idx,
            dimension_idx,
            instance_idx,
            _phantom: PhantomData,
        })
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

/// A specific problem instance.
///
/// Instances can be optained using [Suite::next_problem]
/// and [Suite::problem_by_function_dimension_instance].
pub struct Problem<'suite> {
    inner: *mut coco_problem_t,
    function_idx: usize,
    instance_idx: usize,
    dimension_idx: usize,
    _phantom: PhantomData<&'suite Suite>,
}

unsafe impl Send for Problem<'_> {}

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

    /// Returns the index of the problem.
    pub fn function_index(&self) -> usize {
        self.function_idx
    }

    /// Returns the dimension index of the problem.
    pub fn dimension_index(&self) -> usize {
        self.dimension_idx
    }

    /// Returns the instance of the problem.
    pub fn instance_index(&self) -> usize {
        self.instance_idx
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
    pub fn get_ranges_of_interest(&self) -> Vec<RangeInclusive<f64>> {
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

/// An observer to log results in COCO's data format.
///
/// Can be provided to [Suite::next_problem] and it will
/// automatically be attached to the returned problem.
pub struct Observer {
    inner: *mut coco_observer_t,
}

pub enum ObserverName {
    Bbob,
    BbobBiobj,
    Toy,
    None,
}
impl ObserverName {
    fn as_str(&self) -> &'static str {
        match self {
            ObserverName::Bbob => "bbob",
            ObserverName::BbobBiobj => "bbob-biobj",
            ObserverName::Toy => "toy",
            ObserverName::None => "no-observer",
        }
    }
}

impl Observer {
    /// Creates a new observer.
    ///
    /// # observer_options
    /// A string of pairs "key: value" used to pass the options to the observer. Some
    /// observer options are general, while others are specific to some observers. Here we list only the general
    /// options, see observer_bbob, observer_biobj and observer_toy for options of the specific observers.
    /// - "result_folder: NAME" determines the folder within the "exdata" folder into which the results will be
    /// output. If the folder with the given name already exists, first NAME_001 will be tried, then NAME_002 and
    /// so on. The default value is "default".
    /// - "algorithm_name: NAME", where NAME is a short name of the algorithm that will be used in plots (no
    /// spaces are allowed). The default value is "ALG".
    /// - "algorithm_info: STRING" stores the description of the algorithm. If it contains spaces, it must be
    /// surrounded by double quotes. The default value is "" (no description).
    /// - "number_target_triggers: VALUE" defines the number of targets between each 10^i and 10^(i+1)
    /// (equally spaced in the logarithmic scale) that trigger logging. The default value is 100.
    /// - "target_precision: VALUE" defines the precision used for targets (there are no targets for
    /// abs(values) < target_precision). The default value is 1e-8.
    /// - "number_evaluation_triggers: VALUE" defines the number of evaluations to be logged between each 10^i
    /// and 10^(i+1). The default value is 20.
    /// - "base_evaluation_triggers: VALUES" defines the base evaluations used to produce an additional
    /// evaluation-based logging. The numbers of evaluations that trigger logging are every
    /// base_evaluation * dimension * (10^i). For example, if base_evaluation_triggers = "1,2,5", the logger will
    /// be triggered by evaluations dim*1, dim*2, dim*5, 10*dim*1, 10*dim*2, 10*dim*5, 100*dim*1, 100*dim*2,
    /// 100*dim*5, ... The default value is "1,2,5".
    /// - "precision_x: VALUE" defines the precision used when outputting variables and corresponds to the number
    /// of digits to be printed after the decimal point. The default value is 8.
    /// - "precision_f: VALUE" defines the precision used when outputting f values and corresponds to the number of
    /// digits to be printed after the decimal point. The default value is 15.
    /// - "precision_g: VALUE" defines the precision used when outputting constraints and corresponds to the number
    /// of digits to be printed after the decimal point. The default value is 3.
    /// - "log_discrete_as_int: VALUE" determines whether the values of integer variables (in mixed-integer problems)
    /// are logged as integers (1) or not (0 - in this case they are logged as doubles). The default value is 0.
    pub fn new(name: ObserverName, options: &str) -> Option<Observer> {
        let name = CString::new(name.as_str()).unwrap();
        let options = CString::new(options).unwrap();

        let inner = unsafe { coco_sys::coco_observer(name.as_ptr(), options.as_ptr()) };

        if inner.is_null() {
            None
        } else {
            Some(Observer { inner })
        }
    }

    /// Prints where the result is written to.
    pub fn result_folder(&self) -> &str {
        unsafe {
            CStr::from_ptr(coco_sys::coco_observer_get_result_folder(self.inner))
                .to_str()
                .unwrap()
        }
    }
}

impl Drop for Observer {
    fn drop(&mut self) {
        unsafe {
            coco_sys::coco_observer_free(self.inner);
        }
    }
}

/// COCO specific random number generator.
pub struct RandomState {
    inner: *mut coco_random_state_t,
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
