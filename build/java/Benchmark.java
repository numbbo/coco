public class Benchmark {
	String problem_suite;
	String problem_suite_options;
	String observer;
	String observer_options;
	int len;
	int dimensions;
	int objectives;
	int current_problem_index;
	
	/* Constructor */
	public Benchmark(String problme_suite, String problem_suite_options, String observer, String observer_options) {
		super();
		this.problem_suite = problme_suite;
		this.problem_suite_options = problme_suite_options;
		this.observer = observer;
		this.observer_options = observer_options;
		this.len = 0;
		this.dimensions = 0;
		this.objectives = 0;
		this.current_function_index = -1;
	}
	
	public Problem getProblemUnobserved(int problem_index) {
		Problem problem = new Problem(this.problem_suite, problem_index);
		if (!validProblem(problem)){
			throw new NoSuchProblemException(this.problem_suite, problem_index);
		}
		return problem;
	}
	
	public Problem getProblem(int problem_index) {
		try {
			Problem problem = getProblemUnobserved(problem_index);
			problem.addObserver(self.observer, self.observer_options);
			return problem;
			
		} catch (NoSuchProblemException e) {
			problem.free();
			break;
			}
	}
	
	public int nextProblemIndex(int problem_index) {
		return JNIinterface.cocoNextProblemIndex(this.problem_suite, problem_index, this.problem_suite_options);
	}
	
	public Problem nextProblem() {
		this.current_problem_index = nextProblemIndex(this.current_problem_index);
		return getProblem(this.current_problem_index);
	}
}