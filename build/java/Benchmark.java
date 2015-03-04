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
	public Benchmark(String problem_suite, String problem_suite_options, String observer, String observer_options) {
		super();
		this.problem_suite = problem_suite;
		this.problem_suite_options = problem_suite_options;
		this.observer = observer;
		this.observer_options = observer_options;
		this.len = 0;
		this.dimensions = 0;
		this.objectives = 0;
		this.current_problem_index = -1;
	}
	
	public Problem getProblemUnobserved(int problem_index) {
		Problem problem = null;
		try {
			problem = new Problem(this.problem_suite, problem_index);
		} catch (NoSuchProblemException e) {
			System.out.println(e);
		}
		return problem;
	}
	
	public Problem getProblem(int problem_index) {
		Problem problem = getProblemUnobserved(problem_index);
		problem.addObserver(this.observer, this.observer_options);
		return problem;
	}
	
	public int nextProblemIndex(int problem_index) {
		return JNIinterface.cocoNextProblemIndex(this.problem_suite, problem_index, this.problem_suite_options);
	}
	
	public Problem nextProblem() {
		this.current_problem_index = nextProblemIndex(this.current_problem_index);
		return getProblem(this.current_problem_index);
	}
}