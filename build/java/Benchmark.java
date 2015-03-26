public class Benchmark {
	String problem_suite;
	String problem_suite_options;
	String observer;
	String observer_options;
	int len;
	int dimensions;
	int objectives;
	long current_problem_index;
	
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
	
	public Problem getProblemUnobserved(long problem_index) throws NoSuchProblemException {
		Problem problem = null;
		try {
			problem = new Problem(this.problem_suite, problem_index);
		} catch (NoSuchProblemException e) {
			System.out.println("Benchmark.getProblemUnobserved: " + e);
			throw e;
		}
		return problem;
	}
	
	public Problem getProblem(long problem_index) throws NoSuchProblemException {
		Problem problem = null;
		try {
			problem = getProblemUnobserved(problem_index);
			problem.addObserver(this.observer, this.observer_options);
		} catch (NoSuchProblemException e) {
			System.out.println("Benchmark.getProblem: " + e);
			throw e;
		}
		return problem;
	}
	
	public long nextProblemIndex(long problem_index) {
		return JNIinterface.cocoNextProblemIndex(this.problem_suite, problem_index, this.problem_suite_options);
	}
	
	public Problem nextProblem() throws NoSuchProblemException {
		Problem problem = null;
		try {
			this.current_problem_index = nextProblemIndex(this.current_problem_index);
			problem = getProblem(this.current_problem_index);
		} catch (NoSuchProblemException e) {
			System.out.println("Benchmark.nextProblem: " + e);
			throw e;
		}
		return problem;
	}
}