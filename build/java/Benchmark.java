public class Benchmark {
	String problem_suite;
	String problem_suite_options;
	String observer;
	String observer_options;
	int len;
	int dimensions;
	int objectives;
	
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
			Problem problem = this.getProblemUnobserved(problem_index);
			problem.addObserver(self.observer, self.observer_options);
			return problem;
			
		} catch (NoSuchProblemException e) {
			problem.free();
			break;
			}
	}
}