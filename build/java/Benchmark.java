public class Benchmark {
	String problem_suit;
	String problem_suite_options;
	String observer;
	String observer_options;
	int len;
	int dimensions;
	int objectives;
	
	/* Constructor */
	public Benchmark(String problme_suit, String observer, String options) {
		super();
		this.problem_suit = problme_suit;
		this.observer = observer;
		this.options = options;
		this.function_index = 0;
	}
}