package javacoco;

public class JNIinterface {
	
	/* Load the library */
	static {
		System.loadLibrary("JNIinterface");
	}
	
	/* Class method */
	/* TODO: Handle NoSuchProblem exception */
	public Problem next_problem(Benchmark benchmark) {
		Problem problem = new Problem(benchmark.problem_suit, benchmark.function_index, benchmark.observer, benchmark.options);
		return problem;
	}
	
	/* Native method */
	public static native double[] coco_evaluate_function(Problem p, double[] x);
}
