package javacoco;

public class JNIinterface {
	
	/* Load the library */
	static {
		System.loadLibrary("JNIinterface");
	}
	
	/* Class method */
	/* TODO: Handle NoSuchProblem exception */
	public static Problem next_problem(Benchmark benchmark) {
		Problem problem = new Problem(benchmark.problem_suit, benchmark.function_index, benchmark.observer, benchmark.options);
		return problem;
	}
	
	/* Native methods */
	public static native double[] coco_evaluate_function(Problem p, double[] x);
    public static native int coco_get_number_of_variables(Problem p);
    public static native int coco_get_number_of_objectives(Problem p);
    public static native double[] coco_get_smallest_values_of_interest(Problem p);
    public static native double[] coco_get_largest_values_of_interest(Problem p);
}
