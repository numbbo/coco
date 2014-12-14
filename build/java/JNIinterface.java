public class JNIinterface {
	
	/* Load the library */
	static {
		System.loadLibrary("JNIinterface");
	}
	
	/* Class method */
	/* TODO: Handle NoSuchProblem exception */
	public static Problem nextProblem(Benchmark benchmark) {
		Problem problem = new Problem(benchmark.problem_suit, benchmark.function_index, benchmark.observer, benchmark.options);
        benchmark.function_index ++;
		return problem;
	}
	
	/* Native methods */
	public static native double[] cocoEvaluateFunction(Problem p, double[] x);
    public static native int cocoGetNumberOfVariables(Problem p);
    public static native int cocoGetNumberOfObjectives(Problem p);
    public static native double[] cocoGetSmallestValuesOfInterest(Problem p);
    public static native double[] cocoGetLargestValuesOfInterest(Problem p);
}
