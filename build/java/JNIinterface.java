import java.lang.Long;

public class JNIinterface {
	
	/* Load the library */
	static {
		System.loadLibrary("JNIinterface");
	}
	
	public static Problem nextProblem(Benchmark benchmark) throws NoSuchProblemException {
		if (!validProblem(benchmark.problem_suit, benchmark.function_index)) {
			throw new NoSuchProblemException(benchmark.problem_suit, benchmark.function_index);
		}
        Problem problem = new Problem(benchmark.problem_suit, benchmark.function_index);
        benchmark.function_index ++;
        return problem;
		
		
	}
	
	/* Native methods */
	public static native long cocoGetProblem(String problem_suit, int function_index);
	public static native long cocoObserveProblem(String observer, long problem, String options);
	public static native void cocoFreeProblem(long p);
	public static native double[] cocoEvaluateFunction(Problem p, double[] x);
    public static native int cocoGetNumberOfVariables(long p);
    public static native int cocoGetNumberOfObjectives(long p);
    public static native double[] cocoGetSmallestValuesOfInterest(long p);
    public static native double[] cocoGetLargestValuesOfInterest(long p);
    public static native boolean validProblem(String suit, int function_index);
    public static native String cocoGetProblemId(long p);
    public static native String cocoGetProblemName(long p);
    public static native int cocoGetEvaluations(long p);
}
