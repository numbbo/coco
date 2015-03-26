import java.lang.Long;

public class JNIinterface {
	
	/* Load the library */
	static {
		System.loadLibrary("JNIinterface");
	}
	
	/* Native methods */
	public static native long cocoGetProblem(String problem_suit, long function_index);
	public static native long cocoObserveProblem(String observer, long problem, String options);
	public static native void cocoFreeProblem(long p);
	public static native double[] cocoEvaluateFunction(Problem p, double[] x);
    public static native int cocoGetNumberOfVariables(long p);
    public static native int cocoGetNumberOfObjectives(long p);
    public static native double[] cocoGetSmallestValuesOfInterest(long p);
    public static native double[] cocoGetLargestValuesOfInterest(long p);
    public static native boolean validProblem(long p);
    public static native String cocoGetProblemId(long p);
    public static native String cocoGetProblemName(long p);
    public static native int cocoGetEvaluations(long p);
    public static native long cocoNextProblemIndex(String problem_suite, long problem_index, String select_options);
}
