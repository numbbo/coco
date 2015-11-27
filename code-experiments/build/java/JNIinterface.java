import java.lang.Long;

public class JNIinterface {
	
	/* Load the library */
	static {
		System.loadLibrary("JNIinterface");
	}
	
	/* Native methods */
	public static native long cocoSuiteGetProblem(String problem_suit, long function_index);
	public static native long cocoProblemAddObserverDeprecated(long problem, String observer, String options);
	public static native void cocoProblemFree(long p);
	public static native double[] cocoEvaluateFunction(Problem p, double[] x);
	public static native int cocoProblemGetDimension(long p);
	public static native int cocoProblemGetNumberOfObjectives(long p);
	public static native double[] cocoProblemGetSmallestValuesOfInterest(long p);
	public static native double[] cocoProblemGetLargestValuesOfInterest(long p);
	public static native boolean cocoProblemIsValid(long p);
	public static native String cocoProblemGetId(long p);
	public static native String cocoProblemGetName(long p);
	public static native int cocoProblemGetEvaluations(long p);
	public static native long cocoSuiteGetNextProblemIndex(String problem_suite, long problem_index, String select_options);
}
