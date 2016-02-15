/**
 * This class contains the declaration of all the CocoJNI functions. 
 */
public class CocoJNI {

	/* Load the library */
	static {
		System.loadLibrary("CocoJNI");
	}

	/* Native methods */
	public static native void cocoSetLogLevel(String logLevel);
	
	// Observer
	public static native long cocoGetObserver(String observerName, String observerOptions);
	public static native void cocoFinalizeObserver(long ObserverPointer);

	// Suite
	public static native long cocoGetSuite(String suiteName, String suiteInstance, String suiteOptions);
	public static native void cocoFinalizeSuite(long SuitePointer);

	// Problem
	public static native long cocoGetNextProblem(long suitePointer, long ObserverPointer);

	// Functions
	public static native double[] cocoEvaluateFunction(long problemPointer, double[] x);
	public static native double[] cocoEvaluateConstraint(long problemPointer, double[] x);

	// Getters
	public static native int cocoProblemGetDimension(long problemPointer);
	public static native int cocoProblemGetNumberOfObjectives(long problemPointer);
	public static native int cocoProblemGetNumberOfConstraints(long problemPointer);

	public static native double[] cocoProblemGetSmallestValuesOfInterest(long problemPointer);
	public static native double[] cocoProblemGetLargestValuesOfInterest(long problemPointer);

	public static native String cocoProblemGetId(long problemPointer);
	public static native String cocoProblemGetName(long problemPointer);
	
	public static native long cocoProblemGetEvaluations(long problemPointer);	
	public static native long cocoProblemGetIndex(long problemPointer); 
	
	public static native int cocoProblemIsFinalTargetHit(long problemPointer);
}
