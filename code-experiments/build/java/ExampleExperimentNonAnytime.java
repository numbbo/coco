import java.util.Random;

/**
 * An example experiment for benchmarking non-anytime optimization algorithms
 * with restarts.
 * An example of benchmarking random search on a COCO suite. 
 *
 * Set the parameter BUDGET_MULTIPLIER to suit your needs.
 */

public class ExampleExperimentNonAnytime {
    /**
	 * The maximal number of independent restarts allowed for an algorithm that restarts itself. 
	 */
	public static final int INDEPENDENT_RESTARTS = 10000;
	
	/**
	 * The random seed. Change if needed.
	 */
	public static final long RANDOM_SEED = 0xdeadbeef;

	/**
	 * The problem to be optimized (needed in order to simplify the interface between the optimization
	 * algorithm and the COCO platform).
	 */
	public static Problem PROBLEM;
	
	/**
	 * Interface for function evaluation.
	 */
	public interface Function {
		double[] evaluate(double[] x);
		double[] evaluateConstraint(double[] x);
    }



    public static int[] defaultBudgetList(int maxBudget, int num){

    }
        
}