import java.util.Random;

/**
 * An example of benchmarking random search on a COCO suite. 
 *
 * Set the parameter BUDGET_MULTIPLIER to suit your needs.
 */
public class ExampleExperiment {

	/**
	 * The maximal budget for evaluations done by an optimization algorithm equals 
	 * dimension * BUDGET_MULTIPLIER.
	 * Increase the budget multiplier value gradually to see how it affects the runtime.
	 */
	public static final int BUDGET_MULTIPLIER = 2;
	
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
    }

	/**
	 * Evaluate the static PROBLEM.
	 */
    public static final Function evaluateFunction = new Function() {
    	public double[] evaluate(double[] x) {
    		return PROBLEM.evaluateFunction(x);
    	}
    };

	/**
	 * The main method initializes the random number generator and calls the example experiment on the
	 * bi-objective suite.
	 */
	public static void main(String[] args) {
		
		Random randomGenerator = new Random(RANDOM_SEED);

		/* Change the log level to "warning" to get less output */
		CocoJNI.cocoSetLogLevel("info");

		System.out.println("Running the example experiment... (might take time, be patient)");
		System.out.flush();
		
		/* Call the example experiment */
		exampleExperiment("bbob-biobj", "bbob-biobj", randomGenerator);

		/* Uncomment the line below to run the same example experiment on the bbob suite
	  	exampleExperiment("bbob", "bbob", randomGenerator); */

		System.out.println("Done!");
		System.out.flush();

		return;
	}
	
	/**
	 * A simple example of benchmarking random search on a suite with instances from 2016.
	 *
	 * @param suiteName Name of the suite (use "bbob" for the single-objective and "bbob-biobj" for the
	 * bi-objective suite).
	 * @param observerName Name of the observer (use "bbob" for the single-objective and "bbob-biobj" for the
	 * bi-objective observer).
	 * @param randomGenerator The random number generator.
	 */
	public static void exampleExperiment(String suiteName, String observerName, Random randomGenerator) {
		try {

			/* Set some options for the observer. See documentation for other options. */
			final String observerOptions = 
					  "result_folder: RS_on_" + suiteName + " " 
					+ "algorithm_name: RS "
					+ "algorithm_info: \"A simple random search algorithm\"";

			/* Initialize the suite and observer */
			Suite suite = new Suite(suiteName, "year: 2016", "dimensions: 2,3,5,10,20,40");
			Observer observer = new Observer(observerName, observerOptions);
			Benchmark benchmark = new Benchmark(suite, observer);

			/* Initialize timing */
			Timing timing = new Timing();
			
			/* Iterate over all problems in the suite */
			while ((PROBLEM = benchmark.getNextProblem()) != null) {

				int dimension = PROBLEM.getDimension();

				/* Run the algorithm at least once */
				for (int run = 1; run <= 1 + INDEPENDENT_RESTARTS; run++) {

					long evaluationsDone = PROBLEM.getEvaluations();
					long evaluationsRemaining = (long) (dimension * BUDGET_MULTIPLIER) - evaluationsDone;

					/* Break the loop if the target was hit or there are no more remaining evaluations */
					if (PROBLEM.isFinalTargetHit() || (evaluationsRemaining <= 0))
						break;

					/* Call the optimization algorithm for the remaining number of evaluations */
					myRandomSearch(evaluateFunction,
							       dimension,
							       PROBLEM.getNumberOfObjectives(),
							       PROBLEM.getSmallestValuesOfInterest(),
							       PROBLEM.getLargestValuesOfInterest(),
							       evaluationsRemaining,
							       randomGenerator);

					/* Break the loop if the algorithm performed no evaluations or an unexpected thing happened */
					if (PROBLEM.getEvaluations() == evaluationsDone) {
						System.out.println("WARNING: Budget has not been exhausted (" + evaluationsDone + "/"
								+ dimension * BUDGET_MULTIPLIER + " evaluations done)!\n");
						break;
					} else if (PROBLEM.getEvaluations() < evaluationsDone)
						System.out.println("ERROR: Something unexpected happened - function evaluations were decreased!");
				}

				/* Keep track of time */
				timing.timeProblem(PROBLEM);
			}

			/* Output the timing data */
			timing.output();

			benchmark.finalizeBenchmark();

		} catch (Exception e) {
			System.err.println(e.toString());
		}
	}

	/** 
	 * A simple random search algorithm that can be used for single- as well as multi-objective 
	 * optimization.
	 */
	public static void myRandomSearch(Function f, 
			                          int dimension, 
			                          int numberOfObjectives, 
			                          double[] lowerBounds,
			                          double[] upperBounds, 
			                          long maxBudget, 
			                          Random randomGenerator) {

		double[] x = new double[dimension];
		double[] y = new double[numberOfObjectives];
		double range;
		
		for (int i = 0; i < maxBudget; i++) {
			
		    /* Construct x as a random point between the lower and upper bounds */
			for (int j = 0; j < dimension; j++) {
				range = upperBounds[j] - lowerBounds[j];
				x[j] = lowerBounds[j] + randomGenerator.nextDouble() * range;
			}

		    /* Call the evaluate function to evaluate x on the current problem (this is where all the COCO logging
		     * is performed) */
			y = f.evaluate(x);
		}
		
	}
}
