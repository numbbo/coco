import java.util.Random;

public class ExampleExperiment {

	/**
	 * The max budget for optimization algorithms should be set to dim * BUDGET.
	 * Increase budget for your experiments.
	 */
	public static final int BUDGET = 2;
	public static final long RANDOM_SEED = 123456;

	/** 
	 * A simple random search algorithm that can be used for single- as well as multi-objective 
	 * optimization.
	 */
	public static void myRandomSearch(Problem problem) {

		Random r = new Random(RANDOM_SEED);
		double[] x = new double[(int) problem.getDimension()];
		double[] y = new double[(int) problem.getNumberOfObjectives()];
		int dim = problem.getDimension();
		double range;
		
		long max_budget = dim * BUDGET;
		
		for (int i = 0; i < max_budget; i++) {
			for (int j = 0; j < problem.getDimension(); j++) {
				range = problem.getLargestValueOfInterest(j) - problem.getSmallestValueOfInterest(j);
				x[j] = problem.getSmallestValueOfInterest(j) + range * r.nextDouble();
			}
			
		    // Calls COCO's evaluate function where all the logging is performed 
			y = problem.evaluateFunction(x);
		}
		
	}
	
	/**
	 *  A simple example of how to use the COCO benchmarking on the suite bbob. 
	 */
	public static void exampleBBOB() {
		try {

			final String observer_options = 
					  "result_folder: RS_on_bbob " 
					+ "algorithm_name: RS "
					+ "algorithm_info: \"A simple random search algorithm\"";

			Suite suite = new Suite("bbob", "year: 2016", "dimensions: 2,3,5,10,20,40");
			Observer observer = new Observer("bbob", observer_options);
			Benchmark benchmark = new Benchmark(suite, observer);
			Problem problem;
	
			while ((problem = benchmark.getNextProblem()) != null) {	
				myRandomSearch(problem);	
			}
			
			benchmark.finalizeBenchmark();
			
		} catch (Exception e) {
			System.err.println(e.toString());
		}
	}
	
	/**
	 *  A simple example of how to use the COCO benchmarking on the suite bbob-biobj. 
	 */
	public static void exampleBBOBbiobj() {
		try {

			final String observer_options = 
					  "result_folder: RS_on_bbob-biobj " 
					+ "algorithm_name: RS "
					+ "algorithm_info: \"A simple random search algorithm\"";

			Suite suite = new Suite("bbob-biobj", "year: 2016", "dimensions: 2,3,5,10,20,40");
			Observer observer = new Observer("bbob-biobj", observer_options);
			Benchmark benchmark = new Benchmark(suite, observer);
			Problem problem;
	
			while ((problem = benchmark.getNextProblem()) != null) {	
				myRandomSearch(problem);	
			}
			
			benchmark.finalizeBenchmark();
			
		} catch (Exception e) {
			System.err.println(e.toString());
		}
	}

	/**
	 * The main method calls only the biobjective example experiment
	 */
	public static void main(String[] args) {

		System.out.println("Running the example experiment... (it takes time, be patient)");
		System.out.flush();

		/* Change the log level to "warning" to get less output */
		CocoJNI.cocoSetLogLevel("info");
		
		/* Call the example experiment */
		exampleBBOBbiobj();

		System.out.println("Done!");
		System.out.flush();

		return;
	}
}
