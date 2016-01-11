import java.util.Random;

public class ExampleExperiment {
	
	public static final int MAX_BUDGET = 100;
	public static final long RANDOM_SEED = 123456;

	/** 
	 * A simple random search algorithm that can be used for single- as well as multi-objective 
	 * optimization.
	 */
	public static void myRandomSearch(Problem problem) {

		Random r = new Random(RANDOM_SEED);
		double[] x = new double[(int) problem.getDimension()];
		double[] y = new double[(int) problem.getNumberOfObjectives()];
		double range;
		
		for (int i = 0; i < MAX_BUDGET; i++) {
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
					  "result_folder: GS_on_bbob " 
					+ "algorithm_name: GS "
					+ "algorithm_info: \"A simple grid search algorithm\" ";

			Suite suite = new Suite("bbob", "year: 2009", "dimensions: 2,3,5,10,20 instance_idx: 1");
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
					  "result_folder: GS_on_bbob-biobj " 
					+ "algorithm_name: GS "
					+ "algorithm_info: \"A simple grid search algorithm\" "
					+ "log_decision_variables: low_dim "
					+ "compute_indicators: 1 "
					+ "log_nondominated: all";

			Suite suite = new Suite("bbob-biobj", "", "dimensions: 2,3,5,10,20 instance_idx: 1-5");
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

	public static void main(String[] args) {

		System.out.println("Running the experiments... (it takes time, be patient)");
		System.out.flush();
		
		exampleBBOB();

		System.out.println("First example done!");
		System.out.flush();
		
		exampleBBOBbiobj();

		System.out.println("Second example done!");
		System.out.flush();

		return;
	}
}
