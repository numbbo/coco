import java.util.Random;
import java.lang.reflect.Method;

public class demo {
	public static final int MAXEVALS = 100;
	public static int number_of_batches = 1; // const or not?
	public static int current_batch = 1; // const or not?
	
    public static void my_optimizer(Problem problem,
				    double[] lower_bounds, double[] upper_bounds, double budget) {
        int n = lower_bounds.length;
        int i, j;
        double[] x = new double[n];
        double[] y;
        for (i = 0; i < budget; i++) {
            Random r = new Random();
            for (j = 0; j < n; j++) {
                x[j] = lower_bounds[j] + (upper_bounds[j] - lower_bounds[j]) * r.nextDouble();
            }
            y = JNIinterface.cocoEvaluateFunction(problem, x);
        }
    }
    
    public static void main(String[] args) {
    	System.out.println("'Generic usecase with batches...'");
        Benchmark my_benchmark = new Benchmark("suite_bbob2009", "", "observer_bbob2009", "random_search_on_bbob2009");
        long problem_index = -1;
        long found_problems = 0;
        long addressed_problems = 0;
        while (true) {
        	problem_index = my_benchmark.getNextProblemIndex(problem_index);
        	if (problem_index < 0)
        		break;
        	found_problems++;
        	if ((problem_index + current_batch - 1) % number_of_batches != 0)
                continue;
        	try {
        		Problem problem = my_benchmark.getProblem(problem_index);
            	my_optimizer(problem, problem.lower_bounds, problem.upper_bounds, MAXEVALS);
            	addressed_problems++;
            	// System.out.println("done with problem " + problem + " ...");
            	problem.free();
        	} catch (NoSuchProblemException e) {
    			System.out.println("demo.main: " + e);
    			break;
    		}
        	
        }
    } 
}
