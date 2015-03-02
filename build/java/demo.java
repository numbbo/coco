import java.util.Random;

public class demo {
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
	double MAXEVALS = 10; 
        Benchmark my_benchmark = new Benchmark("bbob2009", "bbob2009_observer", "random_search");
        while (true) {
	    try {
		Problem problem = JNIinterface.nextProblem(my_benchmark);
		// TODO (to be discussed): how about LowerBounds and UpperBounds for a name? 
		my_optimizer(problem, JNIinterface.cocoGetSmallestValuesOfInterest(problem),
			     JNIinterface.cocoGetLargestValuesOfInterest(problem), MAXEVALS);
		System.out.println("done with problem " + problem + " ...");
		// TODO: verify that we don't need to call problem.free();
		// (free is called even in the Python demo)
	    } catch (NoSuchProblemException e) {
		System.out.println("done.");
		break;
		// System.out.println(e);
            } 
        }
    } 
}
