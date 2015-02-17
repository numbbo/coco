import java.util.Random;

public class demo {
    public static void my_optimizer(Problem problem, double[] lower_bounds, double[] upper_bounds, double budget) {
        int n = lower_bounds.length;
        int i, j;
        double[] x = new double[n];
        double[] y;
        for (i = 0; i < budget; i++){
            Random r = new Random();
            for (j = 0; j < n; j++){
                x[j] = lower_bounds[j] + (upper_bounds[j] - lower_bounds[j]) * r.nextDouble();
            }
            y = JNIinterface.cocoEvaluateFunction(problem, x);
        }
        
        
        
        
    }
    
    public static void main(String[] args) {
        Benchmark my_benchmark = new Benchmark("bbob2009", "bbob2009_observer", "random_search");
        while (true) {
        	try {
        		Problem problem = JNIinterface.nextProblem(my_benchmark);
        		System.out.println("Optimizing " + problem);
        		my_optimizer(problem, JNIinterface.cocoGetSmallestValuesOfInterest(problem), JNIinterface.cocoGetLargestValuesOfInterest(problem), 1000);
        	} catch (NoSuchProblemException e) {
        		System.out.println(e);
        	} 
        }
    } 
}
