import java.util.Random;

/* Draft: determine native methods to declare */

public class demo {
    public static void my_optimizer(Problem problem, double[] lower_bounds, double[] upper_bounds, double budget) {
        System.out.println("In optimizer...");
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
        // TODO Auto-generated method stub
        Benchmark my_benchmark = new Benchmark("bbob2009", "bbob2009_observer", "random_search"); // parameters to be defined
        while (true) {
            Problem problem = JNIinterface.nextProblem(my_benchmark);
            System.out.println("Optimizing " + problem.toString());
            my_optimizer(problem, JNIinterface.cocoGetSmallestValuesOfInterest(problem), JNIinterface.cocoGetLargestValuesOfInterest(problem), 10000);
        }
        
        
    }
    
}