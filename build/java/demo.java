import java.util.ArrayList;
import java.util.Random;
import coco.Benchmark;

/* Draft: determine native methods to declare */

public class demo {
    public static void my_optimizer(Problem problem, double[] lower_bounds, double[] upper_bounds, double budget) {
        System.out.println("In optimizer...");
        int n = lower_bounds.length;
        int i, j;
        double[] x = new double[n];
        double y;
        for (i = 0; i < budget; i++){
            Random r = new Random();
            for (j = 0; j < n; j++){
                x[j] = lower_bounds[j] + (upper_bounds[j] - lower_bounds[j]) * r.nextDouble();
            }
            y = problem.coco_evaluate_function(x);
        }
        
        
        
        
    }
    
    public static void main(String[] args) {
        // TODO Auto-generated method stub
        Benchmark my_benchmark = new Benchmark("bbob2009", "bbob_observer", "random_search"); // parameters to be defined
        while(true){
            problem = my_benchmark.nextProblem();
            System.out.println("Optimizing " + problem.toString());
            my_optimizer(problem, problem.lowerBounds(), problem.upperBounds(), 10000);
        }
        
        
    }
    
}