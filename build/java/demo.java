
public class demo {
	
	public static void MY_OPTIMIZER(JNIproblem problem, ArrayList<double> lower_bounds, ArrayList<double> upper_bounds, double budget) {
		int n = 
		
	}
	
	public static void MY_OPTIMIZER(JNIfgeneric fgeneric, int dim, double maxfunevals, Random rand) {

        double[] x = new double[dim];

        /* Obtain the target function value, which only use is termination */
        double ftarget = fgeneric.getFtarget();
        double f;

        if (maxfunevals > 1e9 * dim) {
             maxfunevals = 1e9 * dim;
        }

        for (double iter = 0.; iter < maxfunevals; iter++) {
            /* Generate individual */
            for (int i = 0; i < dim; i++) {
                x[i] = 10. * rand.nextDouble() - 5.;
            }

            /* evaluate X on the objective function */
            f = fgeneric.evaluate(x);

            if (f < ftarget) {
                break;
            }
        }
    }
	
	
	
	
	
	
	

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
