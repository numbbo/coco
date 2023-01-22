import java.util.Arrays;
import java.util.Random;


/**
 * An example experiment for benchmarking non-anytime optimization algorithms
 * with restarts.

 */

public class ExampleExperimentNonAnytime {

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

    /**
     * Evaluate the static PROBLEM.
     */
    public static final ExampleExperiment.Function evaluateFunction = new ExampleExperiment.Function() {
        public double[] evaluate(double[] x) {
            return PROBLEM.evaluateFunction(x);
        }
        public double[] evaluateConstraint(double[] x) {
            return PROBLEM.evaluateConstraint(x);
        }
    };

    /**
     * The main method initializes the random number generator and calls the non-anytime example experiment on the
     * bi-objective suite.
     */
    public static void main(String[] args) {

        Random randomGenerator = new Random(RANDOM_SEED);

        /* Change the log level to "warning" to get less output */
        CocoJNI.cocoSetLogLevel("info");

        System.out.println("Running the example non-anytime experiment... (might take time, be patient)");
        System.out.flush();

        /* Start the actual experiments on a test suite and use a matching logger, for
         * example one of the following:
         *
         *   bbob                 24 unconstrained noiseless single-objective functions
         *   bbob-biobj           55 unconstrained noiseless bi-objective functions
         *   [bbob-biobj-ext       92 unconstrained noiseless bi-objective functions]
         *   bbob-largescale      24 unconstrained noiseless single-objective functions in large dimension
         *   [bbob-constrained*   48 constrained noiseless single-objective functions]
         *   bbob-mixint          24 unconstrained noiseless single-objective functions with mixed-integer variables
         *   bbob-biobj-mixint    92 unconstrained noiseless bi-objective functions with mixed-integer variables
         *
         * Suites with a star are partly implemented but not yet fully supported.
         *
         * Adapt to your need. Note that the experiment is run according
         * to the settings, defined in exampleExperiment(...) below.
         */
        exampleExperimentNonAnytime("bbob", "bbob", randomGenerator);

        System.out.println("Done!");
        System.out.flush();
    }

    /**
     * An example experiment for benchmarking non-anytime optimization algorithms
     * with restarts.
     *
     * @param suiteName Name of the suite (e.g. "bbob", "bbob-biobj", or "bbob-constrained").
     * @param observerName Name of the observer matching with the chosen suite (e.g. "bbob-biobj"
     * when using the "bbob-biobj-ext" suite).
     * @param randomGenerator The random number generator.
     */
    public static void exampleExperimentNonAnytime(String suiteName, String observerName, Random randomGenerator) {
        try {
            final String algorithm_name = "RS";  // no spaces allowed
            final String output_folder = algorithm_name;  // no spaces allowed

            /* Initialize the suite.
             * For more details on how to change the default options, see
             * http://numbbo.github.io/coco-doc/C/#suite-parameters. */
            Suite suite = new Suite(suiteName, "", "");

            // Set some options for the observer. See documentation for other options.
            final String observerOptions = "result_folder: " + output_folder + " algorithm_name: " + algorithm_name;
            /* Initialize the observer.
             * For more details on how to change the default options, see
             * http://numbbo.github.io/coco-doc/C/#observer-parameters. */
            Observer observer = new Observer(observerName, observerOptions);

            // A list of increasing budgets to be multiplied by dimension
            // gradually increase `max_budget` to 10, 100, ...
            // or replace with a user-defined list
            int[] budgetMultiplierList = defaultBudgetList(5, 50);
            System.out.println("Benchmarking with budgets: "+Arrays.toString(budgetMultiplierList)+" (* dimension)");

            Benchmark benchmark = new Benchmark(suite, observer);

            // Iterate over all problems in the suite
            while ((PROBLEM = benchmark.getNextProblem()) != null) {

                int dimension = PROBLEM.getDimension();

                for(int b : budgetMultiplierList){

                    long evaluationsDone = PROBLEM.getEvaluations() + PROBLEM.getEvaluationsConstraints();
                    long evaluationsRemaining = (long) dimension * b - evaluationsDone;

                    /* Break the loop if the target was hit or there are no more remaining evaluations */
                    if (PROBLEM.isFinalTargetHit() || (evaluationsRemaining <= 0))
                        break;

                    /* Call the optimization algorithm for the remaining number of evaluations */
                    MyRandomSearch(evaluateFunction,
                            dimension,
                            PROBLEM.getNumberOfObjectives(),
                            PROBLEM.getNumberOfConstraints(),
                            PROBLEM.getSmallestValuesOfInterest(),
                            PROBLEM.getLargestValuesOfInterest(),
                            evaluationsRemaining,
                            randomGenerator);

                    /* Break the loop if the algorithm performed no evaluations or an unexpected thing happened */
                    if (PROBLEM.getEvaluations() == evaluationsDone) {
                        System.out.println("WARNING: Budget has not been exhausted (" + evaluationsDone + "/"
                                + dimension * b + " evaluations done)!\n");
                        break;
                    } else if (PROBLEM.getEvaluations() < evaluationsDone)
                        System.out.println("ERROR: Something unexpected happened - function evaluations were decreased!");
                }
            }
            benchmark.finalizeBenchmark();
        } catch (Exception e) {
            System.err.println(e);
        }
    }

    /**
     * A simple random search algorithm that can be used for single- as well as multi-objective
     * optimization.
     */
    public static void MyRandomSearch(ExampleExperiment.Function f,
                                          int dimension,
                                          int numberOfObjectives,
                                          int numberOfConstraints,
                                          double[] lowerBounds,
                                          double[] upperBounds,
                                          long budget,
                                          Random randomGenerator) {

        double[] x = new double[dimension];
        double[] y = new double[numberOfObjectives];
        double[] z = new double[numberOfConstraints];
        double range;

        for (int i = 0; i < budget; i++) {

            /* Construct x as a random point between the lower and upper bounds */
            for (int j = 0; j < dimension; j++) {
                range = upperBounds[j] - lowerBounds[j];
                x[j] = lowerBounds[j] + randomGenerator.nextDouble() * range;
            }

            /* Call the evaluate function to evaluate x on the current problem (this is where all the COCO logging
             * is performed) */
            if (numberOfConstraints > 0)
                z = f.evaluateConstraint(x);
            y = f.evaluate(x);
        }
    }

    /**
     * Produces a budget list with at most `num` different increasing budgets
     * within [1, `max_budget`] that are equally spaced in the logarithmic space.
     * @param maxBudget
     * @param num
     * @return
     */
    public static int[] defaultBudgetList(int maxBudget, int num){
        double[] values = logspace(Math.log10(maxBudget), num);
        Integer[] array = new Integer[num];
        for(int i = 0; i < num; i++){
            array[i] = (int) values[i];
        }
        return Arrays.stream(array).distinct().mapToInt(i -> i).toArray();
    }

    /**
     * Generates n logarithmically-spaced points between d1 and d2 using base 10.
     * @param max The max value
     * @param num The number of points to generated
     * @return an array of linearly space points.
     */
    public static double[] logspace(double max, int num) {
        double[] y = new double[num];
        double[] p = linspace(max, num);
        for(int i = 0; i < y.length - 1; i++) {
            y[i] = Math.pow(10, p[i]);
        }
        y[y.length - 1] = Math.pow(10, max);
        return y;
    }

    /**
     * Generates n linearly-spaced points between d1 and d2.
     * @param max The max value
     * @param num The number of points to generated
     * @return an array of linearly space points.
     */
    public static double[] linspace(double max, int num) {
        double[] y = new double[num];
        double dy = max / (num - 1);
        for(int i = 0; i < num; i++) {
            y[i] = (dy * i);
        }
        return y;
    }
}
