/**
 * An example of benchmarking random search on a COCO suite. A grid search optimizer is also
 * implemented and can be used instead of random search.
 *
 * Set the global parameter BUDGET_MULTIPLIER to suit your needs.
 */
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#include "coco.h"

#define max(a,b) ((a) > (b) ? (a) : (b))

/**
 * The maximal budget for evaluations done by an optimization algorithm equals dimension * BUDGET_MULTIPLIER.
 * Increase the budget multiplier value gradually to see how it affects the runtime.
 */
static const unsigned int BUDGET_MULTIPLIER = 2;

/**
 * The maximal number of independent restarts allowed for an algorithm that restarts itself.
 */
static const long INDEPENDENT_RESTARTS = 1e5;

/**
 * The random seed. Change if needed.
 */
static const uint32_t RANDOM_SEED = 0xdeadbeef;

/**
 * A function type for evaluation functions, where the first argument is the vector to be evaluated and the
 * second argument the vector to which the evaluation result is stored.
 */
typedef void (*evaluate_function_t)(const double *x, double *y);

/**
 * A pointer to the problem to be optimized (needed in order to simplify the interface between the optimization
 * algorithm and the COCO platform).
 */
static coco_problem_t *PROBLEM;

/**
 * Calls coco_evaluate_function() to evaluate the objective function
 * of the problem at the point x and stores the result in the vector y
 */
static void evaluate_function(const double *x, double *y) {
  coco_evaluate_function(PROBLEM, x, y);
}

/**
 * Calls coco_evaluate_constraint() to evaluate the constraints
 * of the problem at the point x and stores the result in the vector y
 */
static void evaluate_constraint(const double *x, double *y) {
  coco_evaluate_constraint(PROBLEM, x, y);
}

/* Declarations of all functions implemented in this file (so that their order is not important): */
void example_experiment(const char *suite_name,
                        const char *suite_options,
                        const char *observer_name,
                        const char *observer_options,
                        coco_random_state_t *random_generator);

void my_random_search(evaluate_function_t evaluate_func,
                      evaluate_function_t evaluate_cons,
                      const size_t dimension,
                      const size_t number_of_objectives,
                      const size_t number_of_constraints,
                      const double *lower_bounds,
                      const double *upper_bounds,
                      const size_t number_of_integer_variables,
                      const size_t max_budget,
                      coco_random_state_t *random_generator);

void my_grid_search(evaluate_function_t evaluate_func,
                    evaluate_function_t evaluate_cons,
                    const size_t dimension,
                    const size_t number_of_objectives,
                    const size_t number_of_constraints,
                    const double *lower_bounds,
                    const double *upper_bounds,
                    const size_t number_of_integer_variables,
                    const size_t max_budget);

/* Structure and functions needed for timing the experiment */
typedef struct {
	size_t number_of_dimensions;
	size_t current_idx;
	char **output;
	size_t previous_dimension;
	size_t cumulative_evaluations;
	time_t start_time;
	time_t overall_start_time;
} timing_data_t;
static timing_data_t *timing_data_initialize(coco_suite_t *suite);
static void timing_data_time_problem(timing_data_t *timing_data, coco_problem_t *problem);
static void timing_data_finalize(timing_data_t *timing_data);

/**
 * The main method initializes the random number generator and calls the example experiment on the
 * bbob suite.
 */
int main(void) {
  coco_suite_t *suite;
  coco_problem_t *problem;
  double *f;
  double x[] = {
      -0.029913378181516796, 0.026315915378287816, -0.03428860660160116, 0.04345136845014721, 0.08266691523849175,
      -0.08962646735167284, -0.009321165686034994, 0.01376221367554433, -0.07696021668350436, -0.02576069977277881,
      -0.008619356499880233, -0.05022996917984645, 0.05139343434578984, 0.02265525608745122, -0.07709334144806637,
      0.02631587624277361, 0.02896359868205588, 0.02270174967489067, -0.0982642282988627, 0.003181952166495803,
      0.039266935982427024, -0.015912129807630123, 0.022536724745090882, 0.0841018251575661, -0.06448525371613681,
      0.12405784888895496, 0.04161352929272865, 0.008002379138155984, 0.01983445442663518, 0.024150442954941127,
      -0.05924991555997742, -0.004413179349070246, 0.023392112413846502, 0.09181616489222123, -0.014479570604552329,
      -0.03220786474969741, 0.06509181196853106, 0.05458642804822669, -0.020851850545063628, 0.022237677323086874,
      0.01931746021718089, -0.021898551002960544, -0.09497270903366575, -0.02614910658067987, 0.022661466347187794,
      -0.05197884166642458, -0.031721735875164817, 0.05151130041770149, 0.013100894551186558, -0.09095886314794437,
      -0.06504583722489192, -0.057180502405976365, 0.044164508832802804, -0.04477531652130835, -0.03653480478907052,
      0.016016669883098418, 0.042599898445434534, 0.048777985798221686, -0.02049100500181209, -0.020785635952894,
      0.04947852989598411, 0.06482111548156383, -0.009581030529691677, 0.04793283712358078, -0.01762540683538251,
      -0.05870646595009307, -0.03311020743630627, -0.02115987556690419, -0.05655744991606697, -0.09040393866933927,
      -0.05146608385449538, 0.05518335865947967, -0.0032339352038669228, 0.046336075994504136, -0.014376914076136163,
      0.08904552410809097, 0.012316229472765852, -0.02422682851110348, -0.04526050741813361, 0.005005011628938665,
      0.01579197918075174, 0.0009326040455041038, -0.022012199298815805, 0.06662550690677817, -0.05942185661541584,
      0.025174442425966344, -0.1581378162336398, -0.023656727806465157, -0.06615747779098971, -0.021200461384811502,
      -0.10492590748601793, 0.013216400076207266, -0.07260674866753007, 0.03479282831084702, -0.018945645001625714,
      0.003268352135848968, -0.03942970155957286, 0.05115793113735443, -0.049462953918750205, -0.06425414390633222,
      0.019942613557452644, 0.021791168045178562, -0.021326057970596363, -0.022820935571456678, -0.017533337186683675,
      -0.0015876740277753455, 0.03982048581562447, -0.02960581074611568, -0.01729500921130471, -0.021568541963040835,
      -0.04573349565238891, -0.0237223359914934, -0.002210372541374653, -0.06689519225625376, 0.07848992709873846,
      0.046778763647520344, -0.05911154360042071, -0.03118529822598732, -0.05679523166798126, 0.06162482345971637,
      0.003908817870256519, -0.04683091168544466, -0.009213138429041765, -0.01328771513385485, 0.03259751333720416,
      0.04395073989172737, -0.0811468396830596, 0.04845877535529435, -0.07083769535330343, -0.0507197065826129,
      -0.04441601912578255, -0.04759764884142386, 0.02227314089953461, 0.006962455190572275, 0.0422025111562721,
      -0.005070692541889156, -0.008657786439986553, 0.03334115942876724, 0.05034731296928091, -0.06237273515577752,
      0.03169688261728561, 0.0525301195182327, -0.10242543335831307, 0.00549721825636573, 0.006537354180561284,
      -0.0468299148785863, -0.0064199988217466825, -0.06496973994462289, 0.00791582507495514, 0.02191566399280062,
      0.0850440222011515, 0.02373775599024421, -0.057612662490382785, -0.013210842127082656, 0.04938493184213635,
      -0.00901240847552245, -0.0962640250955775, -0.02882481666799648, 0.08692372361095102, -0.010623250438046138};
  suite = coco_suite("bbob-mixint", "", "dimensions: 160 instance_indices: 5 function_indices: 10");
  problem = coco_suite_get_next_problem(suite, NULL);
  f = coco_allocate_vector(1);
  coco_evaluate_function(problem, x, f);
  printf("\n%s, %f", coco_problem_get_id(problem), f[0]);
  coco_free_memory(f);
  coco_suite_free(suite);
  return 0;
}

int true_main(void) {

  coco_random_state_t *random_generator = coco_random_new(RANDOM_SEED);

  /* Change the log level to "warning" to get less output */
  coco_set_log_level("info");

  printf("Running the example experiment... (might take time, be patient)\n");
  fflush(stdout);

  /**
   * Start the actual experiments on a test suite and use a matching logger, for
   * example one of the following:
   *   bbob                 24 unconstrained noiseless single-objective functions
   *   bbob-biobj           55 unconstrained noiseless bi-objective functions
   *   bbob-biobj-ext       92 unconstrained noiseless bi-objective functions
   *   [bbob-constrained*   48 constrained noiseless single-objective functions]
   *   bbob-largescale      24 unconstrained noiseless single-objective functions in large dimension
   *   bbob-mixint          24 unconstrained noiseless single-objective functions with mixed-integer variables
   *   bbob-biobj-mixint    92 unconstrained noiseless bi-objective functions with mixed-integer variables
   *
   * Suites with a star are partly implemented but not yet fully supported.
   *
   * Adapt to your need. Note that the experiment is run according
   * to the settings, defined in example_experiment(...) below.
   */
  coco_set_log_level("info");

  /**
   * For more details on how to change the default suite and observer options, see
   * http://numbbo.github.io/coco-doc/C/#suite-parameters and
   * http://numbbo.github.io/coco-doc/C/#observer-parameters. */

  example_experiment("bbob", "", "bbob", "result_folder: RS_on_bbob", random_generator);

  printf("Done!\n");
  fflush(stdout);

  coco_random_free(random_generator);

  return 0;
}

/**
 * A simple example of benchmarking random search on a given suite with default instances
 * that can serve also as a timing experiment.
 *
 * @param suite_name Name of the suite (e.g. "bbob" or "bbob-biobj").
 * @param suite_options Options of the suite (e.g. "dimensions: 2,3,5,10,20 instance_indices: 1-5").
 * @param observer_name Name of the observer matching with the chosen suite (e.g. "bbob-biobj"
 * when using the "bbob-biobj-ext" suite).
 * @param observer_options Options of the observer (e.g. "result_folder: folder_name")
 * @param random_generator The random number generator.
 */
void example_experiment(const char *suite_name,
                        const char *suite_options,
                        const char *observer_name,
                        const char *observer_options,
                        coco_random_state_t *random_generator) {

  size_t run;
  coco_suite_t *suite;
  coco_observer_t *observer;
  timing_data_t *timing_data;

  /* Initialize the suite and observer. */
  suite = coco_suite(suite_name, "", suite_options);
  observer = coco_observer(observer_name, observer_options);

  /* Initialize timing */
  timing_data = timing_data_initialize(suite);

  /* Iterate over all problems in the suite */
  while ((PROBLEM = coco_suite_get_next_problem(suite, observer)) != NULL) {

    size_t dimension = coco_problem_get_dimension(PROBLEM);

    /* Run the algorithm at least once */
    for (run = 1; run <= 1 + INDEPENDENT_RESTARTS; run++) {

      long evaluations_done = (long) (coco_problem_get_evaluations(PROBLEM) +
            coco_problem_get_evaluations_constraints(PROBLEM));
      long evaluations_remaining = (long) (dimension * BUDGET_MULTIPLIER) - evaluations_done;

      /* Break the loop if the target was hit or there are no more remaining evaluations */
      if ((coco_problem_final_target_hit(PROBLEM) &&
           coco_problem_get_number_of_constraints(PROBLEM) == 0)
           || (evaluations_remaining <= 0))
        break;

      /* Call the optimization algorithm for the remaining number of evaluations */
      my_random_search(evaluate_function,
                       evaluate_constraint,
                       dimension,
                       coco_problem_get_number_of_objectives(PROBLEM),
                       coco_problem_get_number_of_constraints(PROBLEM),
                       coco_problem_get_smallest_values_of_interest(PROBLEM),
                       coco_problem_get_largest_values_of_interest(PROBLEM),
                       coco_problem_get_number_of_integer_variables(PROBLEM),
                       (size_t) evaluations_remaining,
                       random_generator);

      /* Break the loop if the algorithm performed no evaluations or an unexpected thing happened */
      if (coco_problem_get_evaluations(PROBLEM) == evaluations_done) {
        printf("WARNING: Budget has not been exhausted (%lu/%lu evaluations done)!\n",
        		(unsigned long) evaluations_done, (unsigned long) dimension * BUDGET_MULTIPLIER);
        break;
      }
      else if (coco_problem_get_evaluations(PROBLEM) < evaluations_done)
        coco_error("Something unexpected happened - function evaluations were decreased!");
    }

    /* Keep track of time */
    timing_data_time_problem(timing_data, PROBLEM);
  }

  /* Output and finalize the timing data */
  timing_data_finalize(timing_data);

  coco_observer_free(observer);
  coco_suite_free(suite);

}

/**
 * A random search algorithm that can be used for single- as well as multi-objective optimization. The
 * problem's initial solution is evaluated first.
 *
 * @param evaluate_func The function used to evaluate the objective function.
 * @param evaluate_cons The function used to evaluate the constraints.
 * @param dimension The number of variables.
 * @param number_of_objectives The number of objectives.
 * @param number_of_constraints The number of constraints.
 * @param lower_bounds The lower bounds of the region of interested (a vector containing dimension values).
 * @param upper_bounds The upper bounds of the region of interested (a vector containing dimension values).
 * @param number_of_integer_variables The number of integer variables (if > 0, all integer variables come
 * before any continuous ones).
 * @param max_budget The maximal number of evaluations.
 * @param random_generator Pointer to a random number generator able to produce uniformly and normally
 * distributed random numbers.
 */
void my_random_search(evaluate_function_t evaluate_func,
                      evaluate_function_t evaluate_cons,
                      const size_t dimension,
                      const size_t number_of_objectives,
                      const size_t number_of_constraints,
                      const double *lower_bounds,
                      const double *upper_bounds,
                      const size_t number_of_integer_variables,
                      const size_t max_budget,
                      coco_random_state_t *random_generator) {

  double *x = coco_allocate_vector(dimension);
  double *functions_values = coco_allocate_vector(number_of_objectives);
  double *constraints_values = NULL;
  double range;
  size_t i, j;

  if (number_of_constraints > 0 )
    constraints_values = coco_allocate_vector(number_of_constraints);

  coco_problem_get_initial_solution(PROBLEM, x);
  evaluate_func(x, functions_values);

  for (i = 1; i < max_budget; ++i) {

    /* Construct x as a random point between the lower and upper bounds */
    for (j = 0; j < dimension; ++j) {
      range = upper_bounds[j] - lower_bounds[j];
      x[j] = lower_bounds[j] + coco_random_uniform(random_generator) * range;
      /* Round the variable if integer */
      if (j < number_of_integer_variables)
        x[j] = floor(x[j] + 0.5);
    }

    /* Evaluate COCO's constraints function if problem is constrained */
    if (number_of_constraints > 0 )
      evaluate_cons(x, constraints_values);

    /* Call COCO's evaluate function where all the logging is performed */
    evaluate_func(x, functions_values);

  }

  coco_free_memory(x);
  coco_free_memory(functions_values);
  if (number_of_constraints > 0 )
    coco_free_memory(constraints_values);
}

/**
 * A grid search optimizer that can be used for single- as well as multi-objective optimization.
 *
 * @param evaluate_func The evaluation function used to evaluate the solutions.
 * @param evaluate_cons The function used to evaluate the constraints.
 * @param dimension The number of variables.
 * @param number_of_objectives The number of objectives.
 * @param number_of_constraints The number of constraints.
 * @param lower_bounds The lower bounds of the region of interested (a vector containing dimension values).
 * @param upper_bounds The upper bounds of the region of interested (a vector containing dimension values).
 * @param number_of_integer_variables The number of integer variables (if > 0, all integer variables come
 * before any continuous ones).
 * @param max_budget The maximal number of evaluations.
 *
 * If max_budget is not enough to cover even the smallest possible grid, only the first max_budget
 * nodes of the grid are evaluated.
 */
void my_grid_search(evaluate_function_t evaluate_func,
                    evaluate_function_t evaluate_cons,
                    const size_t dimension,
                    const size_t number_of_objectives,
                    const size_t number_of_constraints,
                    const double *lower_bounds,
                    const double *upper_bounds,
                    const size_t number_of_integer_variables,
                    const size_t max_budget) {


  double *x = coco_allocate_vector(dimension);
  double *func_values = coco_allocate_vector(number_of_objectives);
  double *cons_values = NULL;
  long *nodes = (long *) coco_allocate_memory(sizeof(long) * dimension);
  double *grid_step = coco_allocate_vector(dimension);
  size_t i, j;
  size_t evaluations = 0;
  long *max_nodes = (long *) coco_allocate_memory(sizeof(long) * dimension);
  long integer_nodes = 1;

  /* Initialization */
  for (j = 0; j < dimension; j++) {
    nodes[j] = 0;
    if (j < number_of_integer_variables) {
      grid_step[j] = 1;
      max_nodes[j] = (long) floor(upper_bounds[j] + 0.5);
      assert(fabs(lower_bounds[j]) < 1e-6);
      assert(max_nodes[j] > 0);
      integer_nodes *= max_nodes[j];
    }
    else {
      max_nodes[j] = (long) floor(pow((double) max_budget / (double) integer_nodes,
          1 / (double) (dimension - number_of_integer_variables))) - 1;
      /* Take care of the borderline case */
      if (max_nodes[j] < 1)
        max_nodes[j] = 1;
      grid_step[j] = (upper_bounds[j] - lower_bounds[j]) / (double) max_nodes[j];
    }
  }

  if (number_of_constraints > 0 )
    cons_values = coco_allocate_vector(number_of_constraints);

  while (evaluations < max_budget) {

    /* Stop if there are no more nodes */
    if ((number_of_integer_variables == dimension) && (evaluations >= integer_nodes))
      break;

    /* Construct x and evaluate it */
    for (j = 0; j < dimension; j++) {
      x[j] = lower_bounds[j] + grid_step[j] * (double) nodes[j];
    }

    /* Evaluate COCO's constraints function if problem is constrained */
    if (number_of_constraints > 0 )
      evaluate_cons(x, cons_values);

    /* Call COCO's evaluate function where all the logging is performed */
    evaluate_func(x, func_values);
    evaluations++;

    /* Inside the grid, move to the next node */
    if (nodes[0] < max_nodes[0]) {
      nodes[0]++;
    }

    /* At an outside node of the grid, move to the next level */
    else if (max_nodes[0] > 0) {
      for (j = 1; j < dimension; j++) {
        if (nodes[j] < max_nodes[j]) {
          nodes[j]++;
          for (i = 0; i < j; i++)
            nodes[i] = 0;
          break;
        }
      }

      /* At the end of the grid, exit */
      if ((j == dimension) && (nodes[j - 1] == max_nodes[j - 1]))
        break;
    }
  }

  coco_free_memory(x);
  coco_free_memory(func_values);
  if (number_of_constraints > 0 )
    coco_free_memory(cons_values);
  coco_free_memory(nodes);
  coco_free_memory(grid_step);
  coco_free_memory(max_nodes);
}

/**
 * Allocates memory for the timing_data_t object and initializes it.
 */
static timing_data_t *timing_data_initialize(coco_suite_t *suite) {

	timing_data_t *timing_data = (timing_data_t *) coco_allocate_memory(sizeof(*timing_data));
	size_t function_idx, dimension_idx, instance_idx, i;

	/* Find out the number of all dimensions */
	coco_suite_decode_problem_index(suite, coco_suite_get_number_of_problems(suite) - 1, &function_idx,
			&dimension_idx, &instance_idx);
	timing_data->number_of_dimensions = dimension_idx + 1;
	timing_data->current_idx = 0;
	timing_data->output = (char **) coco_allocate_memory(timing_data->number_of_dimensions * sizeof(char *));
	for (i = 0; i < timing_data->number_of_dimensions; i++) {
		timing_data->output[i] = NULL;
	}
	timing_data->previous_dimension = 0;
	timing_data->cumulative_evaluations = 0;
	time(&timing_data->start_time);
	time(&timing_data->overall_start_time);

	return timing_data;
}

/**
 * Keeps track of the total number of evaluations and elapsed time. Produces an output string when the
 * current problem is of a different dimension than the previous one or when NULL.
 */
static void timing_data_time_problem(timing_data_t *timing_data, coco_problem_t *problem) {

	double elapsed_seconds = 0;

	if ((problem == NULL) || (timing_data->previous_dimension != coco_problem_get_dimension(problem))) {

		/* Output existing timing information */
		if (timing_data->cumulative_evaluations > 0) {
			time_t now;
			time(&now);
			elapsed_seconds = difftime(now, timing_data->start_time) / (double) timing_data->cumulative_evaluations;
			timing_data->output[timing_data->current_idx++] = coco_strdupf("d=%lu done in %.2e seconds/evaluation\n",
					timing_data->previous_dimension, elapsed_seconds);
		}

		if (problem != NULL) {
			/* Re-initialize the timing_data */
			timing_data->previous_dimension = coco_problem_get_dimension(problem);
			timing_data->cumulative_evaluations = coco_problem_get_evaluations(problem);
			time(&timing_data->start_time);
		}

	} else {
		timing_data->cumulative_evaluations += coco_problem_get_evaluations(problem);
	}
}

/**
 * Outputs and finalizes the given timing data.
 */
static void timing_data_finalize(timing_data_t *timing_data) {

	/* Record the last problem */
	timing_data_time_problem(timing_data, NULL);

  if (timing_data) {
  	size_t i;
  	double elapsed_seconds;
		time_t now;
		int hours, minutes, seconds;

		time(&now);
		elapsed_seconds = difftime(now, timing_data->overall_start_time);

  	printf("\n");
  	for (i = 0; i < timing_data->number_of_dimensions; i++) {
    	if (timing_data->output[i]) {
				printf("%s", timing_data->output[i]);
				coco_free_memory(timing_data->output[i]);
    	}
    }
  	hours = (int) elapsed_seconds / 3600;
  	minutes = ((int) elapsed_seconds % 3600) / 60;
  	seconds = (int)elapsed_seconds - (hours * 3600) - (minutes * 60);
  	printf("Total elapsed time: %dh%02dm%02ds\n", hours, minutes, seconds);

    coco_free_memory(timing_data->output);
    coco_free_memory(timing_data);
  }
}
