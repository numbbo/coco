package javacoco;

public abstract class JNIinterface {
	
	/* Load the library */
	static {
		System.loadLibrary("JNIinterface");
	}
	
	/* Inner classes */
	public class Problem {
		int number_of_variables; // update after coco_get_problem(...) call
		int number_of_objectives; // update after coco_get_problem(...) call
		int number_of_constraints; // update after coco_get_problem(...) call
		double[] smallest_values_of_interst; // update after coco_get_problem(...) call
		double[] largest_values_of_interst; // update after coco_get_problem(...) call
		double[] best_value; // update after coco_get_problem(...) call
		double[] best_parameter; // update after coco_get_problem(...) call
		String problem_name; // update after coco_get_problem(...) call
		String problem_id; // update after coco_get_problem(...) call
		String problem_suit;
		int function_index;
		String observer_name;
		String options; // information to generate data directories
		
		/* Constructor(s) */
		public Problem(String problem_suit, int function_index) {
			super();
			this.problem_suit = problem_suit;
			this.function_index = function_index;
		}
		
		public Problem(String problem_suit, int function_index, String observer_name, String options) {
			super();
			this.problem_suit = problem_suit;
			this.function_index = function_index;
			this.observer_name = observer_name;
			this.options = options;
		}
		
		/* TODO: Getters and setters */
		
		/* toString method */
		@Override
		public String toString() {
			if (this.problem_id != null)
				return this.problem_id;
			else
				return "finalized/invalid problem";
		}
	}
	
	public class Benchmark {
		String problem_suit;
		String observer;
		String options;
		int function_index;
		
		/* Constructor */
		public Benchmark(String problme_suit, String observer, String options) {
			super();
			this.problem_suit = problme_suit;
			this.observer = observer;
			this.options = options;
			this.function_index = 0;
		}
	}
	
	/* Class method */
	/* TODO: Handle NoSuchProblem exception */
	public Problem next_problem(Benchmark benchmark) {
		Problem problem = new Problem(benchmark.problem_suit, benchmark.function_index, benchmark.observer, benchmark.options);
		return problem;
	}
	

	/* Native method */
	public native double[] coco_evaluate_function(Problem p, double[] x);
}
