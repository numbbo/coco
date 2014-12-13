package javacoco;

public class Problem {
	int number_of_variables; // update after coco_get_problem(...) call
	int number_of_objectives; // update after coco_get_problem(...) call
	int number_of_constraints; // update after coco_get_problem(...) call
	public double[] smallest_values_of_interest; // update after coco_get_problem(...) call
	public double[] largest_values_of_interest; // update after coco_get_problem(...) call
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
		this.observer_name = new String("");
		this.options = new String("");
	}
	
	public Problem(String problem_suit, int function_index, String observer_name, String options) {
		super();
		this.problem_suit = new String(problem_suit);
		this.function_index = function_index;
		this.observer_name = new String(observer_name);
		this.options = new String(options);
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
