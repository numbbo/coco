public class Problem {
	long problem; // stores the pointer to the C coco_problem_t structure  
	int number_of_variables;
	int number_of_objectives;
	public double[] lower_bounds;
	public double[] upper_bounds;
	String problem_suit;
	int function_index;
	
	/* Constructor */
	public Problem(String problem_suit, int function_index) {
		super();
		this.problem = JNIinterface.cocoGetProblem(problem_suit, function_index);
		this.problem_suit = problem_suit;
		this.function_index = function_index;
		this.observer_name = new String("");
		this.options = new String("");
	}
	
	public void addObserver(String observer, String options) {
		this.problem = JNIinterface.cocoObserveProblem(observer, this.problem, options);
	}
	
	/* toString method */
	@Override
	public String toString() {
        String pb_id = JNIinterface.cocoGetProblemId(this);
        if (pb_id != null) {
            return pb_id;
        }
		else
			return "finalized/invalid problem";
	}
}
