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
		this.lower_bounds = JNIinterface.cocoGetSmallestValuesOfInterest(this.problem);
		this.upper_bounds = JNIinterface.cocoGetLargestValuesOfInterest(this.problem);
		this.number_of_variables = JNIinterface.cocoGetNumberOfVariables(this.problem);
		this.number_of_objectives = JNIinterface.cocoGetNumberOfObjectives(this.problem);
	}
	
	public void addObserver(String observer, String options) {
		this.problem = JNIinterface.cocoObserveProblem(observer, this.problem, options);
	}
	
	public void free() {
		// check this.problem != NULL
		JNIinterface.cocoFreeProblem(this.problem);
	}
	
	public String id() {
		// check this.problem != NULL
		return JNIinterface.cocoGetProblemId(this.problem);
	}
	
	public String name() {
		// check this.problem != NULL
		return JNIinterface.cocoGetProblemName(this.problem);
	}
	
	public int evaluations() {
		return JNIinterface.cocoGetEvaluations(this.problem);
	}
	
	/* toString method */
	@Override
	public String toString() {
        String pb_id = JNIinterface.cocoGetProblemId(this.problem);
        if (pb_id != null) {
            return pb_id;
        }
		else
			return "finalized/invalid problem";
	}
}
