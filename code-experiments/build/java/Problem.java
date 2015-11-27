// TODO: Important: find a way to check if a problem is NULL, i.e. is (coco_problem_t *)this.problem is NULL
public class Problem {
	long problem; // stores the pointer to the C coco_problem_t structure  
	int dim;
	int number_of_objectives;
	public double[] lower_bounds;
	public double[] upper_bounds;
	String problem_suite;
	long function_index; // AKA problem_index
	
	/* Constructor */
	public Problem(String problem_suite, long function_index) throws NoSuchProblemException {
		super();
		this.problem = JNIinterface.cocoSuiteGetProblem(problem_suite, function_index);
		if (!JNIinterface.cocoProblemIsValid(this.problem)){
			throw new NoSuchProblemException(problem_suite, function_index);
		}
		this.problem_suite = problem_suite;
		this.function_index = function_index;
		this.lower_bounds = JNIinterface.cocoProblemGetSmallestValuesOfInterest(this.problem);
		this.upper_bounds = JNIinterface.cocoProblemGetLargestValuesOfInterest(this.problem);
		this.dim = JNIinterface.cocoProblemGetDimension(this.problem);
		this.number_of_objectives = JNIinterface.cocoProblemGetNumberOfObjectives(this.problem);
	}
	
	public void addObserver(String observer, String options) {
		this.problem = JNIinterface.cocoProblemAddObserverDeprecated(this.problem, observer, options);
	}
	
	public void free() {
		// check this.problem != NULL
		JNIinterface.cocoProblemFree(this.problem);
	}
	
	// Methods or attributes? Can these values change after the constructor is called?
	public String id() {
		// check this.problem != NULL
		return JNIinterface.cocoProblemGetId(this.problem);
	}
	
	public String name() {
		// check this.problem != NULL
		return JNIinterface.cocoProblemGetName(this.problem);
	}
	
	public int evaluations() {
		return JNIinterface.cocoProblemGetEvaluations(this.problem);
	}
	
	/* toString method */
	@Override
	public String toString() {
        String pb_id = JNIinterface.cocoProblemGetId(this.problem);
        if (pb_id != null) {
            return pb_id;
        }
		else
			return "finalized/invalid problem";
	}
}