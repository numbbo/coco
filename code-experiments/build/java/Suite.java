public class Suite {
	
	private long pointer; // Pointer to the coco_suite_t object
	private String name;

	/**
	 * Constructs the suite from the given suiteName, suiteInstance and suiteOptions.
	 * See http://numbbo.github.io/coco-doc/C/#suite-parameters for more information on 
	 * valid suite parameters.
	 * @param suiteName
	 * @param suiteInstance
	 * @param suiteOptions
	 * @throws Exception
	 */
	public Suite(String suiteName, String suiteInstance, String suiteOptions) throws Exception {

		super();
		try {
			this.pointer = CocoJNI.cocoGetSuite(suiteName, suiteInstance, suiteOptions);
			this.name = suiteName;
		} catch (Exception e) {
			throw new Exception("Suite constructor failed.\n" + e.toString());
		}
	}
	
	/**
	 * Returns the number of problems in the suite. 
	 * @return the number of problems in the suite
	 * @throws Exception 
	 */
	public long getNumberOfProblems() throws Exception {
		
		try {		
			return CocoJNI.cocoSuiteGetNumberOfProblems(this.getPointer());
		} catch (Exception e) {
			throw new Exception("Fetching of problem failed.\n" + e.toString());
		}
	}
	
	/**
	 * Returns the (unobserved) problem in the suite with the given function, dimension and instance. 
	 * @return the corresponding problem in the suite or null when there is no such problem  
	 * @throws Exception 
	 */
	public Problem getProblemByFuncDimInst(long function, long dimension, long instance) throws Exception {
		
		try {		
			long problemPointer = CocoJNI.cocoSuiteGetProblemByFuncDimInst(this.getPointer(), 
					function, dimension, instance);
			
			if (problemPointer == 0)
				return null;
			
			return new Problem(problemPointer);
		} catch (Exception e) {
			throw new Exception("Fetching of problem failed.\n" + e.toString());
		}
	}
	
	/**
	 * Finalizes the suite.
	 * @throws Exception 
	 */
	public void finalizeSuite() throws Exception {
		try {
			CocoJNI.cocoFinalizeSuite(this.pointer);
		} catch (Exception e) {
			throw new Exception("Suite finalization failed.\n" + e.toString());
		}
	}

	public long getPointer() {
		return this.pointer;
	}
	
	public String getName() {
		return this.name;
	}

	/* toString method */
	@Override
	public String toString() {
		return getName();
	}
}