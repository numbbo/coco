public class Observer {
	
	private long pointer; // Pointer to the coco_observer_t object
	private String name;

	/** 
	 * Constructs the observer from observerName and observerOptions.
	 * See http://numbbo.github.io/coco-doc/C/#observer-parameters for more information on 
	 * valid observer parameters.
	 * @param observerName
	 * @param observerOptions
	 * @throws Exception
	 */
	public Observer(String observerName, String observerOptions) throws Exception {

		super();
		try {
			this.pointer = CocoJNI.cocoGetObserver(observerName, observerOptions);
			this.name = observerName;
		} catch (Exception e) {
			throw new Exception("Observer constructor failed.\n" + e.toString());
		}
	}

	/**
	 * Finalizes the observer.
	 * @throws Exception 
	 */
	public void finalizeObserver() throws Exception {
		try {
			CocoJNI.cocoFinalizeObserver(this.pointer);
		} catch (Exception e) {
			throw new Exception("Observer finalization failed.\n" + e.toString());
		}
	}

	public long getPointer() {
		return this.pointer;
	}
	
	public String getName() {
		return this.name;
	}

	public void signalRestart(Problem problem) {
		CocoJNI.cocoObserverSignalRestart(this.getPointer(), problem.getPointer());
	}

	/* toString method */
	@Override
	public String toString() {
		return getName();
	}
}