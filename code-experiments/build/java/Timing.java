import java.util.Locale;
import java.util.concurrent.TimeUnit;

public class Timing {

	private StringBuilder output;
	private long previousDimension;
	private long cumulativeEvaluations;
	private long startTime;
	private long overallStartTime;

	/**
	 * Constructor
	 */
	public Timing() {
		
		super();
		
		this.output = new StringBuilder();
		this.previousDimension = 0;
		this.cumulativeEvaluations = 0;
		this.startTime = System.nanoTime();
		this.overallStartTime = this.startTime;
	}

	/**
	 * Keeps track of the total number of evaluations and elapsed time. Produces an output string when the
	 * current problem is of a different dimension than the previous one or when null.
	 */
	void timeProblem(Problem problem) {

		if ((problem == null) || (this.previousDimension != CocoJNI.cocoProblemGetDimension(problem.getPointer()))) {

			/* Output existing timing information */
			if (this.cumulativeEvaluations > 0) {
				long elapsedTime = System.nanoTime() - this.startTime;
				String elapsed = String.format(Locale.ENGLISH, "%.2e", elapsedTime / (1.0 * 1e+9) / (1.0 * this.cumulativeEvaluations));
				this.output.append("d=" + this.previousDimension + " done in " + elapsed + " seconds/evaluation\n");
			}

			if (problem != null) {
				/* Re-initialize the timing_data */
				this.previousDimension = CocoJNI.cocoProblemGetDimension(problem.getPointer());
				this.cumulativeEvaluations = CocoJNI.cocoProblemGetEvaluations(problem.getPointer());
				this.startTime = System.nanoTime();
			}

		} else {
			this.cumulativeEvaluations += CocoJNI.cocoProblemGetEvaluations(problem.getPointer());
		}
	}
	
	/**
	 * Outputs the collected timing data.
	 */
	void output() {

		/* Record the last problem */
		timeProblem(null);

		long elapsedTime = System.nanoTime() - this.overallStartTime;
		long hours = TimeUnit.HOURS.convert(elapsedTime, TimeUnit.NANOSECONDS);
		long minutes = TimeUnit.MINUTES.convert(elapsedTime, TimeUnit.NANOSECONDS) - hours * 60;
		long seconds = TimeUnit.SECONDS.convert(elapsedTime, TimeUnit.NANOSECONDS) - hours * 3600 - minutes * 60;
		String elapsed = String.format("Total elapsed time: %dh%02dm%02ds\n", hours, minutes, seconds);

		this.output.insert(0, "\n");
		this.output.append(elapsed);
		
		System.out.append(this.output.toString());
		
	}
}
