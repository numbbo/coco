public class NoSuchProblemException extends Exception {
	
    String suit;
    int function_id;
    public NoSuchProblemException(String suit, int function_id) {
    	this.suit = suit;
    	this.function_id = function_id;
	}
	@Override
	public String toString() {
		return "Problem suit " + this.suit + " lacks a function with function id " + this.function_id;
	}
    
}
