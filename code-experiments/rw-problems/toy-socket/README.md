# The `toy-socket` and `toy-socket-biobj` test suites

These two suites (the first is single-objective and the second bi-objective) serve as 
examples for showing how to evaluate solutions using external evaluators. This can be 
beneficial for benchmarking algorithms on real-world problems.

Two external evaluators are available - one in Python and the other in C (see the files
in this folder). Both are implemented using two files - one takes care of the socket 
communication (files `socket_server.py` and `socket_server.c` for Python and C, respectively) 
and the other of the actual evaluation of solutions (files `toy_evaluator.py` and 
`toy_evaluator.c` for Python and C, respectively). This should make it rather easy to 
replace the toy evaluators with actual (real-world) evaluators. 

The external evaluators are basically servers that need to run and listen for messages
from the client (COCO). At each solution evaluation, COCO sends a message with information
on the problem and the solution to the external evaluator. The external evaluator then 
computes and returns the objective values to COCO as a response message. 

**IMPORTANT: The servers need to be up and running before COCO experiments are started!**

## Running experiments on the two suites

### Running the external evaluators

Compilation of the external evaluator in C can be done through `do.py` (at the root directory
of the repository):
````
python do.py build-socket
````

From the current directory, the Python external evaluator is called with:
````
python socket_server.py
````
and the C external evaluator with:
````
./socket_server
````
on Linux and Mac and
````
socket_server.exe
````
on Windows.

### Running the example experiments

After the external evaluators have been called, the example experiments in the available
languages can be invoked as usual using `do.py`. Note that the `bbob` suite should be 
changed to `toy-socket` in the example experiments!