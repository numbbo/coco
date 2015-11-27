# Unit-testing instructions

We use the following naming convention for unit test files: you add a prefix "test_" to the name of the file that you are testing (e.g. coco_utilities.c -> test_coco_utilities.c). 

All test files should be included in this folder. Each test file should contain a central function, in which are the tests from the file are called (e.g. test_all_coco_utilities in test_coco_utilities.c). A specific test should be testing a small part of the code. See test_coco_utilities.c for some test examples.

The starting file for unit testing is unit_test.c, from which all the other test files are called. When you add a new test file, you need to include it in unit_test.c and add it's central function to the function run_all_tests.

In the root folder you can find the do.py script. You should call "python do.py test-c-unit" to run the tests.

For more information on cmocka framework you can check https://cmocka.org/

