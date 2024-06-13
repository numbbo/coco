# Unit tests for core COCO framework

## Conventions

We use the following naming convention for unit test files: you add a prefix `test_` to the name of the file that you are testing (e.g. `coco_utilities.c` ->
`test_coco_utilities.c`). 

All test files should be included in this folder. Each test file should contain a central function, in which the tests from the file are called (e.g. `test_all_coco_utilities` in `test_coco_utilities.c`).
A specific test should be testing a small part of the code. See `test_coco_utilities.c` for some test examples.

## Running the Test

> [!NOTE]
>
> Please see `DEVELOPMENT.md` in the root directory for detailed instructions.

1. Fabricate the sources: `python ../../../scripts/fabricate`.
1. Configure build: `cmake -B build`
1. Build test suite: `cmake --build build`
1. Run tests: `ctest --test-dir build` or on Windows `ctest --test-dir build -C Debug`