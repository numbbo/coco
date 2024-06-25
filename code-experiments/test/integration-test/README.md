# Integration tests for core COCO framework

## Running the Test

> [!NOTE]
>
> Please see `DEVELOPMENT.md` in the root directory for detailed instructions.

1. Fabricate the sources: `python ../../../scripts/fabricate`.
1. Configure build: `cmake -B build`
1. Build test suite: `cmake --build build`
1. Run tests: `ctest --test-dir build` or on Windows `ctest --test-dir build -C Debug`