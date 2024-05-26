How to release the COCO software?
=================================

Since 2023 it is no longer recommended to use the master branch directly.
Instead, proper releases are cut for each tagged commit on the master branch and published to
the [GitHub Releases](https://github.com/numbbo/coco/releases) tab, [PyPI](https://pypi.org) for Python packages, and [crates.io](https://crates.io) for the Rust crate.

Simple Version (without documentation)
--------------------------------------
1. Draft a release (under the Code / releases tabs). Consider previous releases to see what to write.
   Optimally, drafting the release includes going through the issue tracker and the commits since the
   last release and summarizing the main differences.
2. Merge the `development` branch into the `master` branch. The `master` branch is protected, that is,
   the development branch must have passed all required tests (as should be the case by default) such 
   that this is possible. The nightly builds are explicitely included in these mandatory tests. To
   launch them by hand, if needed, do it directly on our Jenkins CI platform via ci.inria.fr.
3. Publish the release.


Advanced Version (with documentation)
-------------------------------------
 1. Draft a release (under the Code / releases tabs). Consider previous releases to see what to write.
 2. check that instance numbers are up-to-date and `"year: this-year"` is implemented as suite option
 2. check that README.md is up-to-date
 3. clean and test the development branch
 4. Run tests by pushing the development branch to the `devel-test1` and `test-nightly` branches, which
    are the ones, the master branch is protected against
 5. Merge the `development` branch into the `master` branch.
 6. Publish the release. 
 7. Adding the .tar.gz file of the release right after the release by hand will allow for
    tracking downloads later on, see http://mmilidoni.github.io/github-downloads-count/

Afterwards or before (can be skipped potentially if this is not affected by the release):

 8. update version and release numbers in the coco-doc documentation repository in
    all docs/*/source/config.py files
 9. publish the new documentation by running `make html-topublish` in all the docs/* folders 
    if uten 8. and pushing the created html files, see also documentation-howto.md
10. publish the Coco C documentation through doxygen by following the instructions in documentation-howto.md
11. publish the documentation of the cocoex and cocopp modules, see also the
    documentation-howto.md in the coco-doc repository for details

    
Instead of merging directly the `development` branch, another approach can be useful if not all functionality
shall be contained in the release:

2./5.   Create a release branch from `development`
2a./5a. Clean and test the release branch
2b./5b. Merge the release branch back into the `development` branch
2c./5c. Merge the release branch into the `master` branch.
3./6.   Publish the release.
