How to release the Coco software?
=================================

In principle, users shall download the latest release from the master branch of the repository. In order to do a new
release, the following steps must be followed. 

Simple Version (without documentation)
--------------------------------------
1. Draft a release (under the Code / releases tabs). Consider previous releases to see what to write. 
2. Merge the `development` branch into the `master` branch. The `master` branch is protected, that is,
   the development branch must have passed all required tests (as should be the case by default) such 
   that this is possible. 
3. Publish the release. 

Advanced Version (with documentation)
-------------------------------------
1. Draft a release (under the Code / releases tabs). Consider previous releases to see what to write. 
2. update version and release numbers in the documentation in all docs/*/source/config.py files
3. update version and release numbers in C documentation (coco.h) for producing web documentation 
4. publish the new documentation by running `make html-topublish` in all docs/* folders and pushing the created html files,
   see also documentation-howto.md
5. publish the Coco C documentation through doxygen by following the instructions in documentation-howto.md
6. check that README.md is up-to-date
7. clean and test the development branch
8. Merge the `development` branch into the `master` branch.
9. Publish the release. 


Instead of merging directly the `development` branch, another approach can be useful if not all functionality
shall be contained in the release:

2./7. Create a release branch from `development`
2a./7a. Clean and test the release branch
2b./7b. Merge the release branch back into the `development` branch
2c./7c. Merge the release branch into the `master` branch.
3./8. Publish the release.
