How to release the Coco software?
=================================

In principle, users shall download the latest release from the master branch of the repository. In order to do a new
release, the following steps must be followed. 

Simple Version
--------------
1. Draft a release (under the Code / releases tabs). Consider previous releases to see what to write. 
2. Merge the `development` branch into the `master` branch. The `master` branch is protected, that is,
   the development branch must have passed all required tests (as should be the case by default) such 
   that this is possible. 
3. Publish the release. 

Advanced Version
----------------
1. update version and release numbers in the documentation in all docs/*/source/config.py files
2. update version and release numbers in C documentation (coco.h) for producing web documentation 
3. publish the new documentation by running `make html-topublish` in all docs/* folders and pushing the created html files,
   see also documentation-howto.md
4. publish the Coco C documentation through doxygen by following the instructions in documentation-howto.md
5. check that README.md is up-to-date
6. clean and test the development branch
7. merge the cleaned and tested development branch into the master branch and tag it

or

6. create a release branch
6a. clean and test the release branch
6b. merge the release branch into the development branch
7. merge the release branch into the master branch and tag it
