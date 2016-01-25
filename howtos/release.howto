How to release the Coco software?
---------------------------------

In principle, users shall download the latest release from the master branch of the repository. In order to do a new release, the following steps must be followed:

1) update version and release numbers in the documentation in all docs/*/source/config.py files
2) update version and release numbers in C documentation (coco.h) for producing web documentation 
3) publish the new documentation by running 'make html-topublish' in all docs/* folders and pushing the created html files, see also the documentation.howto
4) publish the Coco C documentation through doxygen by following the instructions in the documentation.howto
5) check that README.md is up-to-date
6) clean and test the development branch
7) merge the cleaned and tested development branch into the master branch and tag it

or

6) create a release branch
6a) clean and test the release branch
6b) merge the release branch into the development branch
7) merge the release branch into the master branch and tag it
