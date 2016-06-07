The documentation of the COCO platform is split into several parts:

### Documentation of the COCO source code and its functioning: coco-documentation
The COCO source code itself is documented directly within itself in the code-experiments/ folders and then transformed
via doxygen into html and published at numbbo.github.io/coco-documentation. More details about how to deal with this 
part can be found below.

### Documentation of generic Coco setting
The generic parts of Coco are documented in two separate documents:
* coco-generic/experimental-setup for the description of the experimental setup (source in
  ../docs/coco-generic/experimental-setup/source and then published via Sphinx to
  https://numbbo.github.io/coco-doc/experimental-setup)
* coco-generic/perf-assessment for the description of the performance assessment, plots, etc. (source in
  ../docs/coco-generic/perf-assessment/source and then published via Sphinx to
  https://numbbo.github.io/coco-doc/perf-assessment)

### Documentation of the test suites
Each test suite will have separate documentations for the function documentation and (potentially) the experimental setting.
For the moment, we have the following documentations:
* bbob-biobj/perf-assessment (source in ../docs/bbob-biobj/perf-assessment/source and then published via Sphinx to 
  https://numbbo.github.io/coco-doc/bbob-biobj/perf-assessment)
* bbob-biobj/functions (source in ../docs/bbob-biobj/functions/source and then published via Sphinx to 
  https://numbbo.github.io/coco-doc/bbob-biobj/functions)
* bbob-largescale/functions (source in ../docs/bbob-largescale/functions/source and then published via Sphinx to 
  https://numbbo.github.io/coco-doc/bbob-largescale/functions)
* bbob-constr-lin/functions (source in ../docs/bbob-constr-lin/functions/source and then published via Sphinx to 
  https://numbbo.github.io/coco-doc/bbob-constr-lin/functions)
* bbob-experiments (old documentation available at http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf)
* bbob-functions (old documentation available at http://coco.lri.fr/downloads/download15.03/bbobdocexperiment.pdf)
* bbob-noisy-functions (old documentation available at http://coco.lri.fr/downloads/download15.03/bbobdocnoisyfunctions.pdf)

### Documentations related to the workshops
Web pages for the workshops are also created from reStructuredText via Sphinx. The source files can be found in the 
docs/workshops/ folder of the numbbo/coco github repository.



HowTo: Edit and Publish the experimental setting and function documentations
----------------------------------------------------------------------------
The source files (as reStructuredText) can be found and edited in the subfolders of ../docs/. Their reStructuredText is
translated into html, latex, or pdf with the help of Sphinx (http://www.sphinx-doc.org). To build the html, latex, or pdf
output for each documentation, you need to install sphinx by typing `pip install -U Sphinx` (if you have installed pip) and then
type `make html`, `make latex`, or `make latexpdf` in the corresponding subfolder of ../docs/.

For publishing the html to the web, you need to have, in addition, a clone of the numbbo/coco-doc or numbbo/workshops git
repository at the same level than your numbbo/coco repository clone. Those documentation repositories have only one branch
called `gh-pages` which is directly translated into the web page https://username.github.io/repositoryname, i.e., here for
example https://numbbo.github.io/coco-doc. Once you have the clone of the documentation repository, you can type
`make html-topublish` in the corresponding subfolder of the ../docs/ folder in the numbbo/coco repository. This will create the
same html files than `make html` but instead of the build/ subfolder, it uses the correct folder in the gh-pages branch.
A `git push` of these changes will then directly update the web page.

#### Summary:
- need: python, sphinx, git
- edit `.rst` sources in docs/FOLDERNAME and commit and push as usual within the docs branch of the numbbo/coco repository
- for checking the output, type `make html`, `make latex`, or `make latexpdf` (output written to build/ subfolder)
- for publishing the changes to the web:
  - have the `gh-pages` branch of the corresponding github repository at the same level as your numbbo/coco folder
  - create the html with `make html-topublish`
  - commit and push the changes in the `gh-pages` branch to update the web page

#### Current documentations and their repositories:
- sources in ../docs/bbob-biobj/functions are written into bbob-biobj/functions folder of numbbo/coco-doc repository
- sources in ../docs/bbob-biobj/perf-assessment are written into bbob-biobj/perf-assessment folder of
  numbbo/coco-doc repository
- sources in ../docs/bbob-largescale/functions are written into bbob-largescale/functions folder of numbbo/coco-doc repository
- sources in ../docs/bbob-constr-lin/functions are written into bbob-constr-lin/functions folder of numbbo/coco-doc repository
- sources in ../docs/workshops are written into numbbo/workshops repository
- sources in ../docs/coco-documentation are written into numbbo/coco-doc repository [see also below, because this
  documentation is partly build from the source code]


HowTo: Edit and Publish the COCO documentation
----------------------------------------------
In addition to the reStructuredText which explains the general functioning of the COCO platform in
../docs/coco-documentation/ (which is translated into html via Sphinx as described above), parts of the COCO 
documentation are automatically extracted from the source code.

The C code in code-experiments/src/ for example is translated into html in the C/ subfolder of the `gh-pages` branch
of the numbbo/coco-doc github repository with the help of the doxygen tool (www.doxygen.org/). After installing 
doxygen, and having a clone of the `gh-pages` branch of the numbbo/coco-doc github repository in the same folder 
than your numbbo/coco checkout, you can create the html output in this directory by simply typing `doxygen` in the 
docs/coco-documentation/C/ folder of the numbbo/coco github repository. Afterwards, commit and push of this
repository will again update the web page directly as described above.
