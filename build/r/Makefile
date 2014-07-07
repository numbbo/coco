R	:= R --vanilla
RSCRIPT	:= Rscript --vanilla
DELETE	:= rm -fR

.SILENT:
.PHONEY: clean test check build install package data usage help

usage:
	echo "Available targets:"
	echo ""
	echo " install  - install the package, writing the output into install.log"
	echo " check    - run R CMD check on the package"
	echo " package  - build source package"
	echo " help     - shows all available targets"
	echo " test     - run test suite"

help: usage
	echo " clean    - clean up package cruft"
	echo " pkg      - roxygenize skel/ into pkg/"

install: clean pkg
	echo "Installing package ..."
	${R} CMD INSTALL pkg > install.log 2>&1

check: clean pkg
	echo "Running R CMD check ..."
	${R} CMD check pkg && ${DELETE} pkg.Rcheck

clean:
	echo "Cleaning up ..."
	${DELETE} build.log install.log roxygen.log package.log
	${DELETE} skel/src/*.o skel/src/*.so
	${DELETE} soobench.Rcheck
	${DELETE} pkg
	${DELETE} .RData .Rhistory

package: pkg
	echo "Building package file ..."
	${R} CMD build pkg/ > package.log 2>&1

pkg: clean
	echo "Roxygenizing package ..."
	${RSCRIPT} ./tools/roxygenize > roxygen.log 2>&1

test: install
	${RSCRIPT} ./tools/run-tests
