## Version number:
MAJOR    = 0
MINOR    = 1
REVISION = 0

## Tools:
AMALGAMATE ?= tools/amalgamate
M4 ?= m4
GTAR ?= tar
PYTHON2 ?= python2.7
R ?= R
RSCRIPT ?= Rscript

########################################################################
## No need to change anythong below
########################################################################

GTAR_CREATE_FLAGS ?= --format=posix --owner=0 --group=0 --mode='ug+rw,o+r'

VERSION = ${MAJOR}.${MINOR}.${REVISION}
C_VERSION = ${MAJOR}.${MINOR}.${REVISION}
PYTHON_VERSION = ${MAJOR}.${MINOR}.${REVISION}
R_VERSION = ${MAJOR}.${MINOR}-${REVISION}

C_TARGETS = \
	build/c/coco.c \
	build/c/coco.h \
	build/c/VERSION 

PYTHON_TARGETS = \
	build/python/setup.py \
	build/python/README \
	build/python/coco/coco.c \
	build/python/coco/coco.h 

R_TARGETS = \
	build/r/skel/src/coco.c \
	build/r/skel/src/coco.h \
	build/r/skel/DESCRIPTION

DOXYGEN_TARGETS = \
	build/doxygen/xml/index.xml

TARGETS = ${C_TARGETS} ${PYTHON_TARGETS} ${R_TARGETS} ${DOXYGEN_TARGETS}


## Order matters, do not change! Not all files need to be listed
## because most are included by others during the amalgamation.
COCO_C = \
	src/coco_benchmark.c \
	src/coco_random.c \
	src/coco_generics.c

COCO_H = src/coco.h

.PHONEY: clean all release r_release c_release python_release
.SILENT:

all: ${TARGETS}

clean:
	echo "  RM    ${TARGETS}"
	rm -f ${TARGETS}
	echo "  RM    build/c/demo.o build/c/demo"
	echo "  RM    build/c/cppdemo.o build/c/cppdemo"
	echo "  RM    build/c/coco.o"
	rm -fR build/c/demo.o build/c/demo build/c/cppdemo.o build/c/cppdemo build/c/coco.o
	echo "  RM    build/python/dist"
	rm -fR build/python/dist
	echo "  RM    build/python/coco.egg-info"
	rm -fR build/python/coco.egg-info
	echo "  RM    build/python/MANIFEST"
	rm -f build/python/MANIFEST
	rm -f python-build.log
	echo "  RM    build/r/pkg"
	rm -fR build/r/pkg
	rm -f build/r/roxygen.log
	rm -f r-build.log
	echo "  RM    build/doxygen/*"
	rm -fR build/doxygen

release: c_release python_release r_release

r_release: ${R_TARGETS}

########################################################################
## Doxygen documentation
build/doxygen:
	mkdir -p $@

build/doxygen/xml/index.xml: src/coco.h build/doxygen
	doxygen doxygen.ini

########################################################################
## C framework
build/c/coco.c: ${COCO_C} src/coco_c_runtime.c
	echo "  AM    $@"
	${AMALGAMATE} $+ > $@

build/c/coco.h: ${COCO_H}
	echo "  CP    $@"
	cp $+ $@

build/c/VERSION:
	echo "  MK    $@"
	echo "${C_VERSION}" > $@

## OME: This target is _not_ built by default, only rerun if you
## change the generate-bbob-test.R script. We do not assume that
## everyone working on the C code has a working R installed.
build/c/test_bbob2009.h:
	echo "  MK    $@"
	${RSCRIPT} tools/generate-bbob-tests.R > $@

release/c/coco-${C_VERSION}.tar.gz: ${C_TARGETS}
	echo "  TAR   $@"
	${GTAR} ${GTAR_CREATE_FLAGS} -czf $@ --transform='s,build/c,coco,' $+

c_release: release/c/coco-${C_VERSION}.tar.gz

########################################################################
## Python framework
build/python/coco/coco.c: ${COCO_C} src/coco_c_runtime.c
	echo "  AM    $@"
	${AMALGAMATE} $+ > $@

build/python/coco/coco.h: ${COCO_H}
	echo "  CP    $@"
	cp $+ $@

build/python/%: build/python/%.in
	echo "  M4    $@"
	${M4} -D__COCO_VERSION__=${PYTHON_VERSION} $+ > $@

release/python/coco-${PYTHON_VERSION}.tar.gz: ${PYTHON_TARGETS}
	echo "  PY    $@"
	cd build/python; \
	${PYTHON2} setup.py sdist --dist-dir=${CURDIR}/release/python > \
	  ../../python-build.log

python_release: release/python/coco-${PYTHON_VERSION}.tar.gz

########################################################################
## R framework
build/r/skel/src/coco.c: ${COCO_C} src/coco_r_runtime.c
	echo "  AM    $@"
	${AMALGAMATE} $+ > $@

build/r/skel/src/coco.h: ${COCO_H}
	echo "  CP    $@"
	cp $+ $@

build/r/%: build/r/%.in
	echo "  M4    $@"
	${M4} -D__COCO_VERSION__=${R_VERSION} $+ > $@

release/r/coco_${R_VERSION}.tar.gz: ${R_TARGETS}
	echo "  CP    build/r/pkg"
	cp -R build/r/skel build/r/pkg
	echo "  ROXY  build/r/pkg"
	cd build/r ; ${RSCRIPT} ./tools/roxygenize > roxygen.log 2>&1
	echo "  R     $@"
	cd release/r/ ; \
	${R} CMD build ${CURDIR}/build/r/pkg > ${CURDIR}/r-build.log 2>&1

r_release: release/r/coco_${R_VERSION}.tar.gz
