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
	build/c/numbbo.c \
	build/c/numbbo.h \
	build/c/VERSION

PYTHON_TARGETS = \
	build/python/setup.py \
	build/python/README \
	build/python/numbbo/numbbo.c \
	build/python/numbbo/numbbo.h 

R_TARGETS = \
	build/r/skel/src/numbbo.c \
	build/r/skel/src/numbbo.h \
	build/r/skel/DESCRIPTION

TARGETS = ${C_TARGETS} ${PYTHON_TARGETS} ${R_TARGETS}


## Order matters, do not change! Not all files need to be listed
## because most are included by others during the amalgamation.
NUMBBO_C = \
	src/numbbo_benchmark.c \
	src/numbbo_random.c \
	src/numbbo_generics.c

NUMBBO_H = src/numbbo.h

.PHONEY: clean all release r_release c_release python_release
.SILENT:

all: ${TARGETS}

clean:
	echo "  RM    ${TARGETS}"
	rm -f ${TARGETS}
	echo "  RM    build/python/dist"
	rm -f python_build.log
	echo "  RM    build/python/numbbo.egg-info"
	rm -fR build/python/numbbo.egg-info
	echo "  RM    build/python/MANIFEST"
	rm -f build/python/MANIFEST
	echo "  RM    build/r/pkg"
	rm -fR build/r/pkg
	rm -f build/r/roxygen.log
	rm -f r-build.log

release: c_release python_release r_release

r_release: ${R_TARGETS}
########################################################################
## C framework
build/c/numbbo.c: ${NUMBBO_C} src/numbbo_c_runtime.c
	echo "  AM    $@"
	${AMALGAMATE} $+ > $@

build/c/numbbo.h: ${NUMBBO_H}
	echo "  AM    $@"
	${AMALGAMATE} $+ > $@

build/c/VERSION:
	echo "  MK    $@"
	echo "${C_VERSION}" > $@

release/c/numbbo-${C_VERSION}.tar.gz: ${C_TARGETS}
	echo "  TAR   $@"
	${GTAR} ${GTAR_CREATE_FLAGS} -czf $@ --transform='s,build/c,numbbo,' $+

c_release: release/c/numbbo-${C_VERSION}.tar.gz

########################################################################
## Python framework
build/python/numbbo/numbbo.c: ${NUMBBO_C} src/numbbo_c_runtime.c
	echo "  AM    $@"
	${AMALGAMATE} $+ > $@

build/python/numbbo/numbbo.h: ${NUMBBO_H}
	echo "  AM    $@"
	${AMALGAMATE} $+ > $@

build/python/%: build/python/%.in
	echo "  M4    $@"
	${M4} -D__NUMBBO_VERSION__=${PYTHON_VERSION} $+ > $@

release/python/numbbo-${PYTHON_VERSION}.tar.gz: ${PYTHON_TARGETS}
	echo "  PY    $@"
	cd build/python \
	${PYTHON2} setup.py sdist --dist-dir=${CURDIR}/release/python

python_release: release/python/numbbo-${PYTHON_VERSION}.tar.gz

########################################################################
## R framework
build/r/skel/src/numbbo.c: ${NUMBBO_C} src/numbbo_r_runtime.c
	echo "  AM    $@"
	${AMALGAMATE} $+ > $@

build/r/skel/src/numbbo.h: ${NUMBBO_H}
	echo "  AM    $@"
	${AMALGAMATE} $+ > $@

build/r/%: build/r/%.in
	echo "  M4    $@"
	${M4} -D__NUMBBO_VERSION__=${R_VERSION} $+ > $@

release/r/numbbo_${R_VERSION}.tar.gz: ${R_TARGETS}
	echo "  ROXY  build/r/pkg"
	cd build/r ; ${RSCRIPT} ./tools/roxygenize > roxygen.log 2>&1
	echo "  R     $@"
	cd release/r/ ; \
	${R} CMD build ${CURDIR}/build/r/pkg > ${CURDIR}/r-build.log 2>&1

r_release: release/r/numbbo_${R_VERSION}.tar.gz
