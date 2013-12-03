PYTHON2 ?= python2.7

.PHONEY: clean test package

#.SILENT:

package: numbbo/wrapper.pyx numbbo/numbbo.c numbbo/numbbo.h
	${PYTHON2} setup.py sdist1
	${PYTHON2} setup.py bdist

clean:
	${PYTHON2} setup.py clean -q
	rm -f numbbo/wrapper.c
	rm -f numbbo.so
	rm -fR random_search
	rm -fR build
	rm -fR dist

test:
	rm -fR random_search
	${PYTHON2} demo.py
