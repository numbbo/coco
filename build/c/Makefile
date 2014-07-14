CFLAGS ?= -Wall -Wextra -std=c89 -pedantic -Os -g
CXXFLAGS ?= -Wall -pedantic -Os -g
LDFLAGS ?= -lm
TARGETS = demo demo-cpp test_bbob2009

.PHONEY: clean all
.SILENT:

########################################################################
## Toplevel targets
all: ${TARGETS}

clean:
	echo "RM    ${TARGETS} *.o"
	rm -fR ${TARGETS} *.o

########################################################################
## Programs

demo: demo.o coco.o
	echo "LD    $@"
	${CC} ${LDFLAGS} -o $@ $+

demo-cpp.o: demo.cpp
	echo "CXX   $@"
	${CXX} ${CXXFLAGS} -o $@ -c $<

demo-cpp: demo-cpp.o coco.o
	echo "LD    $@"
	${CXX} ${LDFLAGS} -o $@ $+

test_bbob2009: test_bbob2009.o coco.o
	echo "LD    $@"
	${CC} ${LDFLAGS} -o $@ $+

test_fun_consistancy: test_fun_consistancy.o coco.o
	echo "LD    $@"
	${CC} ${LDFLAGS} -o $@ $+

########################################################################
## Additional dependencies

demo.o: coco.h

test_bbob2009.o: coco.h test_bbob2009.h

########################################################################
## Generic rules
%.o: %.c
	echo "CC    $@"
	${CC} ${CFLAGS} -o $@ -c $<

%.o: %.cpp
	echo "CXX   $@"
	${CXX} ${CXXFLAGS} -o $@ -c $<
