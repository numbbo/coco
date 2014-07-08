CFLAGS ?= -Wall -std=c99 -pedantic -Os -g
CXXFLAGS ?= -Wall -pedantic -Os -g
LDFLAGS ?= -lm
TARGETS = demo cppdemo

.PHONEY: clean all
.SILENT:

all: ${TARGETS}

demo: demo.o numbbo.o
	echo "LD    $@"
	${CC} ${LDFLAGS} -o $@ $+

cppdemo: cppdemo.o numbbo.o
	echo "LD    $@"
	${CXX} ${LDFLAGS} -o $@ $+

%.o: %.c
	echo "CC    $@"
	${CC} ${CFLAGS} -o $@ -c $<

%.o: %.cpp
	echo "CXX   $@"
	${CXX} ${CXXFLAGS} -o $@ -c $<

clean:
	echo "RM    ${TARGETS} *.o"
	rm -fR ${TARGETS} *.o
