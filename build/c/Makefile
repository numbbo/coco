CFLAGS ?= -Wall -std=c99 -pedantic -Os -g
LDFLAGS ?= -lm
TARGETS = demo

.PHONEY: clean all
.SILENT:

all: ${TARGETS}

demo: demo.o numbbo.o
	echo "LD    $@"
	${CC} ${LDFLAGS} -o $@ $+

%.o: %.c
	echo "CC    $@"
	${CC} ${CFLAGS} -o $@ -c $<

clean:
	echo "RM    ${TARGETS} *.o"
	rm -fR ${TARGETS} *.o random_search
