CFLAGS ?= -std=c99 -O2 -g -march=native
CPPFLAGS := -D_POSIX_C_SOURCE=200809
OBJS := run_bbob.o coco.o cmaes.o
LIBS := -lm

.PHONEY: all run clean

all: run_bbob

run: run_bbob
	./run_bbob

clean:
	rm -fR run_bbob run_bbob.o coco.o cmaes.o c-cmaes* actparcmaes.par 

run_bbob: ${OBJS}
	${CC} ${LDFLAGS} ${OBJS} ${LIBS} -o run_bbob

run_bbob.o: run_bbob.c coco.h cmaes_interface.h 

coco.o: coco.c coco.h

cmaes.o: cmaes.c cmaes_interface.h cmaes.h
