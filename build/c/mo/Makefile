## Makefile to build C demo_mo programs included with the COCO distribution
##
## NOTE: We have tried to make this Makefile as generic and portable
## as possible. However, there are many (incompatible) versions of
## make floating around. We regularly test using GNU make and BSD make
## from FreeBSD. If you have trouble compiling the examples, please
## try to use GNU make. 
##
## On Windows it is best to use either the included NMakefile by running
##
##   nmake -f NMakefile
##
## or installing Cygwin and running GNU make from within Cygwin.

LDFLAGS += -lm
CCFLAGS = -g -ggdb -std=c89 -pedantic -Wall -Wextra -Wstrict-prototypes -Wshadow -Wno-sign-compare -Wconversion

########################################################################
## Toplevel targets
all: demo_mo

clean:
	rm -f coco.o demo_mo.o demo_mo

########################################################################
## Programs
demo_mo: demo_mo.o coco.o
	${CC} ${CCFLAGS} -o demo_mo coco.o demo_mo.o ${LDFLAGS}

########################################################################
## Additional dependencies
coco.o: coco.h coco.c
	${CC} -c ${CCFLAGS} -o coco.o coco.c
demo_mo.o: coco.h coco.c demo_mo.c
	${CC} -c ${CCFLAGS} -o demo_mo.o demo_mo.c
