## Makefile to build C demo programs included with the COCO distribution
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
CCFLAGS = -g -ggdb -std=c89 -Wall -pedantic -Wextra -Wstrict-prototypes -Wshadow -Wno-sign-compare

########################################################################
## Toplevel targets
all: demo coco_test bbob2009_test 

clean:
	rm -f bbob2009_test.o bbob2009_test coco.o coco_test.o coco_test demo.o demo

########################################################################
## Programs
demo: demo.o coco.o
	${CC} ${CCFLAGS} -o demo coco.o demo.o ${LDFLAGS}  

coco_test: coco_test.o coco.o
	${CC}  ${CCFLAGS} -o coco_test coco_test.o coco.o ${LDFLAGS}

bbob2009_test: bbob2009_test.o coco.o
	${CC}  ${CCFLAGS} -o bbob2009_test bbob2009_test.o coco.o ${LDFLAGS} 

########################################################################
## Additional dependencies
demo.o: coco.h coco.c demo.c
coco_test.o: coco.h coco.c coco_test.c
bbob2009_test.o: coco.h coco.c bbob2009_test.c
