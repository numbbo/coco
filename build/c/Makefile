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

########################################################################
## Toplevel targets
all: demo test_bbob2009

clean:
	${RM} -f coco.o test_bbob2009.o test_bbob2009 demo.o demo

########################################################################
## Programs
demo: demo.o coco.o
	${CC} ${LDFLAGS} -o demo coco.o demo.o 

test_bbob2009: test_bbob2009.o coco.o
	${CC} ${LDFLAGS} -o test_bbob2009 test_bbob2009.o coco.o

########################################################################
## Additional dependencies
demo.o: coco.h

test_bbob2009.o: coco.h test_bbob2009.h
