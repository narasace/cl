################################################################
# $Id$
# Makefile for OpenCL project
################################################################

### C compliler to use
CC = gcc
CFLAGS = -O3
LDFLAGS = 

#CC = icc
#CFLAGS = -O3 -ip -axT
#LDFLAGS = -static-intel

### OpenCL libraries
ifeq ($(shell uname),Darwin)
    CLINC =
    CLLIB = -framework OpenCL
else
    CLINC =
    CLLIB = -l OpenCL
endif


### Targets
all: default

default: a.out

a.out: main.c kernels.c
	$(CC) $(CFLAGS) $(CLINC) $^ -o $@ $(CLLIB) $(LDFLAGS)

kernels.c: kernels.cl bin2c.py
	./bin2c.py $< kernels > $@

clean:
	rm -f *.bak gmon.out core
