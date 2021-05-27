
sources = cost_terms_main.c
executable = cost_terms
libfile = libcost_terms.dylib

CFLAGS = -Wall -O3 -ffast-math -march=native

all: $(executable) $(libfile)
#all: $(libfile)
$(executable): cost_terms.o

$(libfile): CFLAGS += -fPIC
$(libfile): $(sources) cost_terms.h
	gcc $(CFLAGS) -shared -o $@ -fopenmp $<


clean:
	rm -f cost_terms cost_terms.o
