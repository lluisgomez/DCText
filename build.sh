#!/bin/bash

g++ -O3 -march='core2' -fpermissive `pkg-config opencv --cflags` -c dctext.cpp -o dctext.o

libtool --tag=CXX --mode=link g++ -O3 -march='core2' -o dctext dctext.o `pkg-config opencv --libs` -lfftw3
