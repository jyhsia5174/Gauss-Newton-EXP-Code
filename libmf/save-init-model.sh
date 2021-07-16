#!/bin/bash

make -f Makefile-save-model clean all
./mf-train -l2 1 -f 0 -k 40 -t 1 -r 0.1 -p va tr
make clean all
