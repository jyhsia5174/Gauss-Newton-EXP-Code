#! /bin/bash

for i in sol_*
do
    mv ${i}{,.tmp}
    cp ${i}{.tmp,}
done
