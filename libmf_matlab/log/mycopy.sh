#! /bin/bash

for i in sol_*
do
    echo ${i}
    mv ${i}{,.tmp}
    cp ${i}{.tmp,}
done
