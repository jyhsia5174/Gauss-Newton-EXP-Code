#! /bin/bash

for i in sol_*
do
    mv -f ${i}{,.tmp}
    cp -f ${i}{.tmp,}
done
