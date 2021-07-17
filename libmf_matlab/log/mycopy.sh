#! /bin/bash

for i in *_time_1
do
    mv ${i}{,.tmp}
    cp ${i}{.tmp,}
done
