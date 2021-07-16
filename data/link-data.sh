#! /bin/bash

# Parse arg
while getopts 'h' o; do
    case ${o} in
        h | *) 
          echo './link-data.sh data-dir' >&2
          exit 1
    esac
done

dir=`readlink -f $1`
ln -sf ${dir}/ml.te va
ln -sf ${dir}/ml.tr tr
