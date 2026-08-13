#!/usr/bin/env sh



for (( d=9 ; d<=4*108; d+=9 ))
do
    echo -n $d "  "
    ./Stream 100000000 $d 1024 | grep CUDA
done
