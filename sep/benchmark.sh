#!/bin/zsh

for i in $(seq 1 10)
do
    ipython benchmark.py stances/figure2-single.json 1>>logs/single_bare.log
    ipython benchmark.py stances/figure2-double.json 1>>logs/double_bare.log
    ipython benchmark.py stances/figure2-triple.json 1>>logs/triple_bare.log
done

cat logs/single_bare.log | grep "^%timeit.*10" > logs/single.log
cat logs/double_bare.log | grep "^%timeit.*10" > logs/double.log
cat logs/triple_bare.log | grep "^%timeit.*10" > logs/triple.log
