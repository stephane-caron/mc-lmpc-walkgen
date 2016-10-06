#!/bin/zsh

NB_BENCHMARKS=10

for i in $(seq 1 ${NB_BENCHMARKS})
do
    printf "Extending single-contact benchmark %d / %d...\n" i NB_BENCHMARKS
    ipython benchmark.py ../stances/single.json 1>>logs/single_bare.log
    printf "Extending double-contact benchmark %d / %d...\n" i NB_BENCHMARKS
    ipython benchmark.py ../stances/double.json 1>>logs/double_bare.log
    printf "Extending triple-contact benchmark %d / %d...\n" i NB_BENCHMARKS
    ipython benchmark.py ../stances/triple.json 1>>logs/triple_bare.log
done

cat logs/single_bare.log | grep "^%timeit.*10" > logs/single.log
cat logs/double_bare.log | grep "^%timeit.*10" > logs/double.log
cat logs/triple_bare.log | grep "^%timeit.*10" > logs/triple.log

echo ""
echo "SINGLE-CONTACT RESULTS"
echo "======================"

python parse.py logs/single.log

echo ""
echo "DOUBLE-CONTACT RESULTS"
echo "======================"

python parse.py logs/double.log

echo ""
echo "TRIPLE-CONTACT RESULTS"
echo "======================"

python parse.py logs/triple.log

echo ""
