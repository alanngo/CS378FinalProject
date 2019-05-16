#!/bin/bash
export LD_LIBRARY_PATH="/u/alsritt:$LD_LIBRARY_PATH"
python /v/filer5b/v38q001/alsritt/comparch/CS378FinalProject/Prefetcher.py /scratch/cluster/akanksha/dnn_ordered_traces2/astar /v/filer5b/v38q001/alsritt/comparch/CS378FinalProject/output/Prefetcher/astar/tsne > /v/filer5b/v38q001/alsritt/comparch/CS378FinalProject/output/Prefetcher/astar.txt
