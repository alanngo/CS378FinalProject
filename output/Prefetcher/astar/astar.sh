#!/bin/bash
export LD_LIBRARY_PATH="/u/alsritt:$LD_LIBRARY_PATH"
python /u/alsritt/comparch/CS378FinalProject/Prefetcher.py /scratch/cluster/akanksha/dnn_ordered_traces2/astar /u/alsritt/comparch/CS378FinalProject/output/Prefetcher/astar/tsne > /u/alsritt/comparch/CS378FinalProject/output/Prefetcher/astar.txt
