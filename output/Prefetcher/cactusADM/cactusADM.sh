#!/bin/bash
export LD_LIBRARY_PATH="/u/alsritt:$LD_LIBRARY_PATH"
python /u/alsritt/comparch/CS378FinalProject/Prefetcher.py /scratch/cluster/akanksha/dnn_ordered_traces2/cactusADM /u/alsritt/comparch/CS378FinalProject/output/Prefetcher/cactusADM/tsne > /u/alsritt/comparch/CS378FinalProject/output/Prefetcher/cactusADM.txt
