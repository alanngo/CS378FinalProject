#!/bin/bash
export LD_LIBRARY_PATH="/u/alsritt:$LD_LIBRARY_PATH"
python /v/filer5b/v38q001/alsritt/comparch/CS378FinalProject/Prefetcher.py /scratch/cluster/akanksha/dnn_ordered_traces2/soplex.med /v/filer5b/v38q001/alsritt/comparch/CS378FinalProject/output/Prefetcher/soplex.med/tsne > /v/filer5b/v38q001/alsritt/comparch/CS378FinalProject/output/Prefetcher/soplex.med.txt
