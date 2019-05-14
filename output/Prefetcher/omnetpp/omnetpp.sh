#!/bin/bash
export LD_LIBRARY_PATH="/u/alsritt:$LD_LIBRARY_PATH"
python /u/alsritt/comparch/CS378FinalProject/Prefetcher.py /scratch/cluster/akanksha/dnn_ordered_traces2/omnetpp /u/alsritt/comparch/CS378FinalProject/output/Prefetcher/omnetpp/tsne > /u/alsritt/comparch/CS378FinalProject/output/Prefetcher/omnetpp.txt
