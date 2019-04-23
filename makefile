cluster:
	./run_cluster.sh Prefetcher.py 

tasks:
	/lusr/opt/condor/bin/condor_q

debug:
	./run_debug.sh Prefetcher.py