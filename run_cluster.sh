#!/bin/bash

# Check that we are logged into streetpizza

# Check that args are correct
if [ "$#" -ne 1 ]; then
    echo "Incorrect # of arguments: expected a python program"
    echo "Usage: ./run_cluster.sh name_of_program.py"
    exit
fi

PROGRAM=$1

# Make sure program exists
if test ! -f ${PROGRAM}; then
    echo "${PROGRAM} does not exist!"
    echo "Usage: ./run_cluster.sh name_of_program.py"
    exit
fi

PROGRAM=$(echo "${PROGRAM}" | cut -d'.' -f 1)
PY_DIR=$(pwd)

OUTPUT_DIR="${PY_DIR}/output/${PROGRAM}"

# Ensure output dir exists
if test ! -d ${OUTPUT_DIR}; then
    mkdir ${OUTPUT_DIR}
fi

USER=$(whoami)

while read -r line ;
do
    echo "line!"
    BENCHMARK=$line
    CONDOR_DIR="${OUTPUT_DIR}/${BENCHMARK}"
    SCRIPT_FILE="${CONDOR_DIR}/${BENCHMARK}.sh"
    CONDOR_FILE="${CONDOR_DIR}/${BENCHMARK}.condor"
    OUTPUT_FILE="${OUTPUT_DIR}/${BENCHMARK}.txt"
    TSNE_DIR="${CONDOR_DIR}/tsne"
    
    if test ! -d ${CONDOR_DIR}; then
        mkdir ${CONDOR_DIR}
    fi
        # Ensure tsne dir exists
    if test ! -d ${TSNE_DIR}; then
        mkdir ${TSNE_DIR}
    fi

    # create script file
    echo "#!/bin/bash" > $SCRIPT_FILE
    echo "export LD_LIBRARY_PATH=\"/u/${USER}:\$LD_LIBRARY_PATH\"" >> $SCRIPT_FILE
    echo "python $PY_DIR/${PROGRAM}.py /scratch/cluster/akanksha/dnn_ordered_traces2/${BENCHMARK} $TSNE_DIR > $OUTPUT_FILE" >> $SCRIPT_FILE
    chmod +x $SCRIPT_FILE
    
    # create condor file
    /u/alsritt/comparch/CS378FinalProject/condorize.sh true $CONDOR_DIR $BENCHMARK

    # submit the condor file
    /lusr/opt/condor/bin/condor_submit $CONDOR_FILE
done < traces.txt
