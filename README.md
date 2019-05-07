# CS378FinalProject

t-SNE tutorial: https://www.easy-tensorflow.com/tf-tutorials/tensorboard/tb-embedding-visualization

k-means Clustering example: https://www.tensorflow.org/api_docs/python/tf/contrib/factorization/KMeansClustering, https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/kmeans.py

Link to Noob's guide: http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/

Link to Matthew's tutorial: https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/?spm=5176.100239.0.0.zqcyz5

The trace files containing (PC, address) are in /scratch/cluster/akanksha/dnn_ordered_traces2 (which is only accessible via the cluster).


--
Trace file Formatting:

The format of the traces is as follows:

<\ignore> <PC> <data address> <ignore> <hit/miss with OPT> <accurate/inaccurate with Hawkeye>

More concretely, each line in the trace is a cache access, and the fields represent the following:

1. Bool to indicate whether feedback has been received. You can ignore this field.
2. The memory address of the cache access
3. The load instruction address (PC) of the cache access
4. The probability of a cache hit as measured by my simple PC-based predictor (a value of 2 indicates that the predictor has not been trained).
5. The OPT decision to cache or not cache
6. This bit indicates whether Hawkeye's simple predictor got it right or wrong.

--
