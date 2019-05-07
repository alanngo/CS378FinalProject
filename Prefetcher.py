import numpy as np
import random
from random import shuffle
import tensorflow as tf
import sys
from collections import Counter
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# read file
print("This is the name of the script: ", sys.argv[0])
print("Number of arguments: ", len(sys.argv))
print("The arguments are: " , str(sys.argv))
trace_file = open(sys.argv[1], 'r')

PCs = []
deltas = []
addr = []
next_delta = []
i = 0


for line in trace_file:
	ig1, PC, data_address, ig2, ig3, ig4 = line.split(" ")
	#print(PC)
	PCs.append(float.fromhex(PC[-6:]))
	addr.append(float.fromhex(data_address[-6:]))
	#print(float.fromhex(data_address))
	if (len(deltas) == 0):
		deltas.append(0)
	else :
		deltas.append(int(addr[-1]-addr[-2]))
DEBUG = True
DB_ONE_HOT = False

next_delta = deltas.copy()
next_delta.insert(0, 0)




next_delta_one_hot = []
delta_one_hot = []
pc_one_hot = []
delta_seq = []
pc_seq = []
next_delta_format = []

##DELTA INPUT VOCAB##
#getting all unique values and their counts.
#formatting to make it a dictonary for ease of manip.
delta_frequency_counter = Counter(deltas) 				#used for Next Delta
delta_frequency_dictonary = dict(Counter(deltas)) 

#Reducing the input vocab to only those with 10 or more occurances. Enumerates them for easy one-hot encoding.
deltas_oh_encode = dict()
for counter,value in enumerate(i for i in delta_frequency_dictonary.keys() if delta_frequency_dictonary[i] >= 2):
	deltas_oh_encode[value] = counter

##Delta POH list.
for i in deltas:
	if i in deltas_oh_encode:
		delta_one_hot.append(deltas_oh_encode[i])
	else:
		delta_one_hot.append(len(deltas_oh_encode))

##PC INPUT VOCAB##
pc_frequency_dictonary = dict(Counter(PCs)) 
pcs_oh_encode = dict()
for counter,value in enumerate(i for i in pc_frequency_dictonary.keys()):
	pcs_oh_encode[value] = counter


##PC POH list.
for i in PCs:
	if i in pcs_oh_encode:
		pc_one_hot.append(pcs_oh_encode[i])
	else:
		pc_one_hot.append(len(pcs_oh_encode))

##NEXT_DELTA OUTPUT VOCAB
next_delta_frequency_dictonary = dict(delta_frequency_counter.most_common(49999)) #Creating an array of the top 50k deltas, will make it smaller if avliable.
next_delta_oh_encode = dict()
for counter,value in enumerate(i for i in next_delta_frequency_dictonary.keys()):
	next_delta_oh_encode[value] = counter

##next_delta POH encoding
for i in next_delta:
	if i in next_delta_oh_encode:
		next_delta_one_hot.append(next_delta_oh_encode[i])
	else:
		next_delta_one_hot.append(len(next_delta_oh_encode))

assert(len(pc_one_hot) == len(delta_one_hot))




##sequencing
for i in range(len(pc_one_hot)):
	if i > 64:
		pc_seq.append(pc_one_hot[i-64:i])
		delta_seq.append(delta_one_hot[i-64:i])
		temp = []
		temp.append(next_delta_one_hot[i+1])
		next_delta_format.append(temp)

delta_train, delta_test = np.array_split(delta_seq, indices_or_sections = 2)
pc_train, pc_test = np.array_split(pc_seq, indices_or_sections = 2)
y_train, y_test = np.array_split(next_delta_format, indices_or_sections = 2)

print(delta_train)
print(pc_train)
print(y_train)



if(DEBUG and DB_ONE_HOT):
	print(pc_array)
	for i in pc_one_hot:
		print(i)
	print(most_common_deltas_array)
	for i in deltas_one_hot:
		print(i)
	print(len(pc_one_hot) ==len(deltas_one_hot))

###Training hyperparameters for each model.###
#EMBEDDING 					
#Network Size 				128x2 LSTM
#Learning Rate 				.001
#Number of Train Steps 		500k
#Sequence Length 			64	
#Embedding size 			128

#CLUSTERING
#Network Size 				128x2 LSTM	
#Learning Rate 				.1
#Number of Train Steps 		250k
#Sequence Length 			64
#Number of Centroids		12

##EMBEDDING##

# t1 = [[1, 2, 3], [4, 5, 6]]
# t2 = [[7, 8, 9], [10, 11, 12]]
# tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

# # '''
# # if(DEBUG):
# # 	print(inputs)
# # 	print(count_out_arr)
# # '''
# # print(mnist)
#define constants
#unrolled through 64 time steps
time_steps=64			#how many previous addresses
#hidden LSTM units
num_units=128
#rows of 28 pixels
n_input= 64				##?
#learning rate for adam
learning_rate=0.001
#mnist is meant to be classified in 10 classes(0-9).
n_classes=50000
#size of batch
batch_size=64
#defined in the paper
embedding_size = 128
#vocab sizes
pc_vocab_size = len(pcs_oh_encode)+1
delta_vocab_size = len(deltas_oh_encode)+1


#weights and biases of appropriate shape to accomplish above task
out_weights=tf.Variable(tf.random_normal([num_units,n_classes]))
out_bias=tf.Variable(tf.random_normal([n_classes]))

#batch size, time_steps
np_delta = tf.placeholder(tf.int32,[batch_size,time_steps])
delta_embeddings = tf.Variable(tf.random_normal([delta_vocab_size,embedding_size]))
embedded_deltas = tf.nn.embedding_lookup(delta_embeddings, np_delta)

np_pcs = tf.placeholder(tf.int32,[batch_size,time_steps])
pc_embeddings = tf.Variable(tf.random_normal([pc_vocab_size,embedding_size]))
embedded_pcs = tf.nn.embedding_lookup(pc_embeddings, np_pcs)



embedded_concat = tf.concat([embedded_pcs, embedded_deltas], 2)  # uhhhh.... still not sure here

print(embedded_concat.get_shape()) # return (64, 128, 50000)

print("y values:")
y = tf.placeholder(tf.int32,[batch_size,1])
y_oh = tf.one_hot(y, n_classes)
y_val = tf.reshape( y_oh, (-1, n_classes))
print(y.get_shape()) # returns (64, 50000)

print("!!!!!!!!!done did delats!!!!!!!!!!!")

#print shape of the tensor: tf.shape()
#embedding: take a large feature, reduce it's dimensionality.
#	-creates a mapping to a smaller vector
#Predicts a one-hot

#processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
input=tf.unstack(embedded_concat,time_steps,1)



#defining the network
lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1) 	#really slow, can use gpus
outputs,_=rnn.static_rnn(lstm_layer,input,dtype="float32")

#converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
prediction=tf.matmul(outputs[-1],out_weights)+out_bias

print("***prediction***")
print(prediction.get_shape())

#loss_function
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y_val))
#optimization
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#model evaluation
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#initialize variables
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    iter=1
    while iter<2:
        batch_delta = delta_train[(iter-1)*64:iter*64]
        batch_pc = pc_train[(iter-1)*64:iter*64]
        batch_next_delta = y_train[(iter-1)*64:iter*64]

        #batch_x=batch_x.reshape((batch_size,time_steps,n_input))
        fd = {np_delta:batch_delta, np_pcs:batch_pc, y:batch_next_delta}
        sess.run(opt, feed_dict=fd)

        if iter%1 == 0:
            acc=sess.run(accuracy,feed_dict=fd)
            los=sess.run(loss,feed_dict=fd)
            print("For iter ",iter)
            print("Accuracy ",acc)
            print("Loss ",los)
            print("__________________")

        iter=iter+1

    # #calculating test accuracy 
    # test_data = mnist.test.images[:128].reshape((-1, time_steps, n_input)) #change variable
    # test_label = mnist.test.labels[:128] #change variable
    # print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
'''
NUM_EXAMPLES = len(count_out_arr) / 2
get test_input and test_output
algorithm 0...NUM_EXAMPLES
get train_input and train_output
algorithm NUM_EXAMPLES...len(count_out_arr) / 2

algorithm: 
iterate through each entry
first sub-entry of entry goes into input
second sub-entry of entry goes into output
'''
