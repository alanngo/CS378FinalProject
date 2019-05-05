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
i = 0


for line in trace_file:
	ig1, PC, data_address, ig2, ig3, ig4 = line.split(" ")
	print(PC)
	PCs.append(float.fromhex(PC[-6:]))
	addr.append(float.fromhex(data_address[-6:]))
	#print(float.fromhex(data_address))
	if (len(deltas) == 0):
		deltas.append(0)
	else :
		deltas.append(int(addr[-1]-addr[-2]))
DEBUG = True
DB_ONE_HOT = False

#Remove all deltas with less than 10
# count_deltas = Counter(deltas)

# #iterate through [deltas], if entry in the count_deltas dictonary has less 

# a dictonary of Counter(deltas);
# Most common 50k deltas (MCDA)
# iterate through all deltas
#	if the delta occurs more than 10 times according to the dictonary
#		iterate through the MCDA, 	
#			if delta[i] = MCDA[x], put a 1, otherwise put a 0
#			add pc, delta to a new array for PC encoding.
#


delta_counter = Counter(deltas)
#copy the PC code from below, execpt with deltas
#make CURRENT delta code into change variable names
delta_dictonary = dict(delta_counter) #formatting





#output
most_common_deltas = delta_counter.most_common(49999)
most_common_deltas_array = [k for k , _ in most_common_deltas] #formatting

outputs_one_hot = []
pc_delta = []
deltas_seq =[]
pc_seq = []

print(delta_counter)
for d in range(len(deltas)):
	if delta_dictonary[deltas[d]] >= 1:
		temp = len(most_common_deltas_array)
		for mcda_pos in range(len(most_common_deltas_array)): # add a truth variable that is set to false when a bit is found
			if deltas[d] == most_common_deltas_array[mcda_pos] :
				temp = mcda_pos
			#if len(deltas_one_hot) over our threshold of sequence (64)
				#add the last 64 to a different array
		#right before we append temp, chek the truth value. 
		#Add 1 if still true (Signfies an outside value)
		#add 0 if false
		outputs_one_hot.append(temp)
		pc_delta.append((PCs[d], deltas[d]))

#print(deltas_seq)



used_pc = [k for k , _ in pc_delta] #formatting
pc_counter = Counter(used_pc)
pc_dictonary = dict(pc_counter) #formatting
pc_array = [k for k in pc_dictonary.keys()] #formatting

pc_one_hot = []


for pc , _ in pc_delta:
		temp = len(pc_array)
		for pc_pos in range(len(pc_array)):
			if pc == pc_array[pc_pos] :
				temp = pc_pos
		pc_one_hot.append(temp)
		if len(pc_one_hot) > 64:
			pc_seq.append(pc_one_hot[-1-64:-1])

used_d = [k for _ , k in pc_delta] #formatting
d_counter = Counter(used_d)
d_dictonary = dict(d_counter) #formatting
d_array = [k for k in d_dictonary.keys()] #formatting
d_one_hot = []

for _ , delt in pc_delta:
		temp = len(d_array)
		for d_pos in range(len(d_array)):
			if d == d_array[d_pos] :
				temp = d_pos
		d_one_hot.append(temp)
		if len(d_one_hot) > 64:
			deltas_seq.append(d_one_hot[-1-64:-1])


print(pc_seq)


if(DEBUG and DB_ONE_HOT):
	print(pc_array)
	for i in pc_one_hot:
		print(i)
	print(most_common_deltas_array)
	for i in deltas_one_hot:
		print(i)
	print(len(pc_one_hot) ==len(deltas_one_hot))

print(len(d_one_hot))
print(len(pc_one_hot))









delta_test, delta_train = np.array_split(d_one_hot, indices_or_sections = 2)
pc_test, pc_train = np.array_split(pc_one_hot, indices_or_sections = 2)
y_test, y_train = np.array_split(outputs_one_hot, indices_or_sections = 2)




##TODO##
# DONE-change the 'for d in deltas' to an index
# DONE-after appending temp, create an entry in pc_delta with deltas[d] and pc[d]
# -figure out encoding
# -figure out appending
# -figure out LSTM machine

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
n_input=len(d_one_hot)				##?
#learning rate for adam
learning_rate=0.001
#mnist is meant to be classified in 10 classes(0-9).
n_classes=50000
#size of batch
batch_size=128

#weights and biases of appropriate shape to accomplish above task
out_weights=tf.Variable(tf.random_normal([num_units,n_classes]))
out_bias=tf.Variable(tf.random_normal([n_classes]))

#batch size, time_steps
np_delta = tf.placeholder(tf.int32,[batch_size,time_steps])
delta_embeddings = tf.Variable(tf.random_normal([num_units,n_classes]))
embedded_deltas = tf.nn.embedding_lookup(delta_embeddings, np_delta)

np_pcs = tf.placeholder(tf.int32,[batch_size,time_steps])
pc_embeddings = tf.Variable(tf.random_normal([num_units,n_classes]))
embedded_pcs = tf.nn.embedding_lookup(pc_embeddings, np_pcs)

embedded_concat = tf.concat([embedded_pcs, embedded_deltas], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]



y = tf.placeholder("float",[None,n_classes])

print("done did delats")



#defining placeholders
#input image placeholder
#x=tf.placeholder("float",[None,time_steps,n_input])
#input label placeholder
#y=tf.placeholder("float",[None,n_classes])

#processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
input=tf.unstack(embedded_concat,time_steps,1)

#defining the network
lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
outputs,_=rnn.static_rnn(lstm_layer,embedded_concat,dtype="float32")

#converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
prediction=tf.matmul(outputs[-1],out_weights)+out_bias

#loss_function
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
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
        batch_x =  embedded_concat
        batch_y = delta_test    #change variable

        batch_x=batch_x.reshape((batch_size,time_steps,n_input))

        sess.run(opt, feed_dict={x: batch_x, y: batch_y})

        if iter %10==0:
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            los=sess.run(loss,feed_dict={x:batch_x,y:batch_y})
            print("For iter ",iter)
            print("Accuracy ",acc)
            print("Loss ",los)
            print("__________________")

        iter=iter+1

    #calculating test accuracy 
    test_data = mnist.test.images[:128].reshape((-1, time_steps, n_input)) #change variable
    test_label = mnist.test.labels[:128] #change variable
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
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
