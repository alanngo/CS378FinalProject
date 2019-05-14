import os
import numpy as np
import random
from random import shuffle
import tensorflow as tf
import sys
from collections import Counter
from tensorflow.contrib import rnn
import graphing as gra
import matplotlib.pyplot as plt


#tsne?
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import preprocessing



#Tags:
#	DEBUG
#	TODO

output_dir = sys.argv[2]

# File Input
print("This is the name of the script: ", sys.argv[0])
print("Number of arguments: ", len(sys.argv))
print("The arguments are: " , str(sys.argv))
trace_file = open(sys.argv[1], 'r')

#Raw Input Arrays. May need to optimize for space?
PCs = []
deltas = []
addr = []
next_delta = []
i = 0

##TODO: Figure out where we want to declasre our main variables
time_steps=64			#how many previous addresses


for line in trace_file:
	ig1, PC, data_address, ig2, ig3, ig4 = line.split(" ")
	PCs.append(float.fromhex(PC[-6:]))
	addr.append(float.fromhex(data_address[-6:]))
	if (len(deltas) == 0):
		deltas.append(0)
#		next_delta.append(0)
	else :
		deltas.append(int(addr[-1]-addr[-2]))
		next_delta.append(deltas[-1])

next_delta.append(0)

print(deltas[:5])
print(next_delta[:5])


#O-H Variables
next_delta_one_hot = []
delta_one_hot = []
pc_one_hot = []
delta_seq = []
pc_seq = []
#	getting all unique values and their counts.
#	formatting to make it a dictonary for ease of manip.
delta_frequency_counter = Counter(deltas) 				#used for Next Delta
delta_frequency_dictonary = dict(Counter(deltas)) 


##DELTA INPUT VOCAB##
#	Reducing the input vocab to only those with 10 or more occurances. Enumerates them for easy one-hot encoding.
#	TODO: change to only allow occurances with more than 10. Currently less for testing.
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
#TODO: determine if i need to also use above 10 occuring; ex:
#address occurs less than 10 times, but is still in top 50k.
next_delta_frequency_dictonary = dict(delta_frequency_counter.most_common(49999)) #Creating an array of the top 50k deltas, will make it smaller if avliable.
next_delta_oh_encode = dict()
for counter,value in enumerate(i for i in next_delta_frequency_dictonary.keys()):
	next_delta_oh_encode[value] = counter

##next_delta POH encoding
for i in next_delta:
	temp = np.zeros(len(next_delta_oh_encode))
	if i in next_delta_oh_encode:
		temp[next_delta_oh_encode[i]] = 1
		next_delta_one_hot.append(temp)
	else:
		temp[next_delta_oh_encode[len(next_delta_oh_encode)]] = 1
		next_delta_one_hot.append(temp)

#debug assertion
assert(len(pc_one_hot) == len(delta_one_hot))


##sequencing
for i in range(len(pc_one_hot)):
	if i > time_steps:
		pc_seq.append(pc_one_hot[i-time_steps:i])
		delta_seq.append(delta_one_hot[i-time_steps:i])

delta_train, delta_test = np.array_split(delta_seq, indices_or_sections = 2)
pc_train, pc_test = np.array_split(pc_seq, indices_or_sections = 2)
y_train, y_test = np.array_split(next_delta_one_hot[time_steps:], indices_or_sections = 2)


#DEBUG
print(len(delta_train))
print(len(pc_train))
print(len(y_train))

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
#define constants
#unrolled through 64 time steps.
#AKA how many previous addresses we're feeding into the machine at any given point.
time_steps=64
#hidden LSTM units
#spooky
num_units=128
#learning rate for adam
learning_rate=0.001
#mnist is meant to be classified in 10 classes(0-9).
#our values are however many outputs we are given.
n_classes = len(next_delta_oh_encode)
#size of batch
#AKA how many lines we feed into the machine in any given iteration.
#mostly just for efficency's sake, we could do it one line at a time but that's
#a lot.
batch_size=64
#defined in the paper
embedding_size = 128
#vocab sizes
#AKA n_imput in other tutorials. Size of the vocabularly. Not sure if the +1 is necessary.
pc_vocab_size = len(pcs_oh_encode)+1
delta_vocab_size = len(deltas_oh_encode)+1
with tf.Graph().as_default():

	with tf.Session() as sess:

		#weights and biases of appropriate shape to accomplish above task
		#shape: LSTM units by classes. The learning weights and biasis of the actual machines and levels.
		out_weights=tf.Variable(tf.random_normal([num_units,n_classes]) ) #histogram
		tf.summary.histogram("weights", out_weights)
		out_bias=tf.Variable(tf.random_normal([n_classes]))
		tf.summary.histogram("bias", out_bias)

		#The embedding size. 
		#	ALEX: Pretty sure this is just what we've learned already, applies to our newest values.
		#Delta embadding
		np_delta = tf.placeholder(tf.int32,[batch_size,time_steps])
		#input format:
		# [
		# 	1.[1,2,3,...time_steps]
		# 	2.[1,2,3,...time_steps]
		# 	3.[1,2,3,...time_steps]
		# 	...
		# 	batch_size.[1,2,3,...time_steps]
		# ]
		delta_embeddings = tf.Variable(tf.random_normal([delta_vocab_size,embedding_size]))
		embedded_deltas = tf.nn.embedding_lookup(delta_embeddings, np_delta)

		#PC embedding
		np_pcs = tf.placeholder(tf.int32,[batch_size,time_steps])
		pc_embeddings = tf.Variable(tf.random_normal([pc_vocab_size,embedding_size]))
		embedded_pcs = tf.nn.embedding_lookup(pc_embeddings, np_pcs)

		#Concatenation
		#	uhhhh.... still not sure here. It works, but i'm not sure if it's the right axis.
		embedded_concat = tf.concat([embedded_pcs, embedded_deltas], 2) 

		#DEbUG: shaping
		print(embedded_concat.get_shape()) # return (64, 128, 50000)

		#pre one-hot encoded prediction deltas
		y = tf.placeholder(tf.int32,[batch_size,n_classes])

		#DEBUG:
		print(y.get_shape()) 

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
		tf.summary.histogram('Predictions', prediction)


		#DEBUG:predictions
		print("***prediction***")
		print(prediction.get_shape())

		#loss_function
		loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y)) #summary
		#optimization
		opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

		#model evaluation
		correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
		accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) #summary

		tf.summary.scalar('accuracy', accuracy)


		#summary: merge all
		#tf.summary.FileWriter

		#graph object?!?!??


		#initialize variables


		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(output_dir, sess.graph)
		test_writer = tf.summary.FileWriter(output_dir)

		init=tf.global_variables_initializer()

		vectors = []

		iterations = len(delta_train)/batch_size

		sess.run(init)
		iter=1
		while iter<iterations:
			batch_delta = delta_train[(iter-1)*batch_size:iter*batch_size]
			batch_pc = pc_train[(iter-1)*batch_size:iter*batch_size]
			batch_next_delta = y_train[(iter-1)*batch_size:iter*batch_size]

			#batch_x=batch_x.reshape((batch_size,time_steps,n_input))
			fd = {np_delta:batch_delta, np_pcs:batch_pc, y:batch_next_delta}
			summ, op_run = sess.run([merged, opt], feed_dict=fd)

			if iter%10 == 0:
				train_writer.add_summary(summ, iter)
				saver = tf.train.Saver([pc_embeddings])

				saver.save(sess, "/u/alsritt/comparch/CS378FinalProject/train/model.ckpt", iter)

				acc=sess.run(accuracy,feed_dict=fd)
				summ, los = sess.run([merged, loss],feed_dict=fd)
				print("For iter ",iter)
				print("Accuracy ",acc)
				print("Loss ",los)
				print("__________________")


			iter=iter+1
		# #calculating test accuracy 
		testing_acc = []
		iterations = len(delta_test)/batch_size
		print(iterations)
		iter=1
		while iter<iterations:
			print(iter)
			batch_delta = delta_test[(iter-1)*batch_size:iter*batch_size]
			batch_pc = pc_test[(iter-1)*batch_size:iter*batch_size]
			batch_next_delta = y_test[(iter-1)*batch_size:iter*batch_size]

			#batch_x=batch_x.reshape((batch_size,time_steps,n_input))
			fd = {np_delta:batch_delta, np_pcs:batch_pc, y:batch_next_delta}
			acc=sess.run(accuracy,feed_dict=fd)
			testing_acc.append(acc)
			iter=iter+1

		print("Testing Accuracy:", np.mean(testing_acc))
		config = projector.ProjectorConfig()
		# One can add multiple embeddings.
		embedding = config.embeddings.add()
		embedding.tensor_name = delta_embeddings.name
		# Saves a config file that TensorBoard will read during startup.
		projector.visualize_embeddings(train_writer, config)


# model = TSNE(n_components=2, random_state=0)
# vectors = model.fit_transform(vectors)
# normalizer = preprocessing.Normalizer()
# vectors =  normalizer.fit_transform(vectors, 'l2')
# fig, ax = plt.subplots()
# for out_delta in next_delta:
#     print(out_delta, vectors[next_delta_oh_encode[out_delta]][1])
#     ax.annotate(out_delta, (vectors[next_delta_oh_encode[out_delta]][0],vectors[next_delta_oh_encode[out_delta]][1] ))
# plt.show()

# coverage: benchmark vs coverage (percentage)
# speedup: benchmark vs speedup (decimal/real)
#		coverage_objects = ()
#		coverage_x_pos = tf.constant().eval()
#		coverage_y_pos = tf.constant().eval()
#		#100 x (Prefetch Hits/(Prefetch Hits + Cache Misses))

#		speedup_objects = ()
#		speedup_x_pos = tf.constant().eval()
#		speedup_y_pos = tf.constant().eval()
#
#		plt.subplot(2, 1, 1)
#		plt.bar(coverage_x_pos, coverage_y_pos, align = 'center', alpha = 0.5)
#		plt.title('Coverage')
#		plt.xticks(coverage_x_pos, coverage_objects)
#		plt.xlabel('Benchmarks')
#		plt.ylabel('Coverage (%)')
#
#		plt.subplot(2, 1, 2)
#		plt.bar(speedup_x_pos, speedup_y_pos, align = 'center', alpha = 0.5)
#		plt.title('Speedup')
#		plt.xticks(speedup_x_pos, speedup_objects)
#		plt.xlabel('Benchmarks')
#		plt.ylabel('Speedup (%)')

#		plt.show()