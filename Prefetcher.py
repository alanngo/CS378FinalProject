import numpy as np
import random
from random import shuffle
import tensorflow as tf
import sys
from collections import Counter
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# read file
print("This is the name of the script: ", sys.argv[0])
print("Number of arguments: ", len(sys.argv))
print("The arguments are: " , str(sys.argv))
trace_file = open(sys.argv[1], 'r')

inputs = []
outputs = []
i = 0


for line in trace_file:
	ig1, PC, data_address, ig2, ig3, ig4 = line.split(" ")
	nested_array = []
	nested_array.append(float.fromhex(PC))
	nested_2_array = []
	nested_2_array.append(nested_array)
	inputs.append(nested_2_array)
	if (len(outputs) == 0):
		outputs.append(0)
	else :
		outputs.append(float(inputs[-1][0][0]-inputs[-2][0][0]))
DEBUG = True


count_out = Counter(outputs)
count_out_arr = [(k , v) for k , v in count_out.items()]
count_out_val = count_out.values

while len(count_out_arr) > 50000:
	for i in range(len(count_out_arr)):
		if count_array_arr[i][1] == min(count_out.values()) and len(count_out_arr) > 50000 :
			del count_array_arr[i]
	count_out_val.remove(min(count_out_val))


for i in range(len(count_out_arr)):
	count_out_arr[i] = [(count_out_arr[i][0])]

'''
if(DEBUG):
	print(inputs)
	print(count_out_arr)
'''
print(mnist)
#define constants
#unrolled through 28 time steps
time_steps=28
#hidden LSTM units
num_units=128
#rows of 28 pixels
n_input=28
#learning rate for adam
learning_rate=0.001
#mnist is meant to be classified in 10 classes(0-9).
n_classes=10
#size of batch
batch_size=128

#weights and biases of appropriate shape to accomplish above task
out_weights=tf.Variable(tf.random_normal([num_units,n_classes]))
out_bias=tf.Variable(tf.random_normal([n_classes]))

#defining placeholders
#input image placeholder
x=tf.placeholder("float",[None,time_steps,n_input])
#input label placeholder
y=tf.placeholder("float",[None,n_classes])

#processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
input=tf.unstack(x ,time_steps,1)

#defining the network
lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
outputs,_=rnn.static_rnn(lstm_layer,input,dtype="float32")

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
    while iter<790:
        batch_x,batch_y=mnist.train.next_batch(batch_size=batch_size) #change variable

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
