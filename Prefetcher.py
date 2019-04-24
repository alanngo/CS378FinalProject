import numpy as np
import random
from random import shuffle
import tensorflow as tf
import sys
from collections import Counter

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

if(DEBUG):
	print(inputs)
	print(count_out_arr)


