
import numpy as np
import tensorflow as tf
import h5py

width = 9
height = 9
depth = 200
nLabel = 9

x = tf.placeholder(tf.float32, shape=[None, width*height*depth]) # [None, 9*9*200]
y_ = tf.placeholder(tf.float32, shape=[None, nLabel])  # [None, 9]

## Convolution and Pooling
# Convolution here: stride=1, zero-padded -> output size = input size
def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME') # conv3d, [1, 1, 1, 1]
# Pooling: max pooling 
def max_pool_2x2(x):
  return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 4, 1], strides=[1, 2, 2, 4, 1], padding='SAME')
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


num=16
num1=2*num
num2=2*num1

x_image = tf.reshape(x, [-1,width,height,depth,1])
## Res1
W_conv1_1 = weight_variable([3, 3, 3, 1, num])  
b_conv1_1 = bias_variable([num])

W_conv1_2 = weight_variable([3, 3, 3, num, num])  
b_conv1_2 = bias_variable([num])

W_conv1_3 = weight_variable([3, 3, 3, num, num])  
b_conv1_3 = bias_variable([num])

W_conv1 = weight_variable([3, 3, 3, num, num])  
b_conv1 = bias_variable([num])


h_conv1_1 = tf.nn.relu(conv3d(x_image, W_conv1_1) + b_conv1_1)
h_conv1_2 = tf.nn.relu(conv3d(h_conv1_1, W_conv1_2) + b_conv1_2)
h_conv1_3 = tf.nn.relu(conv3d(h_conv1_2, W_conv1_3) + b_conv1_3)

h_conv1 = tf.nn.relu(conv3d(h_conv1_3, W_conv1) + b_conv1) + h_conv1_1

#dimension reduction
h_pool1 = max_pool_2x2(h_conv1)#5*5*26

#res2
W_conv2_1 = weight_variable([3, 3, 3, num, num1]) 
b_conv2_1 = bias_variable([num1])

W_conv2_2 = weight_variable([3, 3, 3, num1, num1]) 
b_conv2_2 = bias_variable([num1])

W_conv2_3 = weight_variable([3, 3, 3, num1, num1]) 
b_conv2_3 = bias_variable([num1])

W_conv2 = weight_variable([3, 3, 3, num1, num1]) 
b_conv2 = bias_variable([num1])

h_conv2_1 = tf.nn.relu(conv3d(h_pool1, W_conv2_1) + b_conv2_1)
h_conv2_2 = tf.nn.relu(conv3d(h_conv2_1, W_conv2_2) + b_conv2_2)
h_conv2_3 = tf.nn.relu(conv3d(h_conv2_2, W_conv2_3) + b_conv2_3)
h_conv2 = tf.nn.relu(conv3d(h_conv2_3, W_conv2) + b_conv2) + h_conv2_1

#dimension reduction
h_pool2 = max_pool_2x2(h_conv2)#3*3*7


#res3
W_conv3 = weight_variable([3, 3, 3, num1, num2]) 
b_conv3 = bias_variable([num2])

h_conv3 = tf.nn.conv3d(h_pool2, W_conv3, strides=[1, 1, 1, 1, 1], padding='VALID') + b_conv3


## Densely Connected Layer (or fully-connected layer)
# fully-connected layer with 1024 neurons to process on the entire image
W_fc1 = weight_variable([11*num2, 128])  # [7*7*64, 1024]
b_fc1 = bias_variable([128]) # [1024]]

h_flat = tf.reshape(h_conv3, [-1, 11*num2])  # -> output image: [-1, 7*7*64] = 3136
h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)  # ReLU(h_pool2_flat x weight + bias)

## Dropout (to reduce overfitting; useful when training very large neural network)
# We will turn on dropout during training & turn off during testing
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
print(h_fc1_drop.get_shape)  # -> output: 1024

## Readout Layer
W_fc2 = weight_variable([128, nLabel]) # [1024, 10]
b_fc2 = bias_variable([nLabel]) # [10]

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
print(y_conv.get_shape)  # -> output: 10

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess,'./model-200/model.ckpt')

f=h5py.File('./I9-9.h5','r')#./Indian/
test_samples=f['data'][:]
test_labels=f['label'][:]
f.close()

def get_accuracy():
    result=[]
    for i in range(92):
        batch_x=test_samples[i*100:100*(i+1),:]
        batch_y=test_labels[i*100:100*(i+1),:]
        result1=sess.run(correct_prediction,feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
        result.append(result1)
    batch_x=test_samples[92*100:,:]
    batch_y=test_labels[92*100:,:]
    result1=sess.run(correct_prediction,feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
    result=np.float32(result)
    result=result.reshape(-1)
    result1=np.float32(result1)
    result=np.hstack((result,result1))
    return result.mean()

ac=get_accuracy()

file_object = open('./ac.txt', 'wb')
file_object.write(str(ac))
file_object.close( )
