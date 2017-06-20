import cv2
from scipy import misc


from keras.layers.core import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras import backend as k

'''
import tensorlayer as tl
'''
import tensorflow as tf

#image shape is 512,288,3
#convert shape is 64,36,1

def preprocessing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = misc.
    #按8倍缩放
    resize = misc.imresize(gray,size=[64,36])
    resize = resize/255.
    resize.astype('float32')
    resize = resize.reshape([1,64,36,1])
    return resize


def policy_network(input):
    network = Conv2D(nb_filter=32,nb_row=2,nb_col=2,activation='relu',name='conv1')(input)
    network = MaxPooling2D(pool_size=[2,2],name='pool1')(network)

    network = Conv2D(nb_filter=32,nb_row=2,nb_col=2,activation='relu',name='conv2')(network)
    network = MaxPooling2D(pool_size=[2, 2], name='pool2')(network)

    network = Flatten(name='flatten')(network)

    network = Dense(512,activation='relu',name='fc3')(network)
    network = Dense(1,activation='sigmoid',name='fc4')(network)

    return network


'''
def policy_network(input):
    network = tl.layers.InputLayer(input,name='input')

    network = tl.layers.Conv2d(network,32,filter_size=[2,2],act=tf.nn.relu,name='conv1')
    network = tl.layers.MaxPool2d(network,filter_size=(2,2),name='pool1')

    network = tl.layers.Conv2d(network,32, filter_size=[2, 2], act=tf.nn.relu,name='conv2')
    network = tl.layers.MaxPool2d(network, filter_size=(2, 2), name='pool2')

    network = tl.layers.FlattenLayer(network)

    network = tl.layers.DenseLayer(network,n_units=128,act=tf.nn.relu,name='fc3')
    network = tl.layers.DenseLayer(network,n_units=1,act=tf.nn.sigmoid,name='fc4')

    network = network

    return network
'''


x = tf.placeholder(dtype=tf.float32,shape=(None,80,80,1))

prob = policy_network(x)