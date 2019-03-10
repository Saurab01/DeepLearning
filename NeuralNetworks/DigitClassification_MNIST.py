#https://towardsdatascience.com/deep-learning-for-beginners-practical-guide-with-python-and-keras-d295bfca4487
import keras
from keras.datasets import mnist
import numpy as np

(x_train, y_train),(x_test, y_test) = mnist.load_data()
#The images are encoded as Numpy arrays and their corresponding labels ranging from 0 to 9.

import matplotlib.pyplot as plt
plt.imshow(x_train[8], cmap=plt.cm.binary)
print('y_train[8]=',y_train[8])
print('x_train.ndim=',x_train.ndim)
print('x_train.shape=',x_train.shape)
print('type of data in x_train=',x_train.dtype)

#Data Normalization
#These MNIST images of 28×28 pixels are represented as an array of numbers whose values
# range from [0, 255] of type uint8. But it is usual to scale the input values of
# neural networks to certain ranges.
# In the example of this post the input values should be scaled to values of type
# float32 within the interval [0, 1]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


#Transformation  as the matrix of 28×28 numbers can be represented by a vector (array) of 784 numbers
#  (concatenating row by row), which is the format that accepts as input a densely connected
# neural network like the one we will see in this post

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

print('after transformation x_train.shape)=',x_train.shape)
print('after transformation x_test.shape)=',x_test.shape)

'''we have the labels for each input data (remember that in our case they are numbers between 0 and 9 
that indicate which digit represents the image, that is, to which class is associated). 
In this example, and as we have already advanced, we will represent this label with a vector of 
10 positions, where the position corresponding to the digit that represents the image contains 
a 1 and the remaining positions of the vector contain the value 0.
In this example we will use what is known as one-hot encoding, which we have already mentioned, 
which consists of transforming the labels into a vector of as many zeros as the number of 
different labels, and containing the value of 1 in the index that corresponds to the value of the 
label. 
Keras offers many support functions, including to_categorical to perform precisely this 
transformation, which we can import from keras.utils:
'''
from keras.utils import to_categorical

print('\nBefore Trasformation value at y_train[0]=',y_train[0])
print('value at y_test[0]=',y_test[0])
print('y_train.shape=',y_train.shape)
print('y_test.shape=',y_test.shape)

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
print('\nAfter Transformation\ny_train.shape=',y_train.shape)
print('y_test.shape=',y_test.shape)
print('y_train[0]=',y_train[0])
print('y_test[0]=',y_test[0])


#DEFINING THE MODEL

#Here, the neural network has been defined as a sequence of two layers that are densely connected
#  (or fully connected), meaning that all the neurons in each layer are connected to all
# the neurons in the next layer
from keras.models import Sequential
from keras.layers.core import Dense, Activation
model = Sequential()
model.add(Dense(10, activation='sigmoid', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))
#The second layer in this example is a softmax layer of 10 neurons, which means that it will
# return a matrix of 10 probability values representing the 10 possible digits
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size=100, epochs=5)
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
predictions = model.predict(x_test)
print(np.argmax(predictions[11]))
print(predictions[11])
print(np.sum(predictions[11]))