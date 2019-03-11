from keras import layers
from keras import models

'''have 32 filters using a 5×5 window for the convolutional layer and a 2×2 window for the pooling
 We will use the ReLU activation function. 
 In this case, we are configuring a convolutional neural network to process an input tensor of 
 size (28, 28, 1),which is the size of the MNIST images (the third parameter is 
 the color channel which in our case is depth 1)
'''
model = models.Sequential()
model.add(layers.Conv2D(32,(5,5),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2, 2)))
#The number of parameters of the conv2D layer corresponds to the weight matrix W of 5×5
# and a b bias for each of the filters is 832 parameters (32 × (25 + 1)).
#No paramaters for MaxPooling

model.add(layers.Conv2D(64, (5, 5), activation='relu')) #layer 2 added of 64 filters
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())  #4*4*64=1024
model.add(layers.Dense(10, activation='softmax'))
model.summary()


#Training
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(train_images, train_labels, batch_size=100, epochs=5, verbose=1)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)