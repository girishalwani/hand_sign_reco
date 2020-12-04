import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import losses

classifier = Sequential()

classifier.add(Conv2D(32,(3,3),activation='relu',input_shape=(64,64,1)))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=6,activation='softmax'))

classifier.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory('train',target_size=(64,64),batch_size=5,color_mode='grayscale',class_mode='categorical')

test_set = test_datagen.flow_from_directory('test',target_size=(64,64),batch_size=5,color_mode='grayscale',class_mode='categorical')

model = classifier.fit_generator(
    training_set,steps_per_epoch=60,epochs=10,validation_data=test_set,validation_steps=3
)

model_json = classifier.to_json()

with open("model-bw.json","w") as json_file:
    json_file.write(model_json)

classifier.save_weights('model-bw.h5')
