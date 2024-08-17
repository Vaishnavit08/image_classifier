import cv2 as cv 
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import datasets, models, layers

# Load CIFAR-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

# Normalize pixel values
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# Class names for CIFAR-10
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Plotting the first 16 images
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()

# Slicing the dataset for quicker training
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

model=models.load_model('image_classifier.keras')

img=cv.imread('horssee.jpg')
img=cv.cvtColor(img,cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img]))/255.0

index=np.argmax(prediction)
print(f"Prediction is {class_names[index]}")























# the given code is replaced by the "model=models.load_model('image_classifier.keras')"

# first use below code to get know the working and as to create the different file in same directory.

# # [
# # Define the model
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))

# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

# # Evaluate the model
# loss, accuracy = model.evaluate(testing_images, testing_labels)
# print(f"Loss: {loss}")
# print(f"Accuracy: {accuracy}")

# # Save the model
# model.save('image_classifier.keras')  # Use .keras extension
# # Or use the following line if you prefer .h5 extension
# # model.save('image_classifier.h5')
# # ]
