from mnist import Mnist
import mnist_data
import cv2
from random import randint

o_mnist = Mnist()
x_train, y_train, x_test, y_test = mnist_data.process_dataset()
o_mnist.create_model()
# Train the model on the data
o_mnist.model.fit(x_train, y_train, epochs=5, verbose=1)
# Evaluate the model on test data
o_mnist.model.evaluate(x_test, y_test)

n = randint(0,10)
x = cv2.imread("data/{}_L.jpg".format(n), cv2.IMREAD_UNCHANGED)
x = x.reshape((1, 28, 28, 1))
y = o_mnist.model.predict(x)
print("Actural Case {}: \n".format(n))
print("Prediction  y: \n", y)
o_mnist.save()