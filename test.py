import numpy as np
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import matplotlib.pyplot as plt
import keras
from PIL import Image


model = keras.saving.load_model("/Users/mike/Documents/Python/MNIST-Handwritten/mnist_model.keras")

my_image = Image.open("test_img.png").convert('L').resize((28,28))
my_image = np.array(my_image)
my_image = my_image / 255.0
my_image = my_image.reshape(1,28,28,1)
prediction = model.predict(my_image)
print(prediction)
predicted_class = np.argmax(prediction)
print("Predicted digit:", predicted_class)

# matplotlib image result
import matplotlib.pyplot as plt
plt.imshow(my_image.reshape(28,28), cmap = "gray")
plt.title(f"Predicted: {predicted_class}")
plt.show()