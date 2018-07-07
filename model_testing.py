from keras.preprocessing import image
from keras.models import model_from_json
import numpy as np
from keras.applications.imagenet_utils import preprocess_input

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")


# let's test the model with an image 
img_path = 'sad.jpg'
img = image.load_img(img_path, target_size=(64, 64))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
# if the output is 0 this means not happy 
print(model.predict(x))