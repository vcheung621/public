import numpy as np
import keras.models
from keras.models import model_from_json
from skimage import transform,io
import matplotlib.pyplot as plt
import cv2

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#load woeights into new model
loaded_model.load_weights("model.h5")
print("Loaded Model from disk")

#compile and evaluate loaded model
loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#loss,accuracy = model.evaluate(X_test,y_test)
#print('loss:', loss)
#print('accuracy:', accuracy)
#x = io.imread('output.png', as_gray=True)
x = cv2.imread('output.png', 0)
print (x.shape)
io.imshow(x)
plt.show()
#x = np.invert(np.array(x,dtype=np.uint8))
x = np.invert(x)
print (x.shape)
io.imshow(x)
plt.show()
#x = transform.resize(x, (28,28), mode='symmetric', preserve_range=True)
#x = transform.resize(x, (28,28))
x = cv2.resize(x, (28,28))
#x = transform.rescale(x, 0.1, anti_aliasing=False)
print (x.shape)
io.imshow(x)
plt.show()
x = x.reshape(1,28,28,1)
print (x.shape)

out = loaded_model.predict(x)
print(out)
print(np.argmax(out,axis=1))
