# import needed packages
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.models import model_from_json
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import * #file written to help in loading data sets

#import keras.backend as K
#K.set_image_data_format('channels_last')
#import matplotlib.pyplot as plt
#from matplotlib.pyplot import imshow
#%matplotlib inline

class ModelTraining():
    """docstring for ModelTraining"""
   
    def loadDataSet(self):
        """ 
            loadDataSet only load the data set by calling load_dataset from kt_utils file 
            note that orig means original images before normalization 
        """ 
        X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
        # then normalize images by dividing by 255
        X_train = X_train_orig/255. # train set
        X_test = X_test_orig/255.   # test  set 
        
        Y_train = Y_train_orig.T
        Y_test = Y_test_orig.T
        return X_train , Y_train , X_test , Y_test , classes
    def BuildModel(self,input_shape):
        """ build model will build the model architecutre and return model object 
            
            * the model is showed in the image in readme file 
        """
        X_input = Input(input_shape)
        X       = ZeroPadding2D((3,3))(X_input)                  # zero padding
        X       = Conv2D(32,(7,7),strides=(1,1),name="conv0")(X) # conv layer
        X       = BatchNormalization(axis=3, name="bn0")(X)      # batch normalization
        X       = Activation('relu')(X)                          # activation function  
        X       = MaxPooling2D((2,2),name='max_pool')(X)   # max pooling after activation 
        X       = Flatten()(X)                                   # flat the output for FC layer
        X       = Dense(1, activation='sigmoid', name='fc')(X)  # FC layer    
        model = Model(inputs = X_input, outputs = X, name='happiness_model') # put all together

        return model 
    def main(self):
        """put all together to train the model then save it after evaluation """
        X_train , Y_train , X_test , Y_test , classes = self.loadDataSet()

        model = self.BuildModel(X_train.shape[1:]) # create object from the model 
        # then compile it using adam optimizer and binary_crossentropy as loss function 
        model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
        #all thing is ready let's train 

        model.fit(x=X_train,y=Y_train,epochs=40,batch_size = 32)

        print model.summary() # print model summary
        predictions = model.evaluate(X_test,Y_test)
        print ("loss = " + str(predictions[0]))
        print ("Test Accuracy = " + str(predictions[1]))


        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")

 
if __name__ == "__main__":
    modelTrainer = ModelTraining()
    modelTrainer.main()









