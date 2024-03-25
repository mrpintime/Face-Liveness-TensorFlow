import tensorflow as tf
from Model import Model
from Data_Loader import Data_Loader



# Create and compile the model
model = Model()

#======================# prepare data #======================
Data_LoaderObj = Data_Loader(augmentation=False)
directory = 'data/images/Train'
if not Data_LoaderObj.augmentation:
    imgs, labels = Data_LoaderObj.load_images_from_directory(directory=directory)
    # print(imgs.shape, labels.shape)
    # preproccess data
    x_train, y_train_label, y_train_mask = Data_LoaderObj.prepare_data(img=imgs, all_label=labels)
    
else:
    #TODO: Implement data proccesing with augmented data
    pass

#======================# End #======================

#======================# Train model #======================

model.fit(x_train=x_train, y_train_label=y_train_label, y_train_mask=y_train_mask, epochs=5, batch_size=2)