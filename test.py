#from glob import glob
#class_names = glob("105_classes_pins_dataset/*/") # Reads all the folders in which images are present
##class_names = sorted(class_names) # Sorting them
#name_id_map = dict(zip(range(len(class_names)),class_names))
#print(name_id_map)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAINING_DIR = "105_classes_pins_dataset"
img_height =160
img_width = 160
batch_size = 32
train_datagen = ImageDataGenerator(
                rescale=1/255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                rotation_range=40,
                width_shift_range=0.1,
                height_shift_range=0.1,
                validation_split=0.2)

train_set = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') # set as training data

validation_set = train_datagen.flow_from_directory(
    TRAINING_DIR, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') # set as validation data

print(validation_set.class_indices)