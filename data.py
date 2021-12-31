import tensorflow as tf
from sklearn.model_selection import train_test_split
import glob
import pandas as pd
import matplotlib.pyplot as plt

def parse_image(filename, label=None):
    '''filename: tensor'''
    file = tf.io.read_file(filename)
    img = tf.image.decode_image(file)
    #not resized since Keras layers will be used for resizing    
    return img, label

class Data:

    def __init__(self, base, train_file, train_folder, test_file, test_folder):
        self.base = base
        
        self.train_file = train_file
        self.train_folder = train_folder
        
        self.test_file = test_file
        self.test_folder = test_folder
        
    def build_dataset(self, random_state = 1989):

        #add the paths of train images into a column in data frame
        image_df = pd.read_csv(self.train_file)
        image_df['id_path'] = [self.train_folder + id + '.jpg' for id in image_df['Id'].values]
        
        
        # Scale outcome for avoiding hockey stick
        image_df['Pawpularity'] = image_df['Pawpularity'].values - image_df['Pawpularity'].mean()
        (train_df, val_df) =  train_test_split(image_df, train_size=0.85, random_state = random_state)
    
        # add the paths of test images into a column in data frame
        test_df = pd.read_csv(self.test_file)
        test_df['id_path'] = [self.test_folder + id + '.jpg' for id in test_df['Id'].values]

        # To TF dataset
        # convert the features to tensor
        train_files = tf.convert_to_tensor(train_df['id_path'].values)
        val_files = tf.convert_to_tensor(val_df['id_path'].values)
        test_files = tf.convert_to_tensor(test_df['id_path'].values)

        # convert labels to tensor
        y_train_tf = tf.convert_to_tensor(train_df['Pawpularity'])
        y_val_tf = tf.convert_to_tensor(val_df['Pawpularity'])

        train_tfd = tf.data.Dataset.from_tensor_slices((train_files, y_train_tf))
        train_tfd = train_tfd.map(parse_image).batch(1)

        val_tfd = tf.data.Dataset.from_tensor_slices((val_files, y_val_tf))
        val_tfd = val_tfd.map(parse_image).batch(1)

        test_tfd = tf.data.Dataset.from_tensor_slices(test_files)
        test_tfd = test_tfd.map(parse_image).batch(1)

        return train_tfd, val_tfd, test_tfd


if __name__ == '__main__':

    # a simple test is added to make sure dataset object is working
    BASE_DIR = 'C:/Users/aciha/Documents/projects/pawpularity/'
    TRAIN_FILE = BASE_DIR + 'data/train.csv'
    TRAIN_FOLDER = BASE_DIR + 'data/train/'

    TEST_FILE = BASE_DIR + 'data/test.csv'
    TEST_FOLDER = BASE_DIR + 'data/test/'
    
    data = Data(base = BASE_DIR, train_file = TRAIN_FILE, \
                train_folder = TRAIN_FOLDER, test_file = TEST_FILE, \
                test_folder = TEST_FILE)
    train_tfd, val_tfd, test_tfd = data.build_dataset()
    for image, label in train_tfd.take(1):
        plt.imshow(image[0])
        plt.title(round(int(label)))
        plt.axis("off")
        plt.show()
