import shutil
import os
import glob
from sklearn.model_selection import train_test_split
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def split_data(path_to_data, path_to_save_train, path_to_save_val, split_size=0.1):
    folders = os.listdir(path_to_data)

    for folder in folders:
        full_path = os.path.join(path_to_data, folder)
        image_file_ext = ('*.png', '*.jpg', '*.jpeg')
        images_paths = []
        for ext in image_file_ext:
            images_paths.extend(glob.glob(os.path.join(full_path, ext)))

        x_train, x_val = train_test_split(images_paths, test_size=split_size)

        for x in x_train:
            path_to_folder = os.path.join(path_to_save_train, folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder)

        for x in x_val:
            path_to_folder = os.path.join(path_to_save_val, folder)
            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder)

def create_generators(batch_size, train_data_path, val_data_path):
    train_preprocessor = ImageDataGenerator(
        # rescale = 1 / 255.,
        rotation_range=10,
        width_shift_range=0.1,
        zoom_range=0.15,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    test_preprocessor = ImageDataGenerator(
        # rescale = 1 / 255.,
    )

    train_generator = train_preprocessor.flow_from_directory(
        train_data_path,
        class_mode="categorical",
        target_size=(224,224),
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size
    )

    val_generator = test_preprocessor.flow_from_directory(
        val_data_path,
        class_mode="categorical",
        target_size=(224,224),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size,
    )

    return train_generator, val_generator