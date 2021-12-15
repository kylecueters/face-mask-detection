from utils import split_data, create_generators
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, VGG16
from tensorflow.keras.layers import Conv2D, Input, AveragePooling2D, Flatten, Dense, Dropout, MaxPool2D, BatchNormalization, GlobalAvgPool2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

if __name__=="__main__":
    SPLIT_DATA = False
    TRAIN = True

    dataset_path = 'dataset'
    training_data_train_path = 'training_data/train'
    training_data_validation_path = 'training_data/validation'

    if SPLIT_DATA:
        split_data(dataset_path, training_data_train_path, training_data_validation_path)

    if TRAIN:
        print("[INFO] preparing to train...")

        # initialize the initial learning rate, number of epochs to train for,
        # and batch size
        BATCH_SIZE = 64
        INIT_LR = 1e-4
        EPOCHS = 20
        MODEL_PATH = 'mask_detector.model'
        PATIENCE = 8

        train_generator, val_generator = create_generators(BATCH_SIZE, training_data_train_path, training_data_validation_path)
        nbr_classes = train_generator.num_classes
        label_map = (train_generator.class_indices)

        # callbacks
        ckpt_saver = ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            mode='max',
            save_best_only=True,
            save_freq='epoch',
            verbose=1
        )
        early_stop = EarlyStopping(monitor="val_accuracy", patience=PATIENCE)

        # constructing model
        print("[INFO] constructing model...")

        # load the MobileNetV2 network, ensuring the head FC layer sets are
        # left off
        baseModel = MobileNetV2(weights="imagenet", include_top=False,
            input_tensor=Input(shape=(224, 224, 3)))

        # construct the head of the model that will be placed on top of the
        # the base model
        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(128, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(3, activation="softmax")(headModel)

        # place the head FC model on top of the base model (this will become
        # the actual model we will train)
        model = Model(inputs=baseModel.input, outputs=headModel)

        # loop over all layers in the base model and freeze them so they will
        # *not* be updated during the first training process
        for layer in baseModel.layers:
            layer.trainable = False

        # compile our model
        print("[INFO] compiling model...")
        # optimizer = Adam(learning_rate=INIT_LR, amsgrad=True)
        # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
        model.compile(loss="categorical_crossentropy", optimizer=opt,
            metrics=["accuracy"])

        # train the head of the network
        print("[INFO] training head...")
        H = model.fit(
            train_generator,
            validation_data=val_generator,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[ckpt_saver, early_stop]
            )

        # summarize history for accuracy
        print("[INFO] generating model results...")
        plt.plot(H.history['accuracy'])
        plt.plot(H.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        # summarize history for loss
        plt.plot(H.history['loss'])
        plt.plot(H.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        