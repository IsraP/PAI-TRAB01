import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.rcsetup as rcsetup
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from tensorflow.keras.layers import BatchNormalization
import keras.backend as K
import keras_metrics as km
import time
import seaborn as sns 
from sklearn.metrics import confusion_matrix
import pickle
from PAI.service.image_service import getImgFromB64Internal
from PAI.models import Resultado


def classify_alexnet(joelhosTreino, joelhosTeste, joelhosValidacao):
    binary = False

    def activeBinary():
        global binary
        binary = True

    fpath = "C:/PAI/KneeXrayData/ClsKLData/kneeKL224/test"
    trpath = "C:/PAI/KneeXrayData/ClsKLData/kneeKL224/train" #train

    categories = os.listdir(fpath)
    categories = categories[:20]

    categoriesTr = os.listdir(trpath)
    categoriesTr = categoriesTr[:20]

    def load_images_and_labels(joelhos):
        img_lst=[]
        labels=[]

        for (i, joelho) in enumerate(joelhos):
            image = getImgFromB64Internal(joelho.imagem)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_array = Image.fromarray(image, 'RGB')
            resized_img = img_array.resize((227, 227))

            label = joelho.rotulo

            img_lst.append(np.array(resized_img))
            labels.append(label)

        return img_lst, labels

    images, labels = load_images_and_labels(joelhosTeste)
    print("No. of images loaded = ",len(images),"\nNo. of labels loaded = ",len(labels))
    print(type(images),type(labels))

    imagesTr, labelsTr = load_images_and_labels(joelhosTreino)
    print("No. of images loaded = ",len(imagesTr),"\nNo. of labels loaded = ",len(labelsTr))
    print(type(imagesTr),type(labelsTr))

    images = np.array(images)
    labels = np.array(labels)

    if(binary == True):
        for i,label in enumerate(labels):
            if(label == 1 or label == 2 or label == 3 or label == 4):
                labels[i] = 1

    print(labels)

    imagesTr = np.array(imagesTr)
    labelsTr = np.array(labelsTr)

    if(binary == True):
        for i,label in enumerate(labelsTr):
            if(label == 1 or label == 2 or label == 3 or label == 4):
                labelsTr[i] = 1

    print(labelsTr)

    def get_f1(y_true, y_pred): #taken from old keras source code
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val

    def get_precision(y_true, y_pred): #taken from old keras source code
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return precision

    def get_recall(y_true, y_pred): #taken from old keras source code
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return recall

    model=Sequential()
    try:    
        filename = 'Alexnet_model.sav'
        model = pickle.load(open(filename,'rb'))
        
    except FileNotFoundError:
        model.add(Conv2D(filters=96,kernel_size=(11,11),strides=(4,4),padding="valid",activation="relu",input_shape=(227,227,3)))

        #1 max pool layer
        model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

        model.add(BatchNormalization())

        #2 conv layer
        model.add(Conv2D(filters=256,kernel_size=(5,5),strides=(1,1),padding="valid",activation="relu"))

        #2 max pool layer
        model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

        model.add(BatchNormalization())

        #3 conv layer
        model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding="valid",activation="relu"))

        #4 conv layer
        model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding="valid",activation="relu"))

        #5 conv layer
        model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding="valid",activation="relu"))

        #3 max pool layer
        model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

        model.add(BatchNormalization())


        model.add(Flatten())

        #1 dense layer
        model.add(Dense(4096,input_shape=(227,227,3),activation="relu"))

        model.add(Dropout(0.4))

        model.add(BatchNormalization())

        #2 dense layer
        model.add(Dense(4096,activation="relu"))

        model.add(Dropout(0.4))

        model.add(BatchNormalization())

        #3 dense layer
        model.add(Dense(1000,activation="relu"))

        model.add(Dropout(0.4))

        model.add(BatchNormalization())

        #output layer
        model.add(Dense(100,activation="softmax"))

    model.summary()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy",get_f1, get_precision, get_recall])
    filename = 'Alexnet_model.sav'
    training_time_start = time.time()
    model.fit(imagesTr, labelsTr, epochs=1)
    training_time_stop = (time.time() - training_time_start) * 1000
    print(f"Tempo de treino{training_time_stop}")
    class_time_start = time.time()
    loss, accuracy, f1, precision, recall = model.evaluate(images, labels)
    class_time_stop = (time.time() - class_time_start) * 1000
    print(f"Tempo de classificacao {class_time_stop}")
    print(f"accuracy = {accuracy} f1 = {f1} precision = {precision} recall = {recall}")

    filename = 'Alexnet_model.sav'
    pickle.dump(model, open(filename, 'wb'))


    pred = model.predict(images)

    pred.shape

    n = 0 

    labelsPred = [np.argmax(vect) for vect in pred]

    cf = confusion_matrix(labels, labelsPred)

    ax = plt.subplot()
    sns.heatmap(cf, annot= True, fmt= 'g', ax = ax)

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    # ax.xaxis.set_ticklabels(CLASSES)
    # ax.yaxis.set_ticklabels(CLASSES)

    plt.xticks(rotation = 90)
    plt.yticks(rotation = 360)

    plt.show()

    result = Resultado(
        classificador="alexnet", acuracia= accuracy, sensibilidade=recall, 
        especificidade=scores["test_specificity"], precisao=precision, 
        escore=f1, 
        tempo_treino_ms=training_time_stop,
        tempo_classificacao_ms=class_time_stop
    )

    accuracy, f1, precision, recall