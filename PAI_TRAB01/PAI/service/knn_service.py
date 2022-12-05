from sklearn.neighbors import KNeighborsClassifier
from imutils import paths
import numpy as np
import cv2
import os
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import recall_score, make_scorer, confusion_matrix, accuracy_score, f1_score, precision_score
from PAI.service.image_service import getImgFromB64Internal
from math import pi
from PAI.models import Resultado
import time
import pickle


def knn_classify(joelhosTreino, joelhosTeste, joelhosValidacao):
    n_neighbors = 100 # numero de vizinhos a serem considerados
    jobs = 4 # numero de processos a serem criados

    testFeat, testLabels = get_features_and_labels(joelhosTeste)
    valFeat, valLabels = get_features_and_labels(joelhosValidacao)

    training_time_start = time.time()

    # train and evaluate a k-NN classifer on the raw pixel intensities
    try:
        model = pickle.load(open("knn.sav", 'rb'))
    except FileNotFoundError:
        print("[WARN] Gerando modelo novo")

        trainFeat, trainLabels = get_features_and_labels(joelhosTreino)

        model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=jobs, metric="cityblock")
        model.fit(trainFeat, trainLabels)

    training_time_stop = (time.time() - training_time_start) * 1000

    acc = model.score(valFeat, valLabels)

    class_time_start = time.time()

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average="macro"),
        'recall': make_scorer(recall_score, average="macro"),
        'f1': make_scorer(f1_score, average="macro"),
        'specificity': make_scorer(recall_score, pos_label=0, average="macro")
    }

    scores = cross_validate(model, testFeat, testLabels, cv=5, scoring=scoring, return_train_score=False)
    scores = {key:np.mean(value) for (key, value) in scores.items()}

    y_pred = cross_val_predict(model, testFeat, testLabels, cv=5)
    matrix = confusion_matrix(testLabels, y_pred)

    class_time_stop = (time.time() - class_time_start) * 1000

    result = Resultado(
        classificador="raso", acuracia=scores["test_accuracy"], sensibilidade=scores["test_recall"], 
        especificidade=scores["test_specificity"], precisao=scores["test_precision"], 
        escore=scores["test_f1"], 
        tempo_treino_ms=training_time_stop,
        tempo_classificacao_ms=class_time_stop
    )
    
    pickle.dump(model, open("knn.sav", 'wb'))
    
    return result, matrix


def knn_classify_binary(joelhosTreino, joelhosTeste, joelhosValidacao):
    n_neighbors = 100 # numero de vizinhos a serem considerados
    jobs = 4 # numero de processos a serem criados

    testFeat, testLabels = get_features_and_labels_binary(joelhosTeste)
    valFeat, valLabels = get_features_and_labels(joelhosValidacao)

    # train and evaluate a k-NN classifer on the raw pixel intensities

    training_time_start = time.time()

    try:
        model = pickle.load(open("knn_binary.sav", 'rb'))
    except FileNotFoundError:
        print("[WARN] Gerando modelo novo")

        trainFeat, trainLabels = get_features_and_labels_binary(joelhosTreino)

        model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=jobs, metric="cityblock")
        model.fit(trainFeat, trainLabels)

    acc = model.score(valFeat, valLabels)
    print("[INFO] Acurácia encontrada: {:.2f}%".format(acc * 100))

    training_time_stop = (time.time() - training_time_start) * 1000

    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'specificity': make_scorer(recall_score, pos_label=0)
    }

    class_time_start = time.time()

    scores = cross_validate(model, testFeat, testLabels, cv=5, scoring=scoring, return_train_score=False)
    scores = {key:np.mean(value) for (key, value) in scores.items()}

    y_pred = cross_val_predict(model, testFeat, testLabels, cv=5)
    matrix = confusion_matrix(testLabels, y_pred)

    class_time_stop = (time.time() - class_time_start) * 1000

    result = Resultado(
        classificador="raso", acuracia=scores["test_accuracy"], sensibilidade=scores["test_recall"], 
        especificidade=scores["test_specificity"], precisao=scores["test_precision"], 
        escore=scores["test_f1"], 
        tempo_treino_ms=training_time_stop,
        tempo_classificacao_ms=class_time_stop
    )

    pickle.dump(model, open("knn_binary.sav", 'wb'))
    
    return result, matrix



def get_features_and_labels(joelhos):
    rawImages = []
    features = []
    labels = []

    for (i, joelho) in enumerate(joelhos):
        image = getImgFromB64Internal(joelho.imagem)
        label = joelho.rotulo

        imgFeatures = getFeaturesFromImage(image)

        rawImages.append(image)
        features.append(imgFeatures)
        labels.append(label)

        if i > 0 and i % 100 == 0:
            print("[INFO] Processamento dos joelhos {}/{}".format(i, len(joelhos)))

    return features, labels


def get_features_and_labels_binary(joelhos):
    rawImages = []
    features = []
    labels = []

    for (i, joelho) in enumerate(joelhos):
        image = getImgFromB64Internal(joelho.imagem)
        label = 0 if joelho.rotulo == 0 else 1

        imgFeatures = getFeaturesFromImage(image)

        rawImages.append(image)
        features.append(imgFeatures)
        labels.append(label)

        if i > 0 and i % 100 == 0:
            print("[INFO] Processamento dos joelhos {}/{}".format(i, len(joelhos)))

    return features, labels


def getFeaturesFromImage(img):
    # Caso a imagem possua cor, transforma-la em uma imagem com apenas tons de cinza
    if(len(img.shape) > 2):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imageFeatures = []

    mtx = graycomatrix(img, distances = [1, 2, 4, 8, 16], angles=[0, pi/4, pi/2, 3*pi/4], levels=256)

    contrast = np.hstack(graycoprops(mtx, "contrast"))
    homo = np.hstack(graycoprops(mtx, "homogeneity"))
    energy = np.hstack(graycoprops(mtx, "energy"))
    correlation = np.hstack(graycoprops(mtx, "correlation"))

    imageFeatures.append(contrast)
    imageFeatures.append(homo)
    imageFeatures.append(energy)
    imageFeatures.append(correlation)

    return np.concatenate(imageFeatures)
