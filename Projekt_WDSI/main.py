
# !/usr/bin/env python

"""code template"""

import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
import xml.etree.ElementTree as ET
from sklearn.metrics import  confusion_matrix

# 0 - Przejście dla pieszych
# 1 - Ograniczenie prędkości
# 2 - Znak Stop
# 3 - Sygnalizacja Świetlana
class_id_to_new_class_id = {"crosswalk": 0,
                            "speedlimit": 1,
                            "stop": 2,
                            "trafficlight": 3}
def load_data(path,im):
    """
    Wczytywanie danych z podanych ścieżek do struktury
    Wejście: path -> folder z plikami .xml, im -> folder z obrazami
    Wyjście: data -> Struktura słownika zawierająca wpisy: image -> obraz, label -> Identyfikator klasy, name -> nazwę zdjęcia wraz z rozszerzeniem .png, size -> granice obiektu w pikselach
    """
    data =[]
    for plik in os.listdir(path):
        mytree = ET.parse(os.path.join(path, plik))
        myroot = mytree.getroot()
        for x in myroot.findall('object'):
            data.append({'image': cv2.imread(os.path.join(im, myroot[1].text)),
                         'label': class_id_to_new_class_id[x.find('name').text],
                         'name':myroot[1].text,
                         'size':[x.find('bndbox/xmin').text,x.find('bndbox/ymin').text,x.find('bndbox/xmax').text,x.find('bndbox/ymax').text]})
    return data

def learn_bovw(data):
    """
    Nauka Bag of Visual Words (BOVW) i zapisanie nauczonego słownika jako plik slownik.npy
    Wejście: Słownik utworzony w 'load_load'
    Wyjście: nic

    """
    dict_size = 128
    bow = cv2.BOWKMeansTrainer(dict_size)

    sift = cv2.SIFT_create()
    for sample in data:
        kpts = sift.detect(sample['image'], None)
        kpts, desc = sift.compute(sample['image'], kpts)
        if desc is not None:
            bow.add(desc)
    vocabulary = bow.cluster()
    np.save('slownik.npy', vocabulary)

def extract_features(data):
    """
    Wyciąganie opisów dla dostarczonych danych
    Wejście: Słownik utworzony w 'load data'
    Wyjście: Słownik 'load data' z dodanym wpisem zawierającym opis obrazu "desc"
    """
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    vocabulary = np.load('slownik.npy')
    bow.setVocabulary(vocabulary)
    for sample in data:
        kpts = sift.detect(sample['image'], None)
        desc = bow.compute(sample['image'], kpts)
        sample['desc'] = desc

    return data

def train(data):
    """
    Trains Random Forest classifier.
    @param data: List of dictionaries, one for every sample, with entries "image" (np.array with image), "label" (class_id),
                    "desc" (np.array with descriptor).
    @return: Trained model.
    Trenowanie Klasyfikacji random forest
    Wejście: Słownik 'data'
    Wyjście: Wytrenowany model
    """
    descs = []
    labels = []
    for sample in data:
        if sample['desc'] is not None:
            descs.append(sample['desc'].squeeze(0))
            labels.append(sample['label'])
    rf = RandomForestClassifier()
    rf.fit(descs, labels)

    return rf
def predict(rf, data):
    """
    Przewidywanie kategorii obiektu przez wytrenowany model
    Wejście: Wytrenowany model oraz słownik data
    Wyjście: Słownik data z dodanym wpisem predykowanych kategorii przez wytrenowany model
    """
    for sample in data:
        if sample['desc'] is not None:
            predict = rf.predict(sample['desc'])
            sample['label_pred'] = int(predict)
    return data
def evaluate(data):
    """
    Ewaluacja wyników
    Wejście: słownik data
    Wyjście: nic
    """
    correct = 0
    incorrect = 0
    eval = []
    real = []
    for sample in data:
        if sample['desc'] is not None:
            eval.append(sample['label_pred'])
            real.append(sample['label'])
            if sample['label_pred'] == sample['label']:
                correct += 1
            else:
                incorrect += 1

    print('score = %.3f' % (correct / max(correct + incorrect, 1)))

    con_matrix = confusion_matrix(real,eval)
    print(con_matrix)
    return
def display(data):
    """
    Prezentacja wyników
    Wejście: słownik data
    Wyjście: nic
    """
    for idx, sample in enumerate(data):
        if sample['desc'] is not None:
            if sample['label_pred'] == sample['label']:
                if sample['label_pred'] == 0:
                    print(sample['name'])
    # this function does not return anything
    return
def display_dataset_stats(data):
    """
    Pokazywanie statystyk słownika
    Wejście: Słownik data
    Wyjście: nic
    """
    class_to_num = {}
    for idx, sample in enumerate(data):
        class_id = sample['label']
        if class_id not in class_to_num:
            class_to_num[class_id] = 0
        class_to_num[class_id] += 1

    class_to_num = dict(sorted(class_to_num.items(), key=lambda item: item[0]))
    print(class_to_num)
def main():

    data_train = load_data('annotations','images')
    print('train dataset:')
    display_dataset_stats(data_train)

    data_test = load_data('annotations_test','images_test')
    print('test dataset:')
    display_dataset_stats(data_test)

    #print('learning BoVW')
    #learn_bovw(data_train)

    print('extracting train features')
    data_train = extract_features(data_train)

    print('training')
    rf = train(data_train)

    print('extracting test features')
    data_test = extract_features(data_test)

    print('testing on testing dataset')
    data_test = predict(rf, data_test)
    evaluate(data_test)
    display(data_test)
    return

if __name__ == '__main__':
    main()
