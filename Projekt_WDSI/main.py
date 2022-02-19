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

label_to_ID = {"crosswalk": 0,
                "speedlimit": 1,
                "stop": 2,
                "trafficlight": 3}

ID_to_label = {0:'crosswalk',
               1:'speedlimit',
               2:'stop',
               3:'trafficlight'}

def load_data(path,im):

    data =[]
    for plik in os.listdir(path):
        mytree = ET.parse(os.path.join(path, plik))
        myroot = mytree.getroot()
        for x in myroot.findall('object'):
            data.append({'image': cv2.imread(os.path.join(im, myroot[1].text)),
                         'label': label_to_ID[x.find('name').text],
                         'name':myroot[1].text,
                         'size':[x.find('bndbox/xmin').text,x.find('bndbox/ymin').text,x.find('bndbox/xmax').text,x.find('bndbox/ymax').text]})

    return data

def learn_bovw(data):

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

    for sample in data:
        if sample['desc'] is not None:
            predict = rf.predict(sample['desc'])
            sample['label_pred'] = int(predict)

    return data

def evaluate(data):

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
    print('Wynik procentowy = %.3f' % (correct / max(correct + incorrect, 1)))
    con_matrix = confusion_matrix(real,eval)
    print('Macierz pomyłek')
    print(con_matrix)

    return

def display(data):

    disp = dict()
    size =[]
    i=0
    labels = []
    for idx, sample in enumerate(data):
        if sample['desc'] is not None:
            if sample['label_pred'] == sample['label']:
                if sample['name'] not in disp:
                    disp[sample['name']] = 1
                    size.append(sample['size'])
                    labels.append(sample['label_pred'])
                else:
                    disp[sample['name']] += 1
    for name, count in disp.items():
        print(name)
        print('Liczba obiektów na zdjęciu: ',count)
        print('Koordynaty pierwszego poprawnie zklasyfikowanego znaku:',size[i])
        print('Przewidywana Kategoria pierwszego znaku na zdjęciu: ',ID_to_label[labels[i]])
        i+=1

    return

def display_dataset_stats(data):
    class_to_num = {}
    for idx, sample in enumerate(data):
        class_id = sample['label']
        if class_id not in class_to_num:
            class_to_num[class_id] = 0
        class_to_num[class_id] += 1

    for label,count in class_to_num.items():
        print(ID_to_label[label],":",count)

def main():

    data_train = load_data('annotations','images')
    print('Statystyki zbioru treningowego:')
    display_dataset_stats(data_train)

    data_test = load_data('annotations_test','images_test')
    print('Statystyki zbioru testowego:')
    display_dataset_stats(data_test)

    #print('Nauka BoVW')
    #learn_bovw(data_train)
    #print('Nauka zakonczona')

    print('Wyciąganie opisów dla zbioru testowego')
    data_train = extract_features(data_train)
    print('Opisy wyciągnięte')

    print('Trenowanie modelu')
    rf = train(data_train)
    print('Trening zakończony')

    print('Tworzenie opisów dla zbioru testowego')
    data_test = extract_features(data_test)
    print('Zakończono tworzenie opisów zbioru testowego')

    print('Testowanie')
    data_test = predict(rf, data_test)
    print('Testowanie zakończone')

    print('Ewaluacja wyników')
    evaluate(data_test)
    print('Ewaluacja zakończona')

    print('Prezentacja poprawnie zklasyfikowanych obiektów')
    display(data_test)
    return

if __name__ == '__main__':
    main()
