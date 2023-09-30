# Projekt WDSI
Projekt zaliczeniowy laboratorium z Wprowadzenia do Sztucznej Inteligencji

## Opis Projektu
Projekt ma na celuo klasyfikacje obiektów (jednego z: przejście dla pieszych, znak stop, sygnalizacja świetlna, ograniczenie prędkości) z obrazów oraz do prezentacji z podanego forderu, a następnie wyświetla informacje o danym obrazie.
Autorem projektu jest: Adam Przybyła

## Opis poszczególnych funkcji użytych w projekcie
### load_data(path,im)
Wejście: path -> folder z plikami .xml, im -> folder z obrazami
Wyjście: data -> Struktura słownika zawierająca wpisy: image -> obraz, label -> Identyfikator klasy, name -> nazwę zdjęcia wraz z rozszerzeniem .png,
size -> granice obiektu w pikselach
Funkcja wczytuje dane z plików .xml z podanego folderu do struktury. Wykorzystana do tego biblioteka xml.etree.ElementTree, która na wygodne operacje na plikach .xml
Również zastosowana funkcja os.listdir(path), pozwala na pracy na wszystkich plikach w podanym folderze.

### learn_bovw(data)
Wejście: data -> Struktura słownika zawierająca wpisy: image -> obraz, label -> Identyfikator klasy, name -> nazwę zdjęcia wraz z rozszerzeniem .png,
size -> granice obiektu w pikselach
Wyjście: nic
Nauka Bag of Visual Words (BOVW) i zapisanie nauczonego słownika jako plik slownik.npy
BOVW odnosi się do techniki, która pozwala nam na kompleksowe opisywanie obrazów oraz pozwala tworzyć zapytania dotyczące podobieństwa.

### extract_features(data)
Wejście: data -> Struktura słownika zawierająca wpisy: image -> obraz, label -> Identyfikator klasy, name -> nazwę zdjęcia wraz z rozszerzeniem .png,
size -> granice obiektu w pikselach
Wyjście: Struktura słownika zawierająca wpisy: image -> obraz, label -> Identyfikator klasy, name -> nazwę zdjęcia wraz z rozszerzeniem .png,
size -> granice obiektu w pikselach, desc -> opis obrazu utworzony przez nauczony słownik
Wyciąganie opisów dla dostarczonych danych, opisy te zostaną później wykorzystane do trenowania modelu w funkcji train. Do utworzenia opisów został wykorzystany algorytm SIFT,
który pozwala na detekcie i opisanie lokalnych cech w obrazie.

### train(data)
Wejście: Struktura słownika zawierająca wpisy: image -> obraz, label -> Identyfikator klasy, name -> nazwę zdjęcia wraz z rozszerzeniem .png,
size -> granice obiektu w pikselach, desc -> opis obrazu utworzony przez nauczony słownik
Wyjście: Wytrenowany model
Trenowanie klasyfikacji za pomocą algorytmu random forest, która polega na tworzeniu drzew decyzyjnych i generowaniu klas.

### predict(data,model)
Wejście: Wytrenowany model oraz struktura słownika zawierająca wpisy: image -> obraz, label -> Identyfikator klasy, name -> nazwę zdjęcia wraz z rozszerzeniem .png,
size -> granice obiektu w pikselach, desc -> opis obrazu utworzony przez nauczony słownik 
Wyjście: Struktura słownika zawierająca wpisy: image -> obraz, label -> Identyfikator klasy, name -> nazwę zdjęcia wraz z rozszerzeniem .png,
size -> granice obiektu w pikselach, desc -> opis obrazu utworzony przez nauczony słownik, label_pred -> określony identyfikator klasy przez wytrenowany model
Przewidywanie kategorii obiektu przez wytrenowany model

### ewaluate(data)
Wejście: Struktura słownika zawierająca wpisy: image -> obraz, label -> Identyfikator klasy, name -> nazwę zdjęcia wraz z rozszerzeniem .png,
size -> granice obiektu w pikselach, desc -> opis obrazu utworzony przez nauczony słownik, label_pred -> określony identyfikator klasy przez wytrenowany model
Wyjście: nic
Ewaluacja wyników, prezentacja wyniku procentowego poprawności przewidywanych oraz macierzy pomyłek, która jest pokazaniem ile obiektów jakiej klasy było przypisane
przez wytrenowany model do poszególnej klasy, wartości na przekątnej macierzy pokazują ile obiektów poszczególnej klasy zostały przypisane poprawnie natomiast elementy poza
główną przekątną pokazują jakie pomyłki zostały popełnione przez model

### display(data)
Wejście: Struktura słownika zawierająca wpisy: image -> obraz, label -> Identyfikator klasy, name -> nazwę zdjęcia wraz z rozszerzeniem .png,
size -> granice obiektu w pikselach, desc -> opis obrazu utworzony przez nauczony słownik, label_pred -> określony identyfikator klasy przez wytrenowany model
Wyjście: nic
Prezentacja poprawnie przypisanych obiektów na obrazach w następującej kolejności:
- Nazwa pliku (obrazu)
- Liczba poprawnie zdefiniowanych obiektów na obrazie
- Koordynaty pierwszego poprawnie zdefiniowanego obiektu na obrazie
- Etykieta pierwszego poprawnie zdefiniowanego obiektu na obrazie

### display_dataset_stats(data)
Wejście: data -> Struktura słownika zawierająca wpisy: image -> obraz, label -> Identyfikator klasy, name -> nazwę zdjęcia wraz z rozszerzeniem .png,
size -> granice obiektu w pikselach
Wyjście: nic
Pokazywanie statystyk słownika, tzn. liczby obiektów danej kategorii w strukturze słownika podanej na wejściu 

## Podsumowanie
Projekt otrzymując obrazy i pliki .xml opisujące obrazy jest w wytrenować model oraz sklasyfikować kategorie obrazów 
za pomocą algorytmów uczenia maszynowego: BOVW, SIFT, Random Forest
19.02.2022 Politechnika Poznańska

