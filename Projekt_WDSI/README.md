# Projekt WDSI
Projekt zaliczeniowy laboratorium z Wprowadzenia do Sztucznej Inteligencji

## Opis Projektu
Projekt został stworzony to klasyfikacji znaków z obrazów oraz do prezentacji z podanego forderu, a następnie wyświetla informacje o danym obrazie.
Autorem projektu jest: Adam Przybyła 144624

## Opis poszczególnych funkcji użytych w projekcie
### load_data
Wejście: path -> folder z plikami .xml, im -> folder z obrazami
Wyjście: data -> Struktura słownika zawierająca wpisy: image -> obraz, label -> Identyfikator klasy, name -> nazwę zdjęcia wraz z rozszerzeniem .png,
size -> granice obiektu w pikselach
Funkcja wczytuje dane z plików .xml z podanego folderu do struktury.
### learn_bovw
Wejście: data -> Struktura słownika zawierająca wpisy: image -> obraz, label -> Identyfikator klasy, name -> nazwę zdjęcia wraz z rozszerzeniem .png,
size -> granice obiektu w pikselach
Wyjście: nic
Nauka Bag of Visual Words (BOVW) i zapisanie nauczonego słownika jako plik slownik.npy
### extract_features
Wejście: Struktura słownika zawierająca wpisy: image -> obraz, label -> Identyfikator klasy, name -> nazwę zdjęcia wraz z rozszerzeniem .png,
size -> granice obiektu w pikselach
Wyciąganie opisów dla dostarczonych danych
### train
Wejście: 
Wyjście: Wytrenowany model
Trenowanie Klasyfikacji random forest
### predict
Wejście: Wytrenowany model oraz struktura słownika zawierająca wpisy: image -> obraz, label -> Identyfikator klasy, name -> nazwę zdjęcia wraz z rozszerzeniem .png,
size -> granice obiektu w pikselach, desc -> opis obrazu utworzony przez nauczony słownik 
Wyjście: Struktura słownika zawierająca wpisy: image -> obraz, label -> Identyfikator klasy, name -> nazwę zdjęcia wraz z rozszerzeniem .png,
size -> granice obiektu w pikselach, desc -> opis obrazu utworzony przez nauczony słownik, label_pred -> określony identyfikator klasy przez wytrenowany model
Przewidywanie kategorii obiektu przez wytrenowany model
### ewaluate
Wejście: Struktura słownika zawierająca wpisy: image -> obraz, label -> Identyfikator klasy, name -> nazwę zdjęcia wraz z rozszerzeniem .png,
size -> granice obiektu w pikselach, desc -> opis obrazu utworzony przez nauczony słownik, label_pred -> określony identyfikator klasy przez wytrenowany model
Wyjście: nic
Ewaluacja wyników
### display
Wejście: Struktura słownika zawierająca wpisy: image -> obraz, label -> Identyfikator klasy, name -> nazwę zdjęcia wraz z rozszerzeniem .png,
size -> granice obiektu w pikselach, desc -> opis obrazu utworzony przez nauczony słownik, label_pred -> określony identyfikator klasy przez wytrenowany model
Wyjście: nic
Prezentacja wyników
### display_dataset_stats
Wejście: data -> Struktura słownika zawierająca wpisy: image -> obraz, label -> Identyfikator klasy, name -> nazwę zdjęcia wraz z rozszerzeniem .png,
size -> granice obiektu w pikselach
Wyjście: nic
Pokazywanie statystyk słownika