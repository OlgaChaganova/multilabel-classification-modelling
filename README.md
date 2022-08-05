# cvr-hw1-modeling 


### 1 - Подготовить среду для обучения

- dev-requirements
- test-requirements


### 2 - Скачать данные с Kaggle (для linux)

1. Присоединиться к [соревнованию](https://www.kaggle.com/competitions/planet-understanding-the-amazon-from-space/overview) 
и принять все правила;


2. На вкладке Data скачать файлы `train-jpg.tar.7z`, `test-jpg.tar.7z`, `train_v2.csv`


4. Выполнить в терминале `make download_data`, чтобы скачать архив; 


5. Распаковать архив:

```angular2html
7za x train-jpg.tar.7z
tar -xvkf train-jpg.tar
rm train-jpg.tar train-jpg.tar.7z

7za x test-jpg.tar.7z
tar -xvkf test-jpg.tar
rm test-jpg.tar test-jpg.tar.7z

unzip train_v2.csv.zip
rm train_v2.csv.zip
```

