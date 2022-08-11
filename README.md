# cvr-hw1-modeling 


### 1 - Подготовить среду для обучения

```angular2html
conda create --name <venv_name> python=3.10
conda activate <venv_name>
```

Могут возникнуть проблемы с активацией окружения -- тогда выполнить:

```angular2html
source /opt/conda/etc/profile.d/conda.sh
```

### 2 - Скачать данные с Kaggle (для linux)

1. Присоединиться к [соревнованию](https://www.kaggle.com/competitions/planet-understanding-the-amazon-from-space/overview) 
и принять все правила;


2. На вкладке Data скачать файлы `train-jpg.tar.7z`, `test-jpg.tar.7z`, `train_v2.csv` и положить их в корень репозитория,
в папку `raw_data`.


4. Выполнить в терминале `make prepare_data`, чтобы распаковать архив и удалить ненужные файлы.


### 3 - Запустить обучение

1. Настроить конфигурацию эксперимента в файле `src/configs/config.py`;

2. Запустить обучение:

```angular2html
cd src
python train_model.py --config configs/config.py
```


### 4 - Ссылка на эксперимент

[DenseNet121](https://app.clear.ml/projects/be78acda989c46ea965eab2c46b0e170/experiments/58a688c06e414d479c3efacc592cff31/output/execution)

[MobileNet_v3](https://app.clear.ml/projects/be78acda989c46ea965eab2c46b0e170/experiments/e60ede3ea8c441aaa6dd84902a96fd1f/output/execution)