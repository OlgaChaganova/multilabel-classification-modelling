# cvr-hw1-modeling 


## 1 - Подготовить среду для обучения

1. Создать виртуальное окружение (можно любым удобным способом, но нужен python 3.10)

```shell
conda create --name <venv_name> python=3.10
conda activate <venv_name>
```

Могут возникнуть проблемы с активацией окружения -- тогда выполнить:

```shell
source /opt/conda/etc/profile.d/conda.sh
```

2. Установить необходимые зависимости: 

```shell
make install
```

## 2 - Скачать данные с Kaggle

1. Присоединиться к [соревнованию](https://www.kaggle.com/competitions/planet-understanding-the-amazon-from-space/overview) 
и принять все правила;


2. На вкладке Data скачать файлы `train-jpg.tar.7z`, `test-jpg.tar.7z`, `train_v2.csv` и положить их в корень репозитория,
в папку `raw_data`.


4. Распаковать архив и удалить ненужные файлы:

```shell
make prepare_data
```


## 3 - Запустить обучение

1. Настроить конфигурацию эксперимента в файле `src/configs/config.py`;


2. Запустить обучение:

```shell
cd src
python train_model.py --config configs/config.py
```



## 4 - Ссылки на эксперимент

Для сравнения обучены две модели -- DenseNet121 (`src/configs/densenet121_config.py`) и 
MobileNet_v3 (`src/configs/mobilenet_config.py`)

Метрики и артефакты для [DenseNet121](https://app.clear.ml/projects/be78acda989c46ea965eab2c46b0e170/experiments/58a688c06e414d479c3efacc592cff31/output/execution)

Метрики и артефакты для [MobileNet_v3](https://app.clear.ml/projects/be78acda989c46ea965eab2c46b0e170/experiments/e60ede3ea8c441aaa6dd84902a96fd1f/output/execution)



## 5 - DVC и инференс модели

В папке `checkpoints` лежат dvc-файлы полных чекпоинтов обучения, которые могут пригодиться, если понадобится
возобновить обучение. Для инференса в сервисе они не используются.

В папке `weights` лежат dvc-файлы для весов моделей. Для их получения необходимо выполнить команду

```shell
dvc pull <model_name>.pt.dvc
```

Предварительно необходимо настроить подключение dvc к удаленному серверу:

```shell
pip install dvc[ssh]

dvc remote add --default dvc_remote_staging ssh://91.206.15.25/home/oliyyaa/dvc_files
dvc remote modify dvc_remote_staging user oliyyaa
dvc config cache.type hardlink,symlink

dvc remote modify dvc_remote_staging keyfile /home/.ssh/id_rsa
```

Модели скомпилированы в формат `TorchScript`. Для их инференса достаточно загрузить веса (размеры входных тензоров должны
совпадать с размерами тензоров, которые использовались при обучении:
`[batch_size, config.dataset.num_channels, config.dataset.img_size, config.dataset.img_size]`):

```python
...
model = torch.jit.load(model_path, map_location='cpu')
probabilities = model(imgs)
```

Также в папке `weights` лежит .npy файл с классами label encoder. Инициализировать его можно следующим образом:

```python
import numpy as np
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.classes_ = np.load('weights/label_encoder_classes.npy')
```