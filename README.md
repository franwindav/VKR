# Выпускная квалификационная работа 

Данная работа была выполнена на linux и на других системах не испытывалась.

## Конфигурация среды для conda
Для создания среды необходимо выполнить команду:
```
    conda env create -f environment.yml
```
Для того чтобы экспортировать нужно выполнить команду:
```
    conda env export | grep -v "^prefix: " > environment.yml
```
Для активации среды:
```
    source activate ./cenv
```

## CTC
Данная модель МО использует CTC в качестве декодера.

Можно использовать две библиотеки:
- ```ctcdecode``` 

В моей работе используетсе данная библиотека.

Для ее установки в ручном режиме необходимо выполнить команды:
```
    git clone --recursive https://github.com/parlance/ctcdecode.git
    cd ctcdecode && pip install .
```
Примечание: данный пакет невозможно собрать на платформе Windows

Для установки в атоматическом режиме:
```
    pip install ctcdecode 
```

Возможны проблемы с ее использованием в google colab. 

- ```pyctcdecode``` 

Также возможно использовать и данную библиотеку, однако необходимо немного поменять файл ```recognition_model.py```
Для её установки необходимо выполнить команды:
```
    pip install pyctcdecode kenlm
```

## Установка датасета и языковой модели
Необходимо выполнить команду для скачивания датасета:
```
    wget "https://zenodo.org/records/4064409/files/emg_data.tar.gz?download=1"
    mv "./emg_data.tar.gz?download=1" ./emg_data.tar.gz
    tar -xvf ./emg_data.tar.gz
```
Также необходимо скачать языковую модель:
```
    wget "https://github.com/mozilla/DeepSpeech/releases/download/v0.6.1/lm.binary"
```

## Обучение модели
Выполнение скрипта:
```
    python ./recognition_model.py 
```

Основные ключи:
- ```--output_directory "path/to/output"```

    путь где будет сохраняться модель и логи

- ```--start_training_from_model True or False```

    начать обучение с существующей модели лежащей в ```output_directory``` 

## Тестирование модели
Для тестирования модели необходимо запустить скрипт:
```
    python recognition_model.py --evaluate_saved "path/to/model.pt"
```
