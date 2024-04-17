# Выпускная квалификационная работа 

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
- ```--output_directory "path/to/output"``` путь где будет сохраняться модель и логи
- ```start_training_from_model True or False``` начать обучение с существующей модели лежащей в output_directory 
