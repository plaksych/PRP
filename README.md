# PRP - Passport Recognition Project

проект распознавания изображений паспортов

## Задачи

- написать скрипт: скачивание фотографий по запросу "паспорт РФ"
- реализовать распознавание изображений по признаку "есть паспорт" или "нет паспорта"
- реализовать хранение обработанных данных с признаком

## Реализация

### Скрипт-скрапер

#### Инструменты

- python
- selenium

#### Описание

- Датасет состоит из 100 изображений по запросу 'паспорт рф' и 100 изображений по запросу 'картинки' в dataset/val
- Для обучения модели отбирается половина изображений 50 c Sпаспортом или обложкой и 50 картинок в dataset/train

### Распознавание

#### Инструменты

- yolo model recognise
- LabelImg

#### Fixing issues with labelImg:

- cd \venv\Lib\site-packages
- git clone https://github.com/HumanSignal/labelImg.git
- pip install -e labelImg
- pyrcc5 -o libs/resources.py resources.qrc

#### Разметка тренировочного датасета в labelImg:

![alt text](image.png)

### Хранение 

- docker
- postreSQL
- dataset.csv