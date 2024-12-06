# PRP - Passport Recognition Project

проект распознавания изображений паспортов

## Задачи

- написать скрипт: скачивание фотографий по запросу "паспорт РФ"
- реализовать распознавание изображений по признаку "есть паспорт" или "нет паспорта"

## Реализация

### Скрипт-скрапер

#### Инструменты

- python
- selenium

#### Описание

- Датасет состоит из 100 изображений по запросу 'паспорт рф' и 100 изображений по запросу 'картинки' в dataset/val
- Для обучения модели датасет разбивается на 20+20 в val и 80+80 в train

### Распознавание

#### Инструменты

- yoloV11
- LabelImg

#### Fixing issues with labelImg:

- cd \venv\Lib\site-packages
- git clone https://github.com/HumanSignal/labelImg.git
- pip install -e labelImg
- pyrcc5 -o libs/resources.py resources.qrc

#### Разметка тренировочного датасета в labelImg:

![alt text](image.png)

#### Yolo model

##### Подготовка данных

Соотношение train-validation должно быть 80%-20% причём val - объекты которые модель не видела при обучении.

- Структура каталогов dataset перед обучением модели:
    ```
    dataset/
        train/
            images/
                img1.jpg
                img2.jpg
                ...
            labels/
                img1.txt
                img2.txt
                ...
        val/
            images/
                img3.jpg
                ...
            labels/
                img3.txt
                ...
    ```

- Файл `classes.txt`, который содержит названия классов разметки.
    ```
    passport, cover, other
    ```

#### Обучение модели YOLO

После создания структуры датасета и разметки, можно приступать к обучению модели:
Перед обучением необходимо перевести разметку XML в формат yolo

#### Скачивание предобученных весов YOLO

Для данного проекта используется YOLOV11

#### Настройка конфигурации 

   Для обучения модели необходимо создать файл `config.yaml`

Training a YOLO (You Only Look Once) model using a GPU can significantly accelerate your training process due to the parallel processing capabilities of GPUs. Below are detailed steps on how to set up your environment and train a YOLO model using GPU resources.

#### Install Required Software
   - Ensure you have Python installed (preferably version 3.6 to 3.9).
   - Install PyTorch with GPU support. You can find the appropriate installation command for your system configuration (CUDA version) on the [official PyTorch website](https://pytorch.org/get-started/locally/).

   Example command:
   ```bash
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
   ```

### Monitor Training

- During training, you can monitor the loss metrics and performance visually through the command-line output, as well as through saved tensorboard logs.
- You can set up TensorBoard to visualize the training process:
   ```bash
   tensorboard --logdir runs/train
   ```
   Then visit `http://localhost:6006/` in your web browser.

#### Тестирование и оценка модели

После завершения обучения вы можете протестировать модель на новых изображениях, чтобы убедиться, что она распознает паспорта.