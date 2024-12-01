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

#### Скачивание предобученных весов YOLO

Для данного проекта используется YOLOV5

#### Установка необходимых библиотек

    ```bash
    git clone https://github.com/ultralytics/yolov5.git
    cd yolov5
    pip install -r requirements.txt
    ```

#### Настройка конфигурации 

   Для обучения модели необходимо создать файл `config.yaml`

#### Обучение модели
   файл main.ipynb для запуска обучения модели
   Параметры обучения модели:
   
   Используйте следующую команду для запуска процесса обучения:
   ```bash
   python train.py --batch 16 --epochs 50 --data config.yaml --weights yolov5s.pt
   ```

   - `--img`: Размер изображений, на которых вы будете обучаться.
   - `--batch`: Размер пакета.
   - `--epochs`: Количество эпох обучения.
   - `--data`: Укажите путь к вашему `.yaml` файлу.
   - `--weights`: Используйте предобученные веса. Вы можете выбрать различные версии модели (например, `yolov5s.pt`, `yolov5m.pt` и т.д.).

Training a YOLO (You Only Look Once) model using a GPU can significantly accelerate your training process due to the parallel processing capabilities of GPUs. Below are detailed steps on how to set up your environment and train a YOLO model using GPU resources.

#### Install Required Software
   - Ensure you have Python installed (preferably version 3.6 to 3.9).
   - Install PyTorch with GPU support. You can find the appropriate installation command for your system configuration (CUDA version) on the [official PyTorch website](https://pytorch.org/get-started/locally/).

   Example command:
   ```bash
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
   ```

#### Train Command
   - Use the following command to train the YOLO model on your dataset using GPU. Adjust parameters as necessary.

   Example training command for YOLOv5:

   ```bash
   python train.py --batch 16 --epochs 50 --data config.yaml --weights yolov5s.pt --device 0
   ```

   - **Parameters**:
     - `--img`: Specify the image size for training (640 is commonly used).
     - `--batch`: Set the batch size. Depending on your GPU, you may need to adjust this to avoid running out of memory.
     - `--epochs`: Total number of training epochs.
     - `--data`: Path to the dataset YAML file.
     - `--weights`: Use pre-trained weights (for transfer learning). You can replace `yolov5s.pt` with other model weights (`yolov5m.pt`, `yolov5l.pt`, etc.) as needed.
     - `--device`: Specify the GPU device; `0` usually refers to the first GPU. Omit this or set to `cpu` to train on CPU.

### Monitor Training

- During training, you can monitor the loss metrics and performance visually through the command-line output, as well as through saved tensorboard logs.
- You can set up TensorBoard to visualize the training process:
   ```bash
   tensorboard --logdir runs/train
   ```
   Then visit `http://localhost:6006/` in your web browser.

#### Тестирование и оценка модели

После завершения обучения вы можете протестировать модель на новых изображениях, чтобы убедиться, что она распознает паспорта.
Для этого запустите второй блок main.ipynb указав путь к файлу для распознавания.

#### Использование обученной модели

Как только вы получили модель, вы можете использовать `detect.py` для распознавания объектов в новых изображениях.
второй блок файла main.ipynb для распознавания на обученной модели:
```bash
python detect.py --weights runs/train/exp/weights/best.pt --conf 0.25 --source path_to_your_image.jpg
```