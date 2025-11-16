# YOLOv5 Object Detection

Реализация системы детектирования объектов на основе YOLOv5 (You Only Look Once version 5).

## Описание

Проект предоставляет инструменты для обнаружения и классификации объектов в реальном времени с использованием архитектуры YOLOv5. Поддерживается обработка изображений, видеопотоков и видеофайлов.

## Требования

- Python 3.7+
- PyTorch 1.7+
- CUDA 10.2+ (опционально, для GPU ускорения)

## Установка

```bash
# Клонирование репозитория
git clone https://github.com/SalahidinAI/Yolo5ObjectDetection.git
cd Yolo5ObjectDetection

# Установка зависимостей
pip install -r requirements.txt
```

## Использование

### Детектирование на изображении
```bash
python detect.py --source image.jpg --weights yolov5s.pt --conf 0.25
```

### Детектирование на видео
```bash
python detect.py --source video.mp4 --weights yolov5s.pt --conf 0.25
```

### Детектирование с веб-камеры
```bash
python detect.py --source 0 --weights yolov5s.pt --conf 0.25
```

### Использование в коде
```python
import torch

# Загрузка модели
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Детектирование
results = model('image.jpg')

# Вывод результатов
results.print()
results.show()
results.save()
```

## Структура проекта

```
Yolo5ObjectDetection/
├── models/          # Архитектура моделей
├── weights/         # Веса предобученных моделей
├── data/            # Конфигурации датасетов
├── utils/           # Вспомогательные функции
├── detect.py        # Скрипт детектирования
├── train.py         # Скрипт обучения
└── requirements.txt # Зависимости проекта
```

## Доступные модели

| Модель | Размер | mAP⁵⁰ | mAP⁵⁰-⁹⁵ | Параметры | FPS (V100) |
|--------|--------|-------|----------|-----------|------------|
| YOLOv5n | 3.9MB | 45.7 | 28.0 | 1.9M | 455 |
| YOLOv5s | 14.4MB | 56.8 | 37.4 | 7.2M | 278 |
| YOLOv5m | 42.2MB | 64.1 | 45.4 | 21.2M | 165 |
| YOLOv5l | 92.8MB | 67.3 | 49.0 | 46.5M | 127 |
| YOLOv5x | 173.1MB | 68.9 | 50.7 | 86.7M | 97 |

## Параметры командной строки

```
--source        # Путь к источнику (изображение, видео, камера)
--weights       # Путь к весам модели
--conf          # Порог уверенности (default: 0.25)
--iou           # Порог IoU для NMS (default: 0.45)
--imgsz         # Размер входного изображения (default: 640)
--device        # Устройство cuda:0 или cpu (default: '')
--save-txt      # Сохранить результаты в txt
--save-conf     # Сохранить уверенность в результатах
--classes       # Фильтр по классам (например: --classes 0 2 3)
--agnostic-nms  # NMS без учета классов
--augment       # Аугментированное детектирование
--project       # Папка для сохранения результатов
--name          # Имя эксперимента
```

## Обучение на собственных данных

### Подготовка датасета
```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### Запуск обучения
```bash
python train.py --data custom.yaml --weights yolov5s.pt --epochs 100 --batch-size 16
```

## Классы объектов

Модель обучена на датасете COCO и способна детектировать 80 классов объектов:
- Транспорт: автомобиль, мотоцикл, самолет, автобус, поезд, грузовик
- Животные: кошка, собака, лошадь, овца, корова, слон, медведь, зебра, жираф
- Предметы: рюкзак, зонт, сумка, галстук, чемодан
- Спорт: фрисби, лыжи, сноуборд, мяч, воздушный змей, бейсбольная бита
- И другие категории

## Экспорт модели

```bash
# ONNX
python export.py --weights yolov5s.pt --include onnx

# TensorFlow
python export.py --weights yolov5s.pt --include pb

# TFLite
python export.py --weights yolov5s.pt --include tflite

# CoreML
python export.py --weights yolov5s.pt --include coreml
```

## Производительность

Тестирование на COCO val2017:
- Размер изображения: 640x640
- Batch size: 32
- GPU: Tesla V100

## Лицензия

GNU General Public License v3.0

## Автор

[SalahidinAI](https://github.com/SalahidinAI)

## Ссылки

- [Официальный репозиторий YOLOv5](https://github.com/ultralytics/yolov5)
- [Документация](https://docs.ultralytics.com/)
- [Датасет COCO](https://cocodataset.org/)
