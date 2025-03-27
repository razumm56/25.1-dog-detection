from collections import defaultdict
from ultralytics import YOLO

import numpy as np
import cv2

# Загрузка модели YOLO
model = YOLO('weights/best_7000_1500.pt')


def process_video(video_path=0):
    # Открытие видео
    cap = cv2.VideoCapture(video_path)

    writer = ...  # Инициализация класса БД

    if not cap.isOpened():
        print(f"Ошибка открытия {video_path}")
        exit()

    # Создание словаря для хранения истории треков объектов
    track_history = defaultdict(lambda: [])

    # Цикл для обработки каждого кадра видео
    while cap.isOpened():
        # Считывание кадра из видео
        success, frame = cap.read()

        if not success:
            print("Конец видео")
            break

        # Применение YOLOv8 для отслеживания объектов на кадре, с сохранением треков между кадрами
        results = model.track(frame, persist=True)
        classes_names = results[0].names

        # Проверка на наличие объектов
        if results[0].boxes is not None and results[0].boxes.id is not None:

            # Получение координат боксов и идентификаторов треков
            boxes = results[0].boxes.xywh.cpu()  # xywh координаты боксов
            classes = results[0].boxes.cls.cpu().numpy() # классы
            track_ids = results[0].boxes.id.int().cpu().tolist()  # идентификаторы треков

            # Визуализация результатов на кадре
            annotated_frame = results[0].plot()

            # Отрисовка треков
            for box, track_id, class_id in zip(boxes, track_ids, classes):
                x, y, w, h = box  # координаты центра и размеры бокса
                track = track_history[track_id]
                class_name = classes_names[int(class_id)]
                print(class_name)
                track.append((float(x), float(y)))  # добавление координат цента бокса

                # write.add_detection(({track_id: (float(x - w//2), float(y - h//2), float(x + w//2), float(y + h//2), class_name)}) Добавление меток в базу данных

                if len(track) > 5:  # ограничение длины истории до 30 кадров
                    track.pop(0)

                # Рисование линий трека
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Отображение аннотированного кадра
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
        else:
            # Если объекты не обнаружены, просто отображаем кадр
            cv2.imshow("YOLOv8 Tracking", frame)

        # Прерывание цикла при нажатии клавиши 'Esc'
        if cv2.waitKey(1) == 27:
            break

    # Освобождение видеозахвата и закрытие всех окон OpenCV
    cap.release()
    cv2.destroyAllWindows()

    return track_history


track = process_video("test_data/input.mp4")

print(track)
