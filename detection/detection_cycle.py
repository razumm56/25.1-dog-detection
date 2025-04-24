from collections import defaultdict, Counter
from datetime import datetime

from ultralytics import YOLO

import numpy as np
import cv2

from dataset.SQLAlchemy_tabels import Base
from dataset.SQLAlchemy_utils import PostgresConnection
from dataset.DBWriter import DBWriter, TableInsertScheme

connection = PostgresConnection(host="91.239.206.123", port="29923", user="StrayDogDetectionDB", password="12345678")
connection.delete_tables(["label"])
connection.delete_tables(["frame"])
connection.delete_tables(["video"])
connection.create_tables(Base)

def process_video(model_path, video_path=0):
    print(1)
    # Загрузка модели YOLO
    model = YOLO(model_path)

    # Открытие видео
    cap = cv2.VideoCapture(video_path)

    db_writer = DBWriter(connection)
    db_video_scheme = TableInsertScheme("video", columns=["id", "path"])  # Инициализация класса БД
    db_frame_scheme = TableInsertScheme( "frame", columns=["video_id", "timestamp", "dogs_num"])  # Инициализация класса БД

    db_writer.add_scheme(db_video_scheme)
    db_writer.add_scheme(db_frame_scheme)

    db_writer.start()

    db_video_scheme.queue.put([0, video_path])


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

            counter_objects = Counter()

            # Отрисовка треков
            for box, track_id, class_id in zip(boxes, track_ids, classes):
                x, y, w, h = box  # координаты центра и размеры бокса
                track = track_history[track_id]
                class_name = classes_names[int(class_id)]
                counter_objects[class_name] += 1
                track.append((float(x), float(y)))  # добавление координат цента бокса

                # write.add_detection(({track_id: (float(x - w//2), float(y - h//2), float(x + w//2), float(y + h//2), class_name)}) Добавление меток в базу данных
                print({track_id: (float(x - w//2), float(y - h//2), float(x + w//2), float(y + h//2), class_name)})
                if len(track) > 30:  # ограничение длины истории до 30 кадров
                    track.pop(0)

                # Рисование линий трека
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            db_frame_scheme.queue.put([0, datetime.now().timestamp(), counter_objects['dog']])


            print(f"Количество собак на фото: {counter_objects['dog']}")

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
    db_writer.stop()
    return track_history


track = process_video('weights/best_2.pt', "test_data/input.mp4")

