from collections import defaultdict, Counter
from datetime import datetime

from ultralytics import YOLO

import numpy as np
import cv2
import asyncio

from dataset.SQLAlchemy_tabels import Base
from dataset.SQLALchemy_utils import PostgresConnection
from dataset.DBWriter import DBWriter, TableInsertScheme

from bot.bot import DogAlertBot



IMG_SIZE = 1280 # Оптимальное разрешение для баланса скорости/точности

connection = PostgresConnection(
    host="91.239.206.123", port="29923", user="StrayDogDetectionDB", password="12345678"
)
connection.delete_tables(["label"])
connection.delete_tables(["frame"])
connection.delete_tables(["video"])
connection.create_tables(Base)




def initialization_database():
    db_writer = DBWriter(connection)

    db_video_scheme = TableInsertScheme(
        "video", columns=["id", "path"]
    )  # Инициализация класса схемы video
    db_frame_scheme = TableInsertScheme(
        "frame", columns=["id", "video_id", "timestamp", "dogs_num"]
    )  # Инициализация схемы frame
    db_label_scheme = TableInsertScheme(
        "label", columns=["frame_id", "x1", "y1", "x2", "y2", "class_name"]
    )

    db_writer.add_scheme(db_video_scheme)
    db_writer.add_scheme(db_frame_scheme)
    db_writer.add_scheme(db_label_scheme)

    db_writer.start()

    return db_writer, db_video_scheme, db_frame_scheme, db_label_scheme


async def process_video(model_path, video_path=0):
    # Загрузка модели YOLO
    model = YOLO(model_path).to('cuda')

    # Открытие видео
    cap = cv2.VideoCapture(video_path)

    db_writer, db_video_scheme, db_frame_scheme, db_label_scheme = initialization_database()

    db_video_scheme.queue.put([0, video_path])

    bot = DogAlertBot(TOKEN, CHAT_ID)

    if not cap.isOpened():
        print(f"Ошибка открытия {video_path}")
        exit()

    # Создание словаря для хранения истории треков объектов
    track_history_coords  = defaultdict(lambda: [])
    track_history_classes  = defaultdict(lambda: [])

    detected_persons = set()

    frame_counter = 0

    # Цикл для обработки каждого кадра видео
    while cap.isOpened():

        # Считывание кадра из видео
        success, frame = cap.read()

        if not success:
            print("Конец видео")
            break

        # Применение YOLOv8 для отслеживания объектов на кадре, с сохранением треков между кадрами
        results = model.track(
            frame,
            persist=True,
            imgsz=IMG_SIZE,
            conf=0.7,  # Порог уверенности
            iou=0.1,  # Порог IoU для NMS
            half=True,  # Half-precision
            verbose=False  # Отключение логов
        )
        classes_names = results[0].names

        # Проверка на наличие объектов
        if results[0].boxes is not None and results[0].boxes.id is not None:
            # Получение координат боксов и идентификаторов треков
            boxes = results[0].boxes.xywh # xywh координаты боксов
            classes = results[0].boxes.cls.numpy()  # классы
            track_ids = (
                results[0].boxes.id.int().tolist()
            )  # идентификаторы треков

            # Визуализация результатов на кадре
            annotated_frame = results[0].plot()

            counter_objects = Counter()

            # Отрисовка треков
            for box, track_id, class_id in zip(boxes, track_ids, classes):
                x, y, w, h = box  # координаты центра и размеры бокса
                track_coord = track_history_coords[track_id]
                track_class = track_history_classes[track_id]

                class_name = classes_names[int(class_id)]

                if len(track_class) >= 5:
                    last_classes = track_history_classes[track_id][-5:]
                    if last_classes.count("person") > 3:
                        class_name = "person"
                    elif last_classes.count("dog") > 3:
                        class_name = "dog"

                track_coord.append((float(x), float(y)))  # добавление координат цента бокса
                track_class.append(class_name)

                    # Ограничение длины истории
                if len(track_coord) > 30:
                    track_coord.pop(0)
                if len(track_class) > 30:
                    track_class.pop(0)

                # Рисование линий трека
                points = np.hstack(track_coord).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    annotated_frame,
                    [points],
                    isClosed=False,
                    color=(230, 230, 230),
                    thickness=10,
                )
                db_label_scheme.queue.put([frame_counter, x, y, w - x, h - y, class_name])

                # Проверяем, что это собака с высокой уверенностью
                if (
                        len(track_class) >= 10
                        and track_class.count("dog") >= 7
                        and track_class[-5:].count("dog") >= 3  # В последних 5 кадрах 3+ раза
                ):
                    # Если этот человек (track_id) еще не был зарегистрирован
                    if track_id not in detected_persons:
                        detected_persons.add(track_id)  # Добавляем в множество
                        counter_objects['dog'] += 1  # Увеличиваем счетчик
                        await bot.push_notificate(counter_objects['dog'], frame=annotated_frame)  # Отправляем уведомление

            db_frame_scheme.queue.put([frame_counter, 0, datetime.now().timestamp(), counter_objects['dog']])

            # Отображение аннотированного кадра
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
        else:
            # Если объекты не обнаружены, просто отображаем кадр
            cv2.imshow("YOLOv8 Tracking", frame)

        # Прерывание цикла при нажатии клавиши 'Esc'
        if cv2.waitKey(1) == 27:
            break

        frame_counter += 1

    # Освобождение видеозахвата и закрытие всех окон OpenCV
    cap.release()
    cv2.destroyAllWindows()
    # db_writer.stop()

if __name__ == '__main__':
    asyncio.run(
    process_video(
        r"C:\Users\Максим\PycharmProjects\dog_detection\src\weights\best_big.pt",
        "https://video2.interra.ru/glaz.naroda.128-0d7b2d2867/tracks-v1/mono.ts.m3u8?token=3.9CzUU5u-AAAAAAAAAEsAAAAAAAAAAJW2kv1Ovhjy7IOYANd4WdGStdzH",
    ))
