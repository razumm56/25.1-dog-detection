import base64
import json
import os
import shutil
import tempfile
import threading
from typing import Any

import cv2
import numpy as np
from cv2.typing import MatLike
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
from ultralytics import YOLO

app = FastAPI()

model_path = "./weights/best_30.05.2025.pt"
default_conf = 0.25


def process_frame(
    model: YOLO, frame: MatLike, image_size: int, conf: float, return_annotated: bool
) -> dict:
    results = model.track(
        frame, imgsz=image_size, conf=conf, persist=True, verbose=False
    )

    return_data: dict[str, Any] = {}
    return_data["detections"] = [result.summary() for result in results]

    if return_annotated and results:
        annotated_frame = results[0].plot()

        success, buffer = cv2.imencode(".jpg", annotated_frame)
        if not success:
            raise RuntimeError("Failed to encode image")

        return_data["image"] = base64.b64encode(buffer).decode("utf-8")

    return return_data


@app.get("/test")
def test():
    return {"Hello", "FastAPI"}


frame_model = YOLO(model_path)


@app.post("/image")
async def decoded_image(
    frame: UploadFile,
    image_size: int,
    conf: float = default_conf,
    return_annotated: bool = False,
):
    nparr = np.frombuffer(await frame.read(), np.uint8)
    decoded_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if decoded_image is None:
        return JSONResponse({"message": "Image is invalid"}, status_code=400)

    results = process_frame(
        frame_model, decoded_image, image_size, conf, return_annotated
    )

    return JSONResponse(content=results)


video_model = YOLO(model_path)


@app.post("/video")
async def video(
    file: UploadFile,
    image_size: int,
    conf: float = default_conf,
    return_annotated: bool = False,
):
    # Сохраняем входное видео во временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input_path = temp_input.name
        shutil.copyfileobj(file.file, temp_input)

    cap = cv2.VideoCapture(temp_input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if return_annotated:
        annotated_video_path = f"{temp_input_path}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))
    else:
        writer = None

    all_detections = []
    frame_id = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = video_model.predict(frame, imgsz=image_size, conf=conf, verbose=False)
        detections = [r.summary() for r in results]
        all_detections.append({"frame": frame_id, "detections": detections})

        if writer:
            annotated_frame = results[0].plot()
            writer.write(annotated_frame)

        frame_id += 1

    cap.release()
    if writer:
        writer.release()

    os.remove(temp_input.name)

    if return_annotated:
        return FileResponse(
            annotated_video_path,
            media_type="video/mp4",
            filename="annotated_video.mp4",
            headers={"X-Detections-Info": "Full detections in JSON body"},
            background=BackgroundTask(lambda: os.remove(annotated_video_path)),
        )

    return JSONResponse(content={"results": all_detections})


@app.get("/url_stream")
def url_stream(
    request: Request,
    url: str,
    image_size: int,
    conf: float = default_conf,
    annotate_frames=False,
):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(url)

    # в отдельном потоке скипаем все кадры которые не успеваем обработать
    def skip_frames(stop_event: threading.Event):
        while not stop_event.is_set() and cap.isOpened():
            cap.grab()

    stop_event = threading.Event()
    grab_thread = threading.Thread(target=skip_frames, args=(stop_event,))
    grab_thread.start()

    # обработка последнего доступного кадра
    async def generator():
        try:
            while True:
                if await request.is_disconnected():
                    break

                success, frame = cap.retrieve()

                if success:
                    detection_result = process_frame(
                        model, frame, image_size, conf, annotate_frames
                    )
                    detection_result["timestamp"] = cap.get(cv2.CAP_PROP_POS_MSEC)
                    detection_result["frame"] = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    yield (json.dumps(detection_result) + "\n")

        finally:
            stop_event.set()
            grab_thread.join()
            cap.release()

    return StreamingResponse(generator(), media_type="text/event-stream")
