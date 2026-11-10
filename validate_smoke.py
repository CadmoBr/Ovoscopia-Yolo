from ultralytics import YOLO

model = YOLO('yolov8n.pt')
print('ultralytics_ok')

model.train(
    data='data.yaml',
    epochs=1,
    imgsz=640,
    batch=2,
    project='runs/detect',
    name='smoke_test',
    exist_ok=True,
    workers=0,
    verbose=False,
)
print('train_ok')

model.predict(
    source='ovo.jpeg',
    conf=0.25,
    save=True,
    project='runs/detect',
    name='smoke_predict',
    verbose=False,
)
print('predict_ok')
