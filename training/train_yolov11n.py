from ultralytics import YOLO

# Initialize model
model = YOLO('yolov11n.yaml') #load no pretrained weights

#configure model
configuration = dict(
  data = 'path/to/data',
  imgsz = 640,
  batch_size = 16,
  epochs = 50,
  device = '0',
  optimizer = 'AdamW',
  patience = 80,
  save=True,
  project='Yolov11n-pre'
)

#train with gpu on custom dataset
results = model.train(
  data=configuration["data"],
  epochs=configuration["epochs"],
  batch_size=configuration["batch_size"],
  imgsz=configuration["imgsz"],
  device=configuration["device"],
  save=configuration["save"],
  project=configuration["project"],
  optimizer=configuration["optimizer"],
  patience=configuration["patience"]
)
