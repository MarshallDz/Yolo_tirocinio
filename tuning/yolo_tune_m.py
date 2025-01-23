from ultralytics import YOLO

model = YOLO("/home/user/yolo/runs/detect/yolo-m_superiore/weights/best.pt")

# Define search space
# Definiamo la lista dei parametri per il tuning con gli iperparametri
search_space = {
    "lr0": (0.001, 0.1),
    "lrf": (0.01, 1.0),
    "weight_decay": (0.0, 0.001),
    "hsv_h": (0.0, 0.1),
    "hsv_s": (0.0, 0.9),
    "hsv_v": (0.0, 0.9),
    "scale": (0.0, 0.9),
    "degrees": (0.0, 45.0)
}

#tuning con il search space 
model.tune(
    data="/home/user/yolo/composystemSuperioreV3_flip_agumentation/data.yaml",
    space=search_space,
    epochs=30,
    patience=80,
    iterations=50,
    warmup_epochs=0,
    project="yolo_tune_m",
    pretrained=True,
    batch=-1,
    optimizer="AdamW",
    amp=True
)