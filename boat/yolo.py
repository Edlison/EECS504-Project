from ultralytics import YOLO
import roboflow


# Data augmentation is done in Roboflow,
# so here we just need to download the processed dataset

rf = roboflow.Roboflow(api_key="")
project = rf.workspace("504-public-nx7jv").project("satellite-boat-detection")
dataset = project.version(3).download("yolov8", location="yolo-datasets")

yolo = YOLO('yolov8x.pt')
results = yolo.train(
    data=f'yolo-datasets/data.yaml', 
    epochs=100,
    imgsz=640,
    batch=16
) 

valid_results = yolo.val()
print(valid_results)