import roboflow

rf = roboflow.Roboflow(api_key="")
project = rf.workspace("504-public-1tecx").project("boat-detection-bzj6m")
project.version(1).deploy("yolov8", "best.pt")