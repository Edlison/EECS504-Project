from transformers import DetrImageProcessor, DetrForObjectDetection
import os
import detr
from detr import register_coco_instances, add_detr_config, Trainer

register_coco_instances("custom_train",
                        {},
                        "coco-dataset/annotations/custom_train.json",
                        "coco-dataset/train/")

register_coco_instances("custom_val",
                        {},
                        "coco-dataset/annotations/custom_val.json",
                        "coco-dataset/val/")

cfg = detr.get_cfg()

add_detr_config(cfg)
cfg.merge_from_file("detr_256_6_6_torchvision.yaml")

cfg.DATASETS.TRAIN = ("custom_train",)
cfg.DATASETS.TEST = ("custom_val",)
cfg.OUTPUT_DIR = 'outputs/'
cfg.MODEL.WEIGHTS = "converted_model.pth"

cfg.MODEL.DETR.NUM_CLASSES = 1
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025 
cfg.SOLVER.MAX_ITER = 300   
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = Trainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
     