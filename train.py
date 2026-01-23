from config import config
from dataset.kitti import KittiDataset
from model.vic_model import ViCModel
from utils.trainer import Trainer

train_dataset = KittiDataset(config.kitti_train_config)
val_dataset = KittiDataset (config.kitti_val_config)

model = ViCModel(config.vic_config)
trainer = Trainer(model, train_dataset, val_dataset,batch_size=2)

trainer.train(10)
pass