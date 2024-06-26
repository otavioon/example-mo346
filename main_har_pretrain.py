from datetime import datetime
from typing import Any
from base import Configuration, train_model
from har_model import HAR_PreTrain, HAR_Downstream


# Model configuration
model_kwargs = {
    "input_features": 360,
    "hidden_size": 64,
    "num_classes": 6,
    "learning_rate": 0.001,
}

data_module_kwargs = {
    "path": "/workspaces/hiaac-m4/example-mo346/data/example/"
}

trainer_kwargs = {
    "save_dir": "logs/pretrain/",
    "name": "mlp",
    "version": datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
    "epochs": 1,
    "accelerator": "cpu",
    "monitor": None,
    "mode": "min",
}

backbone_ckpt_path = None
resume_ckpt_path = None

def main():
    config = HAR_PreTrain()
    train_model(config, model_kwargs, data_module_kwargs, trainer_kwargs)

if __name__ == "__main__":
    main()