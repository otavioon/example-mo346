from datetime import datetime
from typing import Any
from base import Configuration, evaluate_model
from har_model import HAR_Downstream


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
    "save_dir": "logs/evaluate/",
    "name": "mlp",
    "version": datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
    "epochs": 1,
    "accelerator": "cpu",
    "monitor": None,
    "mode": "min",
}

backbone_ckpt_path = None
resume_ckpt_path = "/workspaces/hiaac-m4/example-mo346/logs/downstream/mlp/18-04-2024_02-45-37/checkpoints/epoch=0-step=2.ckpt"


def main():
    config = HAR_Downstream()
    evaluate_model(
        config,
        model_kwargs,
        data_module_kwargs,
        trainer_kwargs,
        resume_ckpt_path=resume_ckpt_path,
    )


if __name__ == "__main__":
    main()
