# Example-MO346

This is an example of a MO346 project. This project has the following structure:

- `configs/`: contains the configuration files to instantiate models, data modules, and trainers. It is used by LightningCLI to instantiate experiments.
- `data/`: contains the data used in the project. Your data should be stored here. This directory is not tracked by git.
- `data_modules/`: contains implementations of LightningDataModule classes. These classes are responsible for loading and preprocessing data.
- `models`: contains implementations of LightningModule classes. These classes are responsible for defining the model architecture and the forward method.
- `transforms/`: contains implementations of PyTorch transforms. These transforms are used to preprocess data.

The `main.py` file is the entry point of the project. 
It is responsible for instantiating the trainer, model, and data module, and also performing the training and testing of the model.
Configuration files are used to instantiate these objects. The configuration files are stored in the `configs/` directory.

For instance, to train and test the model, you can run the following command:

```bash
python main.py --trainer configs/trainer/default.yaml --model configs/models/mlp.yaml --data configs/data_modules/har.yaml
```

The `--trainer` contains the class and parameters for instantiating a Lightning.Trainer object.
The `--model` contains the name of the class and parameters for instantiating a LightningModule object.
The `--data` contains the name of the class and parameters for instantiating a LightningDataModule object.
Output files are stored in the `logs/` directory. This directory is not tracked by git.

The trainer will use standard `fit` and `test` methods from the LightningModule class to train and test the model. 
The `fit` method is used to train the model, and the `test` method is used to test the model.


## Installation

To install the dependencies, you can run the following command:

```bash
pip install -r requirements.txt
```

## Running the tests

To run the tests, you can run the following command:

```bash
run.sh
```

## Adding new features

To add new features to the project, you can create new files in the `data_modules/`, `models/`, and `transforms/` directories.
You can also create new configuration files in the `configs/` directory.

## Authors

- [Otávio Napoli](https://github.com/otavioon)

## License

This project is licensed under GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

