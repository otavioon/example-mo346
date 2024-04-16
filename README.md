# Example-MO346

This is an example of a MO346 project. This project has the following structure:

- `configs/`: contains the configuration files to instantiate models, data modules, and trainers. It is used by `LightningCLI` to instantiate experiments. The configuration files are written in YAML format and contain the class path (package and class name) and parameters to instantiate the objects.
- `data/`: contains the data used in the project. Your data should be stored here. This directory is not tracked by git.
- `data_modules/`: contains implementations of `LightningDataModule` classes. These classes are responsible for loading and preprocessing data, and also for splitting the data into training, validation, and test sets.
- `models/`: contains implementations of `LightningModule` classes. These classes are responsible for defining the model architecture and implementing the `forward`, `training_step`, and `configure_optimizers`  methods.
- `transforms/`: contains implementations of Numpy/PyTorch transforms. These transforms are used to preprocess data before feeding it to the model.

The `main.py` file is the entry point of the project. 
It is a CLI that is responsible for instantiating the trainer, model, and data module, and also performing the training and testing of the model.
Configuration files are used to instantiate each of these objects. 
The configuration files are stored in the `configs/` directory.

For instance, to train and test the model, you can run the following command:

```bash
python main.py --trainer configs/trainer/default.yaml --model configs/models/mlp.yaml --data configs/data_modules/har.yaml
```

The `--trainer` contains the class and parameters for instantiating a `Lightning.Trainer` object.
The `--model` contains the name of the class and parameters for instantiating a `LightningModule` object.
The `--data` contains the name of the class and parameters for instantiating a `LightningDataModule` object.
Output files are stored in the `logs/` directory. This directory is not tracked by git.

The trainer will use standard `fit` and `test` methods from the `LightningModule` class to train and test the model. 
The `fit` method is used to train the model, and the `test` method is used to test the model.


## Installation


We use VSCode Containers to run the project. To run the project, you need to have Docker and VSCode installed on your machine.
If you don't know how to use Containers in VSCode, you can follow the instructions in the [following link](https://github.com/otavioon/container-workspace).

Once inside the container, you can install the dependencies, running the following command:

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
Once you have created the new files, you can create new configuration files in the `configs/` directory.

## Authors

- [Otávio Napoli](https://github.com/otavioon)

## License

This project is licensed under GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

