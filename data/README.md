# Data

Data should be stored in the `data/` directory. This directory is not tracked by git, so you can store large files in it.
Remember to put your datasets here, and to add a `README.md` file with a description of the data.
Besides that, also remember to download data at LightningDataModule's `prepare_data` method (if the data is not already downloaded).


## Dataset: example

The dataset is composed of 3 files:
- `train.csv`: contains the training set
- `validation.csv`: contains the validation set
- `test.csv`: contains the test set

Each file contains time-series from accelerometers and gyroscope sensors of a smartphone and a `standard activity code` column that represents the activity performed by the user during the data collection.