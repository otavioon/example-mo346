from data_modules.har import HarDataModule

def main():
    # Instantiating the HarDataModule with root dir at data/example
    my_datamodule = HarDataModule(root_data_dir="data/example", 
                                flatten = True, 
                                target_column = "standard activity code", 
                                batch_size=16)

    my_datamodule.prepare_data()

    train_dl = my_datamodule.train_dataloader()

    print(f"There are {len(train_dl)} batches in the training set!")
    for batch in train_dl:
        print("- Batch object is of type:", type(batch))

    print("Done!")

if __name__ == "__main__":
    main()