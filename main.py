#!/usr/bin/env python3

from lightning.pytorch.cli import (
    ArgsType,
    LightningCLI,
)

import pandas as pd


def cli_main(args: ArgsType = None):
    """Main function for the CLI.

    Parameters
    ----------
    args : ArgsType, optional
        Command line arguments. If None, it will use `sys.argv`, by default None
    """
    # Lightning CLI will instantiate the model, datamodule and trainer
    cli = LightningCLI(args=args, run=False)

    # Fit model
    cli.trainer.fit(cli.model, cli.datamodule)
    
    # Test
    metrics = cli.trainer.test(cli.model, cli.datamodule)
    
    # Print metrics
    metrics = pd.DataFrame(metrics)
    print(metrics.to_markdown())


if __name__ == "__main__":
    cli_main()
