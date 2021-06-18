import os

import pandas as pd
from sklearn.model_selection import train_test_split
import click


@click.command("predict")
@click.option("--input-dir", required=True, type=click.Path(file_okay=False))
@click.option("--train-size", required=True, type=float)
@click.option("--seed", required=True, type=int)
def preprocess(input_dir: str, train_size: float, seed: int):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))

    x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                        train_size=train_size,
                                                        random_state=seed)

    x_train.to_csv(os.path.join(input_dir, "x_train.csv"), index=False)
    x_test.to_csv(os.path.join(input_dir, "x_test.csv"), index=False)
    y_train.to_csv(os.path.join(input_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(input_dir, "y_test.csv"), index=False)


if __name__ == '__main__':
    preprocess()
