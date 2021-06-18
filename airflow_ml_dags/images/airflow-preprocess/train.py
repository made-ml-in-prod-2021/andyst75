import os
import pickle
import logging

import click
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger("train")


@click.command("download")
@click.option("--input-dir", required=True, type=click.Path(file_okay=False))
@click.option("--models-dir", required=True, type=click.Path(file_okay=False))
@click.option("--seed", required=True, type=int)
def train(input_dir: str, models_dir: str, seed: int):
    logger.info("Start train")

    x_train = pd.read_csv(os.path.join(input_dir, "x_train.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv"))

    clf = RandomForestClassifier(max_depth=3, random_state=seed)
    clf.fit(x_train, y_train)

    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "model.pkl")

    with open(model_path, "wb") as fio:
        pickle.dump(clf, fio, protocol=3)

    logger.info("Complete")


if __name__ == '__main__':
    train()
