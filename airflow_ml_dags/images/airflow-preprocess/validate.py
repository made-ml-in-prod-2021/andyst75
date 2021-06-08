import os
import pickle
import logging

import click
import pandas as pd
from sklearn.metrics import classification_report

logger = logging.getLogger("validate")


@click.command("download")
@click.option("--input-dir", required=True, type=click.Path(file_okay=False))
@click.option("--models-dir", required=True, type=click.Path(file_okay=False))
def train(input_dir: str, models_dir: str):
    logger.info("Start validate")

    x_test = pd.read_csv(os.path.join(input_dir, "x_test.csv"))
    y_test = pd.read_csv(os.path.join(input_dir, "y_test.csv"))

    model_path = os.path.join(models_dir, "model.pkl")
    report_path = os.path.join(models_dir, "report.txt")

    with open(model_path, "rb") as fio, open(report_path, "w") as rep:
        clf = pickle.load(fio)
        y_pred = clf.predict(x_test)
        rep.writelines(classification_report(y_test, y_pred))

    logger.info("Complete")


if __name__ == '__main__':
    train()
