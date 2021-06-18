import os
import pandas as pd
import logging
import pickle

import click

logger = logging.getLogger("predict")


@click.command("predict")
@click.option("--input-dir", type=click.Path(exists=True))
@click.option("--output-dir", type=click.Path())
@click.option("--model-path", type=click.Path(exists=True))
def predict(input_dir: str, output_dir: str, model_path: str):

    logger.info("Start validate")

    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    predict_path = os.path.join(output_dir, "predictions.csv")

    with open(model_path, "rb") as fio:
        clf = pickle.load(fio)
        y_pred = clf.predict(data)
        data = pd.DataFrame(y_pred, columns=["target"])
        os.makedirs(output_dir, exist_ok=True)
        data.to_csv(predict_path, index=False)

    logger.info("Complete")


if __name__ == '__main__':
    predict()
