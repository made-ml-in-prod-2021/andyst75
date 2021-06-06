import os

import click
from sklearn.datasets import load_wine

import logging

logger = logging.getLogger("download")

@click.command("download")
@click.option("--output-dir", "output_dir", required=True,
              type=click.Path(file_okay=False))
def download(output_dir: str):

    data, y = load_wine(return_X_y=True, as_frame=True)

    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    y.to_csv(os.path.join(output_dir, "target.csv"), index=False)


if __name__ == '__main__':
    download()
