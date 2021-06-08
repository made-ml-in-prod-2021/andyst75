import os
import pandas as pd
import click


@click.command("predict")
@click.option("--input-dir", required=True,
              type=click.Path(file_okay=False))
@click.option("--output-dir", required=True,
              type=click.Path(file_okay=False))
def preprocess(input_dir: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))

    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    target.to_csv(os.path.join(output_dir, "target.csv"), index=False)


if __name__ == '__main__':
    preprocess()
