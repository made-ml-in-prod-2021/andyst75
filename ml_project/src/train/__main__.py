import logging
import os

import hydra

from .make_report import build_train_report
from ..classes import ConfigParams
from ..data import read_data, check_data, split_train_val_data
from ..data.data_transformer import DatasetTransformer
from ..model import build_model, dump_model

logger = logging.getLogger(__name__)


@hydra.main(config_path=os.path.join("..", "..", "configs"),
            config_name="train")
def train_pipeline(cfg: ConfigParams) -> None:
    """
    Main train pipeline.
    Parameters read from YAML-file.
    For run with custom parameters usage --config-name=config_name and
    --config-path=config_path (absolute or relative)
    """

    logger.info("Start train pipeline")

    df = read_data(cfg.input_data_path)

    check_df, cat_error, num_error = check_data(df, cfg.features)

    if not check_df:
        error_msg = f"Some features not found: { {*cat_error, *num_error} }"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    train_df, test_df = split_train_val_data(df, cfg.split)

    ft_transforms = DatasetTransformer(cfg.features, cfg.models.transforms)
    ft_transforms.fit(train_df)
    ft_transforms.dump(cfg.transform_path)

    x_train = ft_transforms.transform(train_df)
    x_test = ft_transforms.transform(test_df)

    y_train = train_df[cfg.features.target_col].values
    y_test = test_df[cfg.features.target_col].values

    model = build_model(cfg.models)
    model.fit(x_train, y_train)
    dump_model(cfg.model_path, model)

    report = build_train_report(model, x_test, y_test, df, cfg)

    logger.info(f"Accuracy: {report.accuracy:.4f}, " +
                f"F1 metric: {report.f1_metric:.4f}")

    logger.info("Finish train pipeline")


if __name__ == '__main__':
    train_pipeline()
