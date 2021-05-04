import logging
import os

import hydra

from ..model import build_model
from ..classes import ConfigParams
from ..data import read_data, check_data, split_train_val_data
from ..data.data_transformer import DatasetTransformer

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(config_path=os.path.join("..", "..", "configs"),
            config_name="train")
# def train_pipeline(cfg: hydra.utils.DictConfig) -> None:
def train_pipeline(cfg: ConfigParams) -> None:
    df = read_data(cfg.input_data_path)

    # print(OmegaConf.to_yaml(cfg))

    check_df, cat_error, num_error = check_data(df, cfg.features)

    if not check_df:
        error_msg = f"Some features not found: { {*cat_error, *num_error} }"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    train_df, test_df = split_train_val_data(df, cfg.split)

    # encode params +
    # save params +
    # make model
    # train model
    # save model
    print(train_df.shape, test_df.shape)

    trans = DatasetTransformer(cfg.features,
                               cfg.models.transforms)
    trans.fit(train_df)
    trans.dump(cfg.transform_path)

    print(OmegaConf.to_yaml(cfg.models.model_parameters))
    model = build_model(cfg.models)



    # tranformed_df = trans.transform(train_df)


if __name__ == '__main__':
    train_pipeline()
