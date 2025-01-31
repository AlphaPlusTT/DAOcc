import argparse
import copy
import os
import random
import time

import numpy as np
import torch
from mmcv import Config
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval


def main():
    # dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    parser.add_argument("--dist", action='store_true', help="distributed or not")
    args, opts = parser.parse_known_args()

    if args.dist:
        dist.init()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    if args.dist:
        torch.cuda.set_device(dist.local_rank())
    else:
        torch.cuda.set_device(0)

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir

    # dump config
    cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))

    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    # log some basic info
    logger.info(f"Config:\n{cfg.pretty_text}")
    if args.dist:
        logger.info('-' * 66)
        logger.info('[CUSTOM] distributed mode is on!')
        logger.info('-' * 66)
    else:
        logger.info('-' * 66)
        logger.info('[CUSTOM] distributed mode is off!')
        logger.info('-' * 66)

    # set random seeds
    if cfg.seed is not None:
        logger.info(
            f"Set random seed to {cfg.seed}, "
            f"deterministic mode: {cfg.deterministic}"
        )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    datasets = [build_dataset(cfg.data.train)]

    model = build_model(cfg.model)
    model.init_weights()
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    debug = cfg.get('debug', False)
    if debug:
        # freeze
        logger.info('-' * 66)
        logger.info('[CUSTOM] freeze params !')
        logger.info('-' * 66)
        for _, v in model.encoders.items():
            v.eval()
            for param in v.parameters():
                param.requires_grad = False
        if model.fuser is not None:
            model.fuser.eval()
            for param in model.fuser.eval():
                param.requires_grad = False
        for _, v in model.decoder.items():
            v.eval()
            for param in v.parameters():
                param.requires_grad = False

    logger.info(f"Model:\n{model}")
    train_model(
        model,
        datasets,
        cfg,
        distributed=args.dist,
        validate=True,
        timestamp=timestamp,
    )


if __name__ == "__main__":
    main()
