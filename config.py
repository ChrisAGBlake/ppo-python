from omegaconf import OmegaConf
import sys
import logging
from os.path import join, splitext, relpath
import os
from glob import iglob
from typing import Set


SUPPORTED_EXTENSIONS = [".yml", ".yaml"]


def load_config():

    # create empty OmegaConf object
    cfg = OmegaConf.create({})

    root_dir = _get_root_dir()

    # concatenate all config together
    conf_path_set = _concate_conf(root_dir)
    for conf_dir in conf_path_set:
        cfg_temp = OmegaConf.load(join(root_dir, conf_dir))
        _check_dup_key(cfg, cfg_temp)
        cfg = OmegaConf.merge(cfg, cfg_temp)

    # Enable "struct" mode to prevent accessing non-existing keys
    OmegaConf.set_struct(cfg, True)
    # Enable "readonly" mode to prevent overwriting keys accidentally
    OmegaConf.set_readonly(cfg, True)

    # Setup logging level
    # if cfg.log_stdout_level in ("NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
    #     os.environ["LOG_STDOUT_LEVEL"] = cfg.log_stdout_level

    return cfg


class LevelCappedStreamHandler(logging.StreamHandler):
    """
    Custom version of StreamHandler that excludes processing at level>=level_cap
    """

    def __init__(self, stream=None, level=logging.NOTSET, level_cap=logging.WARNING):
        super().__init__(stream)
        self.setLevel(level)
        self.level_cap = level_cap

    def emit(self, record):
        if record.levelno >= self.level_cap:
            return
        super().emit(record)


def setup_logging(name, stdout_level=logging.INFO, stderr_level=None):
    LOG_FORMAT = (
        "[%(asctime)s.%(msecs)03d][%(threadName)s][%(levelname)5.8s][%(filename).20s:%(lineno)d][%(name)5.15s] "
        "%(message)s"
    )
    DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    stdout_level = logging._checkLevel(
        stdout_level
    )  # If str, convert to int; also validate
    if stderr_level is not None:
        stderr_level = logging._checkLevel(
            stderr_level
        )  # If str, convert to int; also validate

    # Set root logger to process all events by default so as to allow this logger to handle arbitrary levels;
    # assume that other loggers have been configured properly to use their own handlers rather than the root one
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.NOTSET)

    # Remove previous root handlers
    for handler in list(reversed(root_logger.handlers)):
        if type(handler) == logging.StreamHandler:
            root_logger.removeHandler(handler)

    # These groups are known to use logging.NOTSET
    for group in (
        "tensorflow",
        "botocore",
        "boto3",
        "urllib3",
        "ray",
        "s3transfer",
        "matplotlib",
        "paramiko",
    ):
        new_level = max(stdout_level, logging.INFO)
        logging.getLogger(group).setLevel(new_level)

    logger = logging.getLogger(name)

    # If wants stderr stream handler, only log other messages to stdout
    if stderr_level is not None:
        stderr_handler = logging.StreamHandler()  # No args = uses sys.stderr
        stderr_handler.setFormatter(formatter)
        stderr_handler.setLevel(stderr_level)
        root_logger.addHandler(stderr_handler)

        stdout_handler = LevelCappedStreamHandler(
            stream=sys.stdout, level=stdout_level, level_cap=stderr_level
        )
        stdout_handler.setFormatter(formatter)
        root_logger.addHandler(stdout_handler)

    else:
        # Setup requested logger with attached stdout stream handler
        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        stdout_handler.setFormatter(formatter)
        stdout_handler.setLevel(stdout_level)
        root_logger.addHandler(stdout_handler)

    return logger


def _concate_conf(root_dir):
    config_path = os.path.join(root_dir, "conf")
    config_path_set = _get_configuration_files(config_path)
    return config_path_set


def _get_configuration_files(conf_path) -> Set[str]:
    # save all relative paths of configs to a dictionary
    return {
        filepath
        for filepath in iglob(join(conf_path, "**"), recursive=True)
        if _extension(filepath) in SUPPORTED_EXTENSIONS
    }


def _extension(file_path):
    return splitext(file_path)[1].lower()


def _key(file_path):
    return splitext(file_path)[0].lower()


def _get_root_dir():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return root_dir


def _check_dup_key(cfg, new_cfg):
    for key in list(new_cfg):
        if key in list(cfg):
            raise ValueError(
                "{} has already existed in more than one yaml file. "
                "This is not currently supported, as it is unclear "
                "how to merge them. Please either rename one of them "
                "or merge them in a single file.".format(key)
            )
