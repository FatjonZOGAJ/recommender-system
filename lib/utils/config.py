import yaml as yaml
from easydict import EasyDict as edict

config = edict()
config.NUM_USERS, config.NUM_MOVIES = (10000, 1000)
config.RANDOM_STATE = 42

config.OUTPUT_DIR = 'output'
config.FINAL_OUTPUT_DIR = ''
config.SUBMISSION_NAME = ''
config.TIME_STR = ''
config.DATA_DIR = 'data/'
config.TRAIN_SIZE = 0.9

config.VALIDATE = True

config.MODEL = 'kernelNet'
config.DEFAULT_VALUE = 'user_mean'
config.STRATIFY = 'movies'  # or users

config.K_SINGULAR_VALUES = 3
config.MAX_ITER = 2

config.TEST_EVERY = 3

# saving config file
def gen_config(config_file, config):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def _update_dict(k, v):
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} does not exist in config.py".format(k, vk))


def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))
