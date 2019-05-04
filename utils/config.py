"""Module containing config methods and global variables."""
from __future__ import absolute_import
import yaml
from collections import OrderedDict

dic = dict()
path_to_config = None


def loadconfig(config_path):
    global dic
    global path_to_config
    if len(dic) > 0:
        raise Exception('Attempted to load config file twice')
    else:
        path_to_config = config_path
        print('loading local config file {}'.format(config_path))
        with open(config_path) as stream:
            local_dict = ordered_load(stream, yaml.SafeLoader)
            for k, v in local_dict.items():
                dic[k] = v


def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


def get_param(par):
    global dic
    if len(dic) > 0:
        try:
            val = dic[par]
            if isinstance(val, str):
                val = val[:val.rfind("#")] if "#" in val else val
                val = val.rstrip().lstrip()
                val = None if val == 'None' else val
            return val
        except:
            raise Exception(
                "{} not in config file or commented out".format(par))
    else:
        raise ValueError('No config has been loaded.')
