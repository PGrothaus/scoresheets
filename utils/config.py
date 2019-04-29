"""Module containing config methods and global variables."""
from __future__ import absolute_import
import yaml
import json
import os
from shutil import copyfile
from collections import OrderedDict
if False:
    from typing import Optional, Any, Dict, List, Tuple

dic = dict()  # type: Dict[Any, Any]
path_to_config = None  # type: Optional[str]


def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):  # type: ignore
    class OrderedLoader(Loader):  # type: ignore
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,  # type: ignore
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


def forceloadconfig_for_deployment(path):
    # type: (str) -> None
    global dic
    with open(path) as stream:
        local_dict = ordered_load(stream, yaml.SafeLoader)  # type: ignore
        for k, v in local_dict.iteritems():
            dic[k] = v


def forceloadconfig_for_test(path):
    # type: (str) -> None
    global dic
    global path_to_config
    if len(dic) > 0:
        dic = dict()
    print 'force-load config file!'
    loadconfig(path)


def load_global_config():
    global dic
    pythonpath = os.environ.get('PYTHONPATH').split(':')
    idx = [i for i, x in enumerate(pythonpath) if 'image-to-pgn' in x]
    if not idx:
        raise Exception('PYTHONPATH not set!' +
                        'You need to either specify PYTHONPATH '
                        'or specify path in loadconfig() call')

    pypath = pythonpath[idx[0]]
    pypath = pypath if '/' == pypath[-1] else pypath + '/'
    global_config_path = pypath + "config.global.yaml"
    print 'loading global config file {}'.format(global_config_path)
    with open(global_config_path) as stream:
        dic = ordered_load(stream, yaml.SafeLoader)  # type: ignore


def loadconfig(config_path=None):
    # type: (Optional[str]) -> None
    global dic
    global path_to_config
    if len(dic) > 0:
        raise Exception('Attempted to load config file twice')
    load_global_config()
    if config_path is None:
        return
    else:
        path_to_config = config_path
        print 'loading local config file {}'.format(config_path)
        with open(config_path) as stream:
            local_dict = ordered_load(stream, yaml.SafeLoader)  # type: ignore
            for k, v in local_dict.iteritems():
                dic[k] = v


def load_single_config(config_path):
    # type: (str) -> None
    global dic
    if len(dic) > 0:
        raise Exception('Attempted to load config file twice')
    print 'loading local config file {}'.format(config_path)
    with open(config_path) as stream:
        local_dict = ordered_load(stream, yaml.SafeLoader)  # type: ignore
        for k, v in local_dict.iteritems():
            dic[k] = v


def loadconfig_model_combination(combination_id):
    # type: (int) -> None
    global dic
    global path_to_config
    if len(dic) > 0:
        raise Exception('Attempted to load config file twice')
    load_global_config()
    path = ''.join([dic['homeDir'], '/model_combinations/',
                    str(combination_id), '.yaml'])
    with open(path) as stream:
        local_dict = ordered_load(stream, yaml.SafeLoader)  # type: ignore
        for k, v in local_dict.iteritems():
            dic[k] = v


def copy_config_file(filepath):
    global path_to_config
    if path_to_config is not None:
        if filepath != path_to_config:
            copyfile(path_to_config, filepath)
    else:
        raise ValueError('Need to specify path_to_config.')


def write_config_to_file(filepath):
    # type: (str) -> None
    global dic
    if len(dic) > 0:
        f = open(filepath, 'w')
        for k, v in dic.iteritems():
            f.write(str(k) + ': ' + str(v) + '\n')
        f.close()
    else:
        raise ValueError(
            'Trying to write config to file, but no config has been loaded.')


def get_from_config(par):
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


def gfc(par):
    return get_from_config(par)


def get_config():
    # type: () -> Dict[Any, Any]
    global dic
    return dic


def collect_trf_keywords():
    # type: () -> Dict[str, Any]
    global dic
    kws = ['base_dimension',
           'networkType',
           'resizeFactor',
           'training',
           'TRIM',
           'WARP',
           'RANDOM_CROP',
           'CENTRAL_CROP',
           'SCALE_COLORS',
           'HORIZONTAL_FLIP']
    return {kw: get_from_config(kw) for kw in kws}


if __name__ == '__main__':
    loadconfig()
