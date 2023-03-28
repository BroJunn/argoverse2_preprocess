import json, pickle
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
import os, sys

from torch.utils.data import DataLoader

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from argoverse_v2_gen_dataset import Argov2PreprocessDataset
from utils.utils_general import read_yaml

def gen_dir_name(config_yaml_path: str):
    config_dict = read_yaml(config_yaml_path)
    return os.path.join('{}_{}_{}_{}_{}_{}'.format(config_dict['grid_height_cells'],
                                                   config_dict['grid_width_cells'],
                                                   config_dict['sdc_y_in_grid'],
                                                   config_dict['sdc_x_in_grid'],
                                                   config_dict['pixels_per_meter'],
                                                   config_dict['referenced_time_step']), config_dict['dataset_type'])

def build_dataset_processor(args):
    args = args.__dict__
    gen_dataset_path = os.path.join(args['gen_dataset_root_path'], gen_dir_name(args['gen_dataset_config_path']))
    
    if not os.path.exists(gen_dataset_path):
        os.makedirs(gen_dataset_path)
    
    gen_dataset = Argov2PreprocessDataset(args['raw_dataset_path'], read_yaml(args['gen_dataset_config_path']))
    for i in tqdm(range(gen_dataset.__len__())):
        data = gen_dataset.__getitem__(i)
        np.save(os.path.join(gen_dataset_path, '{}.npy'.format(data['argo_id'])), data)

if __name__ == '__main__':
    parser_gen_datset = ArgumentParser()
    parser_gen_datset.add_argument('--raw_dataset_path',
                                   type=str,
                                   default='/home/yujun/Dataset/train/')
    parser_gen_datset.add_argument('--gen_dataset_config_path',
                                   type=str,
                                   default='/home/yujun/Code/argoverse2_preprocess/generate_dataset/config/gen_dataset.yaml')
    parser_gen_datset.add_argument('--gen_dataset_root_path',
                                   type=str,
                                   default='/home/yujun/Code/argoverse2_preprocess/preprocessed_dataset')
    args_gen_datset = parser_gen_datset.parse_args()
    build_dataset_processor(args_gen_datset)
