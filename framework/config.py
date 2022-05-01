import os, pickle, torch
import numpy as np


dataset_dir="E:\Dataset\DCASE2018_T1_A"



subdir = os.path.join(dataset_dir, 'TUT-urban-acoustic-scenes-2018-development')
if not os.path.exists(subdir):
    subdir = os.path.join(dataset_dir, 'development')

# 2021-9-7 add
cuda_seed = 1024
cuda = 1

if cuda:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
else:
    device = torch.device('cpu')

if cuda_seed:
    torch.manual_seed(cuda_seed)
    if cuda:
        torch.cuda.manual_seed(cuda_seed)
        torch.cuda.manual_seed_all(cuda_seed)


sample_rate = 44100
window_size = 2048
overlap = 672
seq_len = 320
mel_bins = 64

epochs = 10000
batch_size = 64
only_save_best = False
shared_layers = 0

devices = ['a']

labels = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
          'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']

lb_to_ix = {lb: ix for ix, lb in enumerate(labels)}
ix_to_lb = {ix: lb for ix, lb in enumerate(labels)}

filepath = os.path.join(os.getcwd(), 'all_10scenes_527events_relation_pro.pkl')
data = pickle.load(open(filepath, 'rb'))
event_labels = data['id_names']
events_index = data['id_index']

relation_matrix = data['relation_matrix'].T

relation_scene_event = data['relation_matrix']

endswith = '.pth'
