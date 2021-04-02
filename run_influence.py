import yaml
import argparse
import numpy as np

from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
##
import os
import glob
import pytorch_lightning as pl
# from pytorch_lightning.utilities.cloud_io import load as pl_load
import torch.autograd as autograd
from tqdm import tqdm, trange
from collections import defaultdict
import pdb


def gather_flat_grad(grads):
    views = []
    for p in grads:
        if p.data.is_sparse:
            view = p.data.to_dense().view(-1)
        else:
            view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')
##
parser.add_argument('--model_ckpt_dir',
                    help='path to the model checkpoint',
                    type=str,
                    default='')
parser.add_argument('--test_idx',
                    type=int,
                    default=1,
                    help="test index we want to examine")
parser.add_argument('--start_test_idx',
                    type=int,
                    default=-1,
                    help="when not -1, --test_idx will be disabled")
parser.add_argument('--end_test_idx',
                    type=int,
                    default=-1,
                    help="when not -1, --test_idx will be disabled")
parser.add_argument("--influence_metric",
                    default="",
                    type=str,
                    help="standard dot product metric or theta-relative")

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tt_logger = TestTubeLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    debug=False,
    create_git_tag=False,
)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'])

# Han: above mostly unchanged, now substituting the trainer with influence analysis

# currently not working for model loading because of some issues over pytorch_lightning's version?
if args.model_ckpt_dir:
    checkpoints = list(sorted(glob.glob(os.path.join(args.model_ckpt_dir, "*.ckpt"), recursive=True)))
#     trained_ckpt = pl_load(trained_ckpt_file, map_location=lambda storage, loc: storage)
#     model.load_state_dict(trained_ckpt['state_dict'])
    trained_ckpt = torch.load(checkpoints[-1])
    experiment.load_state_dict(trained_ckpt['state_dict'])

print('++++++++++ number of params +++++++++', sum(p.numel() for n, p in model.named_parameters()))

# prepare parameters
param_influence = []
frozen = []
# frozen = ['decoder']
for n, p in list(model.named_parameters()):
    if (not any(fr in n for fr in frozen)):
        param_influence.append(p)
    else:
        p.requires_grad = False

# dataloaders
train_dataloader = experiment.train_dataloader() # note that currently shuffle=True, also need to set batch_size=1
test_dataloader = experiment.test_dataloader()[0]

agg_influence_dict = defaultdict(list)
ihvp_dict = dict()

# L_TEST
for test_idx, test_batch in enumerate(tqdm(test_dataloader, desc="Test set index")): # somehow need to squeeze the test dataloader
    assert len(test_batch[0]) == 1 # check whether only one image is passed in
    
    if args.start_test_idx != -1 and args.end_test_idx != -1:
        if test_idx < args.start_test_idx:
            continue
        if test_idx > args.end_test_idx:
            break
    else:
        if test_idx < args.test_idx:
            continue
        if test_idx > args.test_idx:
            break

    ######## L_TEST GRADIENT ########
    model.eval()
    model.zero_grad()
    test_loss = experiment.testing_step(test_batch, batch_idx=test_idx)
    test_grads = autograd.grad(test_loss['loss'], param_influence)
    ################

    ihvp_dict[test_idx] = gather_flat_grad(test_grads).detach().cpu() # put to CPU to save GPU memory

ihvp_stack = torch.stack([ihvp_dict[tmp_idx] for tmp_idx in sorted(ihvp_dict.keys())], dim=0).to(experiment.curr_device)
ihvp_dict_keys = sorted(ihvp_dict.keys())
del ihvp_dict
influence_list = []

# L_TRAIN
for train_idx, train_batch in enumerate(tqdm(train_dataloader, desc="Train set index")):
    assert len(train_batch[0]) == 1 # check whether only one image is passed in
    
    if train_idx >= 500: # just checking a few examples for now due to speed
        break
    
    ######## L_TRAIN GRADIENT ########
    model.eval()
    model.zero_grad()
    train_loss = experiment.training_step(train_batch, batch_idx=train_idx, no_log=True)
    train_grads = autograd.grad(train_loss['loss'], param_influence)
    ################

    with torch.no_grad(): # check the speed of the block below
        if args.influence_metric == "cosine":
            influence_list.append(torch.nn.functional.cosine_similarity(ihvp_stack,
                                                                        torch.unsqueeze(gather_flat_grad(train_grads), 0),
                                                                        dim=1, eps=1e-12).detach().cpu())
        elif args.influence_metric == "dotprod":
            influence_list.append(torch.matmul(ihvp_stack, gather_flat_grad(train_grads)).detach().cpu())
        else:
            raise ValueError("specified influence metric does not exist")
            
# wrap up the influence scores
for test_idx in ihvp_dict_keys:
    agg_influence_dict[test_idx].append(np.zeros(len(train_dataloader)))
for train_i, train_i_inf in enumerate(influence_list):
    for test_idx, train_i_inf_on_test_j in zip(ihvp_dict_keys, train_i_inf):
        agg_influence_dict[test_idx][-1][train_i] = train_i_inf_on_test_j.item()
del influence_list

# information stored in agg_influence_dict
print(agg_influence_dict.keys())

pdb.set_trace()