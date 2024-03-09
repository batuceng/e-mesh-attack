import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from datetime import datetime
import argparse
import os

from dataset.coma2 import Coma
from dataset.bosphorus_stl import Bosphorus
from dataset.facewarehouse_stl import Facewarehouse
from classifiers import PCT_cls, DGCNN_cls, CurveNet_cls, PointMLP_cls, PointNet_cls, PointNet2_cls
from attacks import EpsMeshAttack, PGDL2, PGDLinf, VANILA

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def none_or_str(value):
    if value.lower() == 'none':
        return None
    return value

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Arguments
parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--z-filter', type=str2bool, default=True)
parser.add_argument('--dataset', type=str, default='coma', metavar='N',
                    choices=['coma', 'bosphorus', 'facewarehouse'],
                    help='Select the dataset: Coma, BosphorusDB, FaceWarehouse')
parser.add_argument('-model', type=str, default="dgcnn", metavar='N', choices=['pointnet', 'pointnet2', 'dgcnn', 'curvenet', 'pct', 'pointmlp'],
                    help='Model to use, [pointnet, pointnet2, dgcnn, curvenet, pct, pointmlp]')
parser.add_argument('--save_path', type=none_or_str, default="./data_attacked",
                    help="Path to save the attacked dataset. Give None to not save. Creates subdirs.")
parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"])
parser.add_argument('--seed', type=int, default=0)

subparsers = parser.add_subparsers(dest='attack')
subparsers.required = True

# subparsers for Vanila Attack
parser_vanila = subparsers.add_parser('vanila')

# subparsers for PGDL2 Attack
parser_pgdl2 = subparsers.add_parser('pgdl2')
parser_pgdl2.add_argument('--eps', type=float, default=1.25,
                        help='L2 eps bound')
parser_pgdl2.add_argument('--alpha', type=float, default=0.05,
                        help='Size of each step')
parser_pgdl2.add_argument('--steps', type=int, default=50,
                        help='Number of iteration steps for attack')

# subparsers for PGDLinf Attack
parser_pgd = subparsers.add_parser('pgd').add_argument_group()
parser_pgd.add_argument('--eps', type=float, default=0.05,
                        help='Linf eps bound')
parser_pgd.add_argument('--alpha', type=float, default=0.002,
                        help='Size of each step')
parser_pgd.add_argument('--steps', type=int, default=50,
                        help='Number of iteration steps for pgd')

# subparsers for e-mesh Attack
parser_emesh = subparsers.add_parser('e-mesh')
parser_emesh.add_argument('--projection', type=str, default="perpendicular", choices=["perpendicular", "central"])
parser_emesh.add_argument('--eps', type=float, default=1.0,
                        help='Mesh bound scale. Can be 0 < eps <=1')
parser_emesh.add_argument('--alpha', type=float, default=0.05,
                        help='Size of each step')
parser_emesh.add_argument('--steps', type=int, default=50,
                        help='Number of iteration steps for attack')

# Parse args
parser._action_groups.reverse()

args = parser.parse_args()
print(args)

model_dict = {
    'curvenet': CurveNet_cls, 
    'pct':      PCT_cls, 
    'pointmlp': PointMLP_cls, 
    'dgcnn':    DGCNN_cls, #0.79
    'pointnet2':PointNet2_cls, 
    'pointnet': PointNet_cls, #0.71
}

dataset_dict = {
    'coma':         Coma,
    'bosphorus':    Bosphorus,
    'facewarehouse':Facewarehouse
}
# Ensure that z_filter is closed in BosphorusDB dataset
if args.dataset == 'bosphorus': assert args.z_filter == False 
if args.dataset == 'facewarehouse': assert args.z_filter == True 

# Attack Dictionary
attack_dict = {
    'e-mesh':   EpsMeshAttack,
    'pgdl2':    PGDL2,
    'pgd':    PGDLinf,
    'vanila':   VANILA
}

if args.dataset == 'coma':
    pretrained = "coma_trained"
elif args.dataset == 'bosphorus':
    pretrained = "bosphorus_trained"
elif args.dataset == 'facewarehouse':
    pretrained = "facewarehouse_trained"
else: pretrained = True
    

dataset = dataset_dict[args.dataset](partition="test", z_filter=args.z_filter)
test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() and args.device == "cuda" else 'cpu')
no_of_classes = len(dataset.label_dict)
model = model_dict[args.model](output_channels=no_of_classes, pretrained=pretrained).to(device)
model.eval()

if args.attack == "e-mesh":
    atk = EpsMeshAttack(model=model, device=device, projection=args.projection, eps=args.eps, alpha=args.alpha, steps=args.steps, seed=args.seed)
    args.attack += f"-{args.projection}"
elif args.attack == "pgdl2":
    atk = PGDL2(model=model, device=device, eps=args.eps, alpha=args.alpha, steps=args.steps, random_start=False, seed=args.seed)
elif args.attack == "pgd":
    atk = PGDLinf(model=model, device=device, eps=args.eps, alpha=args.alpha, steps=args.steps, random_start=False, seed=args.seed)
else:
    atk = VANILA(model=model, device=device, seed=args.seed)


# Rename save_path as {dataset_name}_attacked
args.save_path = f"{args.dataset}_attacked"

# Define subfolders to save

savedir = os.path.join(*[args.save_path, args.attack, args.model]) if args.save_path is not None else None
filename = os.path.join(savedir, datetime.today().strftime('ATK_%Y_%m_%d__%H_%M_%S')) if savedir is not None else None

# Apply Attack & Save
atk.save(dataloader=test_loader, root=savedir, file_name=filename, args=vars(args))

