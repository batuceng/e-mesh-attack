import torch
import os
import numpy as np
import random

from dataset.coma2 import Coma
from torch.utils.data import DataLoader
from classifiers import DGCNN_cls, PointNet2_cls, PointNet_cls, PCT_cls, PointMLP_cls, CurveNet_cls, GACNet_cls
from attacks import EpsMeshAttack, PGDL2, PGDLinf, VANILA

from torch.utils.tensorboard import SummaryWriter
import json
import argparse
import time


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

# Datasets and loaders

parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--z-filter', type=str2bool, default=True)

parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                    choices=['pointnet', 'pointnet2', 'dgcnn', 'curvenet', 'pct', 'pointmlp', 'gacnet'],
                    help='Model to use, [pointnet, pointnet2, dgcnn, curvenet, pct, pointmlp, gacnet]')

parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning Rate')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='Weight Decay')
parser.add_argument('--lr_step_size', type=int, default=25,
                    help='Decay the LR every n step')
parser.add_argument('--lr_decay', type=float, default=0.95,
                    help='LR Decay at each step')
parser.add_argument('--seed', type=int, default=1)

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
parser_pgdl2.add_argument('--steps', type=int, default=30,
                        help='Number of iteration steps for attack')

# subparsers for PGDLinf Attack
parser_pgd = subparsers.add_parser('pgd').add_argument_group()
parser_pgd.add_argument('--eps', type=float, default=0.05,
                        help='Linf eps bound')
parser_pgd.add_argument('--alpha', type=float, default=0.002,
                        help='Size of each step')
parser_pgd.add_argument('--steps', type=int, default=30,
                        help='Number of iteration steps for pgd')

# subparsers for e-mesh Attack
parser_emesh = subparsers.add_parser('e-mesh')
parser_emesh.add_argument('--projection', type=str, default="perpendicular", choices=["perpendicular", "central"])
parser_emesh.add_argument('--eps', type=float, default=1.0,
                        help='Mesh bound scale. Can be 0 < eps <=1')
parser_emesh.add_argument('--alpha', type=float, default=0.05,
                        help='Size of each step')
parser_emesh.add_argument('--steps', type=int, default=30,
                        help='Number of iteration steps for attack')


# Convert np.inf & np.nan to str before json dump
def convert_numpy_objects(dict_to_convert):
    new = {}
    for k, v in dict_to_convert.items():
        if isinstance(v, dict):
            new[k] = convert_numpy_objects(v)
        else:
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                new[k] = str(v)
            else:
                new[k] = v
    return new

def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# Parse args
parser._action_groups.reverse()

args = parser.parse_args()
seed_all(args.seed)
print(args)

model_dict = {
    'curvenet': CurveNet_cls, #0.
    'pct':      PCT_cls, #0.
    'pointmlp': PointMLP_cls, #0.
    'dgcnn':    DGCNN_cls, #0.79
    'pointnet2':PointNet2_cls, #0.
    'pointnet': PointNet_cls,
    'gacnet': GACNet_cls
}

# Attack Dictionary
attack_dict = {
    'e-mesh':   EpsMeshAttack,
    'pgdl2':    PGDL2,
    'pgd':    PGDLinf,
    'vanila':   VANILA
}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model_dict[args.model](pretrained=True).to(device)
model.eval()

MAX_EPOCH = 1000
trainloader = DataLoader(dataset=Coma(partition='train', z_filter=args.z_filter), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
testloader = DataLoader(dataset=Coma(partition='test', z_filter=args.z_filter), batch_size=args.batch_size, num_workers=args.num_workers)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

writer = SummaryWriter()


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd) # added 1e-5 reg
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_decay)  # changed step size to 25
loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')


# Dump Args to json
args_dict = vars(args)
with open(os.path.join(writer.log_dir,'args.json'), 'w') as fp:
    json.dump(convert_numpy_objects(args_dict), fp, indent=4)
    fp.close()


# Lists to store training scores
train_epoch_loss_list = []
train_acc_list = []
test_epoch_loss_list = []
test_acc_list = []
best_acc = 0
    
for epoch in range(MAX_EPOCH):
    if args.attack == "e-mesh":
        atk = EpsMeshAttack(model=model, device=device, projection=args.projection, eps=args.eps, alpha=args.alpha, steps=args.steps, seed=args.seed)
    elif args.attack == "pgdl2":
        atk = PGDL2(model=model, device=device, eps=args.eps, alpha=args.alpha, steps=args.steps, random_start=False, seed=args.seed)
    elif args.attack == "pgd":
        atk = PGDLinf(model=model, device=device, eps=args.eps, alpha=args.alpha, steps=args.steps, random_start=False, seed=args.seed)
    else:
        atk = VANILA(model=model, device=device, seed=args.seed)
    time0 = time.time()

    # TRAIN
    train_running_loss = 0
    train_true_pred_count = 0
    length = len(trainloader)
    for i, data_dict in enumerate(trainloader):
        if i % (100//args.batch_size) == 0:
            print("Batch:", i, "/", length)
        adv_pc = atk.attack(data=data_dict['pc'], labels=data_dict['cate'], meshvectors=data_dict['meshvectors'], meshnormals=data_dict['meshnormals'])
        model.train()
        
        pc, label = adv_pc.to(device), data_dict['cate'].to(device)
        bs = pc.shape[0]
        # print(pc.shape)

        optimizer.zero_grad()
        logits = model(pc.transpose(1,2))
        preds = logits.argmax(dim=1)
        # print(logits.argmax())
        loss = loss_fn(logits, label)
        
        loss.backward()
        
        optimizer.step()
        
        train_running_loss += loss
        train_true_pred_count += torch.sum(preds==label)
    train_epoch_loss_list.append(train_running_loss/len(trainloader.dataset))
    train_acc_list.append(train_true_pred_count/len(trainloader.dataset))
    
    # EVAL
    with torch.no_grad():
        model.eval()
        test_running_loss = 0
        test_true_pred_count = 0
        for data_dict in testloader:
            pc, label = data_dict['pc'].to(device), data_dict['cate'].to(device)
            bs = pc.shape[0]

            logits = model(pc.transpose(1,2))
            preds = logits.argmax(dim=1)
            loss = loss_fn(logits, label)
            
            test_running_loss += loss
            test_true_pred_count += torch.sum(preds==label)
        test_epoch_loss_list.append(test_running_loss/len(testloader.dataset))
        test_acc_list.append(test_true_pred_count/len(testloader.dataset))
    
    # Verbose    
    print(f"""------Model:{args.model} EPOCH:{epoch}, lr:{optimizer.param_groups[0]['lr']}, Time: {time.time()-time0:.6f}-------
        Train Loss:{train_epoch_loss_list[epoch]:.6f}, Train Acc:{train_acc_list[epoch]:.6f}
        Test Loss:{test_epoch_loss_list[epoch]:.6f}, Test Acc:{test_acc_list[epoch]:.6f}""")
    # Log Tensorboard
    writer.add_scalar("Loss/Train", train_epoch_loss_list[epoch], epoch)
    writer.add_scalar("Loss/Test", test_epoch_loss_list[epoch], epoch)
    writer.add_scalar("Accuracy/Train", train_acc_list[epoch], epoch)
    writer.add_scalar("Accuracy/Test", test_acc_list[epoch], epoch)
    writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
    # Save model every n ep
    if epoch%10==0:
        torch.save(model.state_dict(), os.path.join(writer.log_dir, f'epoch-{epoch}.pt'))
    if test_acc_list[epoch] >= best_acc:
        best_acc = test_acc_list[epoch]
        torch.save(model.state_dict(), os.path.join(writer.log_dir, f'best-model.pt'))
        with open(os.path.join(writer.log_dir, "best_model.txt"), "w") as file:
            file.write(f"Best test acc: {best_acc}\n")
            file.write(f"Epoch : {epoch}")

    scheduler.step()