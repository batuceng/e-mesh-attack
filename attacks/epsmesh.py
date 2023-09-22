import numpy as np
import os
import json
import random

import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score
from os.path import dirname, abspath

def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# Epsilon-Mesh Attack
class EpsMeshAttack(object):
    def __init__(self, model, device, eps=1.0, alpha=0.02, steps=10, random_start=False, seed=3):
        # Model info
        self.model = model.to(device).eval()
        self.name = "EpsMeshAttack"
        self.device = device
        self.targeted = False
        # Attack Vals
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        # Set seed
        self.seed = seed
        if seed is not None: seed_all(seed)

# def triangle_projection(trianle, points):
    
    def attack(self, data, labels, mesh):
        # data: (B, N, d); labels: (B, 1)
        data = data.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        # meshvector: (B, N, d, d); meshnormal: (B, N, d)
        meshvector = torch.tensor(mesh.vectors, dtype=data.dtype, device=data.device)
        meshnormal = torch.tensor(mesh.normals, dtype=data.dtype, device=data.device)
        
        loss = nn.CrossEntropyLoss()
        
        adv_data = data.clone().detach()
        batch_size = data.shape[0]
        
        # if self.random_start:
        #     # Starting at a uniformly random point
        #     delta = torch.empty_like(adv_data).normal_()
        #     d_flat = delta.view(batch_size, -1)
        #     n = d_flat.norm(p=2, dim=1).view(batch_size, 1, 1)
        #     r = torch.zeros_like(n).uniform_(0, 1)
        #     delta = (delta*r*self.eps)/n
        #     adv_data = torch.clamp(adv_data + delta, min=0, max=1).detach()
        
        for _ in range(self.steps):
            adv_data.requires_grad = True
            outputs = self.get_logits(adv_data)
            
            # Calculate loss
            if self.targeted:
                # cost = -loss(outputs, target_labels)
                raise NotImplementedError
            else:
                cost = loss(outputs, labels.squeeze())
            
            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_data,
                                       retain_graph=False, create_graph=False)[0]
            # print(grad.shape)
            grad_norms = torch.norm(grad.reshape(batch_size, -1), p=2, dim=1) + 1e-10  # nopep8
            grad = grad / grad_norms.reshape(batch_size, 1, 1)
            adv_data = adv_data.detach() + self.alpha * grad
            # Project L2 Ball
            delta = adv_data - data
            delta_norms = torch.norm(delta.reshape(batch_size, -1), p=2, dim=1)
            factor = self.eps / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor.reshape(-1, 1, 1)

            adv_data = torch.clamp(data + delta, min=-1, max=1).detach()

        return adv_data
            
    def get_logits(self, inputs):
        logits = self.model(inputs.permute(0,2,1).to(self.device))
        return logits
        
    def save(self, dataloader, root=os.path.join(dirname(abspath(__file__)), "data_attacked"), file_name=None, args=None):
        true_labels, clean_preds, attack_preds = [], [], []
        attacked_batches = []
        for i,batch in enumerate(dataloader):
            print(f"batch {i}/{len(dataloader)}")
            pc, label = batch['pointcloud'], batch['cate']
            mesh = batch['mesh']
            # Forward
            logits = self.get_logits(pc)
            adv_data = self.attack(data=pc, labels=label, mesh=mesh)
            # Check attack
            attacked_logits = self.get_logits(adv_data)
            # Store for Acc Stats
            true_labels.append(label.squeeze().detach().cpu().numpy())
            clean_preds.append(logits.detach().cpu().numpy().argmax(axis=1))
            attack_preds.append(attacked_logits.detach().cpu().numpy().argmax(axis=1))
            # Store Attacked Data
            batch['attacked'] = adv_data.detach().cpu()
            if not self.targeted:           # Save target too
                batch['target'] = torch.ones_like(batch['cate']) * -1
            else: 
                raise NotImplementedError
            # Add each instance to total list
            attacked_batches.extend([{key:batch[key][i] for key in batch} for i in range(pc.shape[0])])
            
        clean_acc = accuracy_score(np.concatenate(true_labels), np.concatenate(clean_preds))
        attack_acc = accuracy_score(np.concatenate(true_labels), np.concatenate(attack_preds))
        datasetname, modelname = dataloader.dataset.__class__.__name__, self.model.__class__.__name__
        print(f"Clean Acc:{clean_acc} on dset:{datasetname} model:{modelname} atk:{self.name}")
        print(f"Attacked Acc:{attack_acc} on dset:{datasetname} model:{modelname} atk:{self.name}")
        
        # Write to root. Do not save if root is None!
        if root != None:
            DATA_DIR = root
            os.makedirs(DATA_DIR, exist_ok=True)
            if file_name == None:
                if self.name == 'EpsMeshAttack':
                    FILE_NAME = os.path.join(DATA_DIR, f'{datasetname}_{modelname}_{self.name}_eps_{self.eps}')
                else:
                    raise NotImplementedError
            else: FILE_NAME = file_name
            
            # Dump Data as .pt
            torch.save(attacked_batches,f=FILE_NAME+'.pt')
            # Dump args as .json, pass a dictionary (even empty one) to save additional args
            if args != None:
                with open(FILE_NAME+'.json', 'w') as fp:
                    args |= {"data_path": FILE_NAME, "clean_acc": clean_acc, "attack_acc": attack_acc}
                    json.dump(args, fp, indent=4)
            
        return attacked_batches

    
