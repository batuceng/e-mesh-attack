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
    def __init__(self, model, device, projection="central", eps=1.0, alpha=0.02, steps=10, random_start=False, seed=3):
        # Model info
        self.model = model.to(device).eval()
        self.name = "EpsMeshAttack"
        self.device = device
        self.targeted = False
        # Attack Vals
        if projection not in ["central"]: raise NotImplementedError
        if not 0 < eps <= 1: raise Exception("Mesh bound epsilon can only be 0 < eps <= 1")
        
        self.projection = projection
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        # Set seed
        self.seed = seed
        if seed is not None: seed_all(seed)
    
    def attack(self, data, labels, meshvectors, meshnormals):
        # data: (B, N, d); labels: (B, 1)
        data = data.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        meshvectors = meshvectors.clone().detach().to(self.device)
        meshnormals = meshnormals.clone().detach().to(self.device)
        # meshvectors: (B, N, v, d); meshnormals: (B, N, d)
        
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
                cost = loss(outputs.squeeze(), labels.squeeze())
            
            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_data,
                                       retain_graph=False, create_graph=False)[0]
            # print(grad.shape)
            grad_norms = torch.norm(grad.reshape(batch_size, -1), p=2, dim=1) + 1e-10  # nopep8
            grad = grad / grad_norms.reshape(batch_size, 1, 1)
            adv_data = adv_data.detach() + self.alpha * grad
            
            
            # project delta onto plane of mesh
            delta = adv_data - data
            delta = delta - (torch.einsum('bnd,bnd->bn', delta, meshnormals) / torch.einsum('bnd,bnd->bn', meshnormals, meshnormals)).unsqueeze(-1) * meshnormals 
            
            intersection = self.central_projection(meshvectors, delta)
            
            # delta_norms = torch.norm(delta.reshape(batch_size, -1), p=2, dim=1)
            # factor = self.eps / delta_norms
            # factor = torch.min(factor, torch.ones_like(delta_norms))
            # delta = delta * factor.reshape(-1, 1, 1)

            # adv_data = torch.clamp(data + delta, min=-1, max=1).detach()
            adv_data = intersection.detach()

        return adv_data

    def central_projection(self, triangle, delta):
        """_summary_

        Args:
            triangle (_type_): B, N, V, D
            delta (_type_): B, N, D
        """
        center = triangle.mean(dim=2) # B, N, D
        if self.eps < 1.0:
            triangle = center.unsqueeze(2) + self.eps*(triangle - center.unsqueeze(2))
            
        point = center + delta # B, N, D
        
        closest_ind = torch.norm(triangle-point.unsqueeze(-1), 2, dim=2).argmin(dim=2) # B, N
        next_ind = (closest_ind + 1) % 3 # Get random index by choosing next index
        
        B, N, V, D = triangle.shape
        
        closest_points = triangle[torch.arange(B), torch.arange(N), closest_ind]
        next_points = triangle[torch.arange(B), torch.arange(N), next_ind]
        
        cosagb = self.cosangle(closest_points, center, next_points)
        cospgb = self.cosangle(point, center, next_points)
        
        # Select true next index
        next_ind = torch.where(cosagb - cospgb < 0, next_ind, (next_ind+1)%3)

        # Solve Ax=b
        A = torch.cat([next_points.reshape(B, N, -1, 1) - closest_points.reshape(B, N, -1, 1), -(point - center).reshape(B, N, -1, 1)], dim=-1)
        
        b = (center - closest_points).reshape(B, N, -1, 1)
        solution = torch.linalg.lstsq(A, b).solution
        point_intersec = closest_points +  solution[:, :, 0] * (next_points - closest_points)
        
        # Compare length of delta and intersection vector to determine if the point is in triangle
        norm1 = torch.norm(point_intersec - center, 2, dim=2)
        norm2 = torch.norm(delta, 2, dim=2)
        
        result = torch.where((norm1 < norm2).unsqueeze(-1), point_intersec, center+delta)
        
        return result
        
    @staticmethod
    def cosangle(p0,p1,p2):
        """_summary_
        Args:
            p0 (_type_): B, N, D
            p1 (_type_): B, N, D
            p2 (_type_): B, N, D
        Returns:
            _type_: angle
        """
        v1 = p0 - p1 # B, N, D
        v2 = p2 - p1
        # θ = cos-1 [ (a · b) / (|a| |b|) ]
        costheta = torch.einsum("bnd,bnd->bn", v1, v2) / (torch.norm(v1, 2, dim=2) * torch.norm(v2, 2, dim=2))
        return costheta
        

    def perpendicular_projection(triangle, delta):
        pass

            
    def get_logits(self, inputs):
        logits = self.model(inputs.permute(0,2,1).to(self.device))
        return logits
        
    def save(self, dataloader, root=os.path.join(dirname(abspath(__file__)), "data_attacked"), file_name=None, args=None):
        true_labels, clean_preds, attack_preds = [], [], []
        attacked_batches = []
        for i,batch in enumerate(dataloader):
            print(f"batch {i}/{len(dataloader)}")
            pc, label = batch['pc'], batch['cate']
            meshvectors = batch['meshvectors']
            meshnormals = batch['meshnormals']
            
            shift = batch["shift"]
            scale = batch["scale"]
            # Forward
            logits = self.get_logits(pc)
            adv_data = self.attack(data=pc.clone()*scale+shift, labels=label, meshvectors=meshvectors, meshnormals=meshnormals)
            # Check attack
            attacked_logits = self.get_logits(adv_data)
            # Store for Acc Stats
            true_labels.append(label.detach().cpu().numpy())
            clean_preds.append(logits.detach().cpu().numpy().argmax(axis=1))
            attack_preds.append(attacked_logits.detach().cpu().numpy().argmax(axis=1))
            # Store Attacked Data
            batch["clean_pred"] = clean_preds[-1]
            batch["attack_pred"] = attack_preds[-1]
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
    
    def __str__(self):
        return(f"e-mesh-{self.projection}")

    
