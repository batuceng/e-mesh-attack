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
    def __init__(self, model, device, projection="perpendicular", eps=1.0, alpha=0.05, steps=10, random_start=False, seed=3):
        # Model info
        self.model = model.to(device).eval()
        self.name = "EpsMeshAttack"
        self.device = device
        self.targeted = False
        # Attack Vals
        if projection not in ["central","perpendicular"]: raise NotImplementedError
        if not 0 < eps <= 1: raise Exception("Mesh bound epsilon can only be 0 < eps <= 1")
        
        self.projection = projection
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.stepscale = 5
        # Set seed
        self.seed = seed
        if seed is not None: seed_all(seed)
    
    def attack(self, data, labels, meshvectors, meshnormals):
        # data: (B, N, d); labels: (B, 1)
        data = data.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        meshvectors = meshvectors.clone().detach().to(self.device)
        meshnormals = meshnormals.clone().detach().to(self.device)
        center = meshvectors.mean(dim=2).unsqueeze(2)
        # meshvectors: (B, N, v, d); meshnormals: (B, N, d); barycenters: (B, N, d)
        
        if self.eps < 1.0:
            meshvectors = center + self.eps*(meshvectors - center)
            
        # dist = torch.norm(meshvectors - center, 2, dim=3, keepdim=True)
        # alpha = torch.max(dist, dim=2)[0] / (self.steps//self.stepscale)
        
        
        loss = nn.CrossEntropyLoss()
        
        adv_data = data.clone().detach()
        batch_size = data.shape[0]
        
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
            
            
            # project delta onto plane of mesh; projv_a = (a.v / ||v||^2) * v
            delta = adv_data - data
            delta = delta - (self.dotprod(delta, meshnormals) / self.dotprod(meshnormals, meshnormals)) * meshnormals 
            
            # Scale the triangle around barycenter based on eps value
            
                
            if self.projection=="central":
                intersection = self.central_projection(meshvectors, delta)
            elif self.projection=='perpendicular':
                intersection = self.perpendicular_projection(meshvectors, delta)
            else:
                raise NotImplementedError
            
            # delta_norms = torch.norm(delta.reshape(batch_size, -1), p=2, dim=1)
            # factor = self.eps / delta_norms
            # factor = torch.min(factor, torch.ones_like(delta_norms))
            # delta = delta * factor.reshape(-1, 1, 1)

            # adv_data = torch.clamp(data + delta, min=-1, max=1).detach()
            adv_data = intersection.detach()

        return adv_data
    
    # Dot product between for batched tensors a & b along dimension d. a: (B,N,d), b: (B,N,d), out: (B,N,1)
    @staticmethod
    def dotprod(a,b):
        return torch.einsum('bnd,bnd->bn', a, b).unsqueeze(-1)

    def central_projection(self, triangle, delta):
        """_summary_

        Args:
            triangle (_type_): B, N, V, D
            delta (_type_): B, N, D
        """
        # Barycenter of each triangle, a point named G
        center = triangle.mean(dim=2) # B, N, D

        # Current point on the same plane, a point named P
        point = center + delta # B, N, D
        
        # Select closest corner to the point, a point named A
        closest_ind = torch.norm(triangle-point.unsqueeze(-1), 2, dim=2).argmin(dim=2) # B, N
        # Get another random corner by choosing next index, a point named B
        next_ind = (closest_ind + 1) % 3 
        
        B, N, V, D = triangle.shape
        
        closest_points = triangle[torch.arange(B), torch.arange(N), closest_ind]
        next_points = triangle[torch.arange(B), torch.arange(N), next_ind]
        # Calculate cosine values for angle AGB & angle PGB
        cosagb = self.cosangle(closest_points, center, next_points)
        cospgb = self.cosangle(point, center, next_points)
        
        # Select the line that is cloesest to the Point
        next_ind = torch.where(cosagb - cospgb < 0, next_ind, (next_ind+1)%3)

        # Calculate Intersection of lines PG & AB, Solve Ax=b
        A = torch.cat([next_points.reshape(B, N, -1, 1) - closest_points.reshape(B, N, -1, 1), -(point - center).reshape(B, N, -1, 1)], dim=-1)
        b = (center - closest_points).reshape(B, N, -1, 1)
        solution = torch.linalg.lstsq(A+1e-8, b).solution
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
        costheta = torch.einsum("bnd,bnd->bn", v1, v2) / ((torch.norm(v1, 2, dim=2) * torch.norm(v2, 2, dim=2)) + 1e-8)
        return costheta
        

    def perpendicular_projection(self, triangle, delta):
        A, B, C = triangle[:,:,0,:], triangle[:,:,1,:], triangle[:,:,2,:]
        center = triangle.mean(axis=2)
        P = center + delta
        res = P.clone()
        
        AB = B-A
        AC = C-A

        AP = P-A
        dA1 = self.dotprod(AB,AP)
        dA2 = self.dotprod(AC,AP)
        # if( dA1<=0 and dA2 <=0):
        pos1 = torch.logical_and((dA1<=0),(dA2<=0))
        res = torch.where(pos1, A, res)
        # print(f"Condition 1 triggered:{pos1.sum()}")
        
        BP = P - B
        dB1 = self.dotprod(AB, BP)
        dB2 = self.dotprod(AC, BP)
        # if( dB1 >= 0 and dB2 <=0 ):
        pos2 = torch.logical_and((dB1>=0),(dB2<=0))
        res = torch.where(pos2, B, res)
        # print(f"Condition 2 triggered:{pos2.sum()}")

        CP = P - C
        dC1 = self.dotprod(AB,CP)
        dC2 = self.dotprod(AC,CP)
        # if( dC2 >= 0 and dC1 <= dC2 ):
        pos3 = torch.logical_and((dC2>=0),(dC1<=dC2))
        res = torch.where(pos3, C, res)
        # print(f"Condition 3 triggered:{pos3.sum()}")
        
        EdgeAB = dA1*dB2 - dB1*dA2
        # if( EdgeAB <= 0 and dA1 >= 0 and dB1 <=0):
        pos4 = torch.logical_and(torch.logical_and((EdgeAB<=0),(dA1>=0)), (dB1<=0))
        res = torch.where(pos4, (self.dotprod(AP,AB)/(self.dotprod(AB,AB)+1e-8)*AB)+A, res)
        # print(f"Condition 4 triggered:{pos4.sum()}")
        
        BC = C - B
        EdgeBC = dB1*dC2 - dC1*dB2; 
        # if( EdgeBC <= 0 and (dB2-dB1)>=0 and (dC1-dC2)>=0):
        pos5 = torch.logical_and(torch.logical_and((EdgeBC<=0),((dB2-dB1)>=0)), ((dC1-dC2)>=0))
        res = torch.where(pos5, (self.dotprod(BP,BC)/(self.dotprod(BC,BC)+1e-8)*BC)+B, res)
        # print(f"Condition 5 triggered:{pos5.sum()}")

        EdgeAC = dC1*dA2 - dA1*dC2
        # if( EdgeAC <= 0 and dA2>=0 and dC2<=0 ):
        pos6 = torch.logical_and(torch.logical_and((EdgeAC <= 0),(dA2>=0)), (dC2<=0))
        res = torch.where(pos6, (self.dotprod(AP,AC)/(self.dotprod(AC,AC)+1e-8)*AC)+A, res)
        # print(f"Condition 6 triggered:{pos6.sum()}")
        
        return res            
        

            
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
            adv_data = self.attack(data=pc.clone(), labels=label, meshvectors=meshvectors, meshnormals=meshnormals)
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

    
