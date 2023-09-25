import torch
import numpy as np
from sklearn.metrics import accuracy_score
import os
from os.path import dirname, abspath
import random
import json

def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
class Attack(object):
    def __init__(self, name, model, device, seed):
        self.model = model
        self.model.to(device).eval()
        self.name = name
        self.seed = seed
        self.device = device
        if seed is not None: seed_all(seed)
        
    def attack(self):
        raise NotImplementedError
        
    def get_logits(self, inputs):
        logits = self.model(inputs.permute(0,2,1).to(self.device))
        return logits
        
    def save(self, dataloader, root=os.path.join(dirname(dirname(abspath(__file__))), "data_attacked"), file_name=None, args=None):
        true_labels, clean_preds, attack_preds = [], [], []
        attacked_batches = []
        for i,batch in enumerate(dataloader):
            print(f"batch {i}/{len(dataloader)}")
            pc, label = batch['pc'], batch['cate']
            # Forward
            logits = self.get_logits(pc)
            adv_data = self.attack(data=pc, labels=label)
            # Check attack
            attacked_logits = self.get_logits(adv_data)
            # Store for Acc Stats
            true_labels.append(label.detach().cpu().numpy())
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
                if self.name == 'PGD_Linf' or self.name == 'PGDL2':
                    FILE_NAME = os.path.join(DATA_DIR, f'{datasetname}_{modelname}_{self.name}_eps_{self.eps}')
                elif self.name == 'PointDrop':
                    FILE_NAME = os.path.join(DATA_DIR, f'{datasetname}_{modelname}_{self.name}_num_points_{self.num_points}')
                elif self.name == "PointAdd":
                    FILE_NAME = os.path.join(DATA_DIR, f'{datasetname}_{modelname}_{self.name}_num_points_{self.num_points}')
                elif self.name == 'cw':
                    FILE_NAME = os.path.join(DATA_DIR, f'{datasetname}_{modelname}_{self.name}_c_{self.c}_kappa_{self.kappa}_lr_{self.lr}')
                elif self.name == 'knn':
                    FILE_NAME = os.path.join(DATA_DIR, f'{datasetname}_{modelname}_{self.name}_c_{self.c}_kappa_{self.kappa}_lr_{self.lr}')
                elif self.name == 'VANILA':
                    FILE_NAME = os.path.join(DATA_DIR, f'{datasetname}_{modelname}_{self.name}_kappa_{self.kappa}_lr_{self.lr}')
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
