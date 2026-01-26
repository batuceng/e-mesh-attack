import torch
import numpy as np
from sklearn.metrics import accuracy_score
import os
from os.path import dirname, abspath
import random
import json
import time

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
        # Logger to list acc per step
        self.logger = []
        
    def attack(self):
        raise NotImplementedError
        
    def get_logits(self, inputs):
        logits = self.model(inputs.permute(0,2,1).to(self.device))
        return logits
        
    def save(self, dataloader, root=os.path.join(dirname(dirname(abspath(__file__))), "data_attacked"), file_name=None, args=None):
        true_labels, clean_preds, attack_preds = [], [], []
        attacked_batches = []
        # Wall-clock timing. We report per-sample averages to match typical "attack runtime" tables.
        attack_time_per_sample_s = []
        total_time_per_sample_s = []
        for i,batch in enumerate(dataloader):
            print(f"batch {i}/{len(dataloader)}")
            pc, label = batch['pc'], batch['cate']
            bs = int(pc.shape[0])
            t0 = time.perf_counter()
            # Forward
            logits = self.get_logits(pc)
            t_attack0 = time.perf_counter()
            adv_data = self.attack(data=pc, labels=label)
            t_attack1 = time.perf_counter()
            # Check attack
            attacked_logits = self.get_logits(adv_data)
            t1 = time.perf_counter()

            # Normalize by batch size so results are comparable even if bs != 1.
            if bs > 0:
                attack_time_per_sample_s.append((t_attack1 - t_attack0) / bs)
                total_time_per_sample_s.append((t1 - t0) / bs)
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
            
        if self.logger == []:
            loss_per_step, acc_per_step = None, None
        else:
            # (no_of_batches, no_of_steps, 3)->(no_of_steps, 3), Last axis: [Total Loss, TP Count, Data Count]
            logger_lists = np.array(self.logger).sum(axis=0)
            loss_per_step = (logger_lists[:,0]/logger_lists[:,2]).tolist()
            acc_per_step = (logger_lists[:,1]/logger_lists[:,2]).tolist()
            
        clean_acc = accuracy_score(np.concatenate(true_labels), np.concatenate(clean_preds))
        attack_acc = accuracy_score(np.concatenate(true_labels), np.concatenate(attack_preds))
        datasetname, modelname = dataloader.dataset.__class__.__name__, self.model.__class__.__name__
        print(f"Clean Acc:{clean_acc} on dset:{datasetname} model:{modelname} atk:{self.name}")
        print(f"Attacked Acc:{attack_acc} on dset:{datasetname} model:{modelname} atk:{self.name}")

        if len(attack_time_per_sample_s) > 0:
            atk_mean = float(np.mean(attack_time_per_sample_s))
            atk_std = float(np.std(attack_time_per_sample_s))
            tot_mean = float(np.mean(total_time_per_sample_s))
            tot_std = float(np.std(total_time_per_sample_s))
            print(f"Avg Attack Time (per sample): {atk_mean:.4f}s \u00b1 {atk_std:.4f}s (n={len(attack_time_per_sample_s)})")
            print(f"Avg Total Time  (per sample): {tot_mean:.4f}s \u00b1 {tot_std:.4f}s (n={len(total_time_per_sample_s)})")
        else:
            atk_mean = atk_std = tot_mean = tot_std = None
        
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
                    # append followint to the dictonary
                    # args |= ({"data_path": FILE_NAME, "clean_acc": clean_acc, "attack_acc": attack_acc}) # Python 3.9+
                    args.update({"data_path": FILE_NAME, "clean_acc": clean_acc, "attack_acc": attack_acc,
                                 "loss_curve": loss_per_step, "acc_curve:": acc_per_step,
                                 "avg_attack_time_s": atk_mean, "std_attack_time_s": atk_std,
                                 "avg_total_time_s": tot_mean, "std_total_time_s": tot_std})
                    json.dump(args, fp, indent=4)
            
        return attacked_batches
