import torch
import torch.nn as nn
from .attack import Attack


class PGDLinf(Attack):
    def __init__(self, model, device, eps=0.05, alpha=0.01, steps=10, random_start=True, seed=3):
        super().__init__("PGDLinf", model, device, seed)
        # Attack Vals
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default']
        # self.supported_mode = ['default', 'targeted']
        self.device = device
        self.targeted = False

    
    def attack(self, data, labels):
        # data: (B, N, d); labels: (B, 1)
        data = data.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        loss = nn.CrossEntropyLoss()
        
        adv_data = data.clone().detach()
        
        if self.random_start:
            adv_data = adv_data + \
                torch.empty_like(adv_data).uniform_(-self.eps, self.eps)
            adv_data = torch.clamp(adv_data, min=-1, max=1).detach()
        
        for _ in range(self.steps):
            adv_data.requires_grad = True
            outputs = self.get_logits(adv_data)
            # print(adv_data.requires_grad, outputs.requires_grad)
            
            # Calculate loss
            if self.targeted:
                # cost = -loss(outputs, target_labels)
                raise NotImplementedError
            else:
                cost = loss(outputs, labels)
            
            # Update adversarial data
            grad = torch.autograd.grad(cost, adv_data,
                                       retain_graph=False, create_graph=False)[0]
            # Linf Projection
            adv_data = adv_data.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_data - data,
                                min=-self.eps, max=self.eps)
            adv_data = torch.clamp(data + delta, min=-1, max=1).detach()
        return adv_data
            
            

    
