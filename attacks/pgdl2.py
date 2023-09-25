import torch
import torch.nn as nn
from .attack import Attack


class PGDL2(Attack):
    def __init__(self, model, device, eps=1.0, alpha=0.02, steps=10, random_start=True, seed=3):
        super().__init__("PGDL2", model, device, seed)
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
        batch_size = data.shape[0]
        
        if self.random_start:
            # Starting at a uniformly random point
            delta = torch.empty_like(adv_data).normal_()
            d_flat = delta.view(batch_size, -1)
            n = d_flat.norm(p=2, dim=1).view(batch_size, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta = (delta*r*self.eps)/n
            adv_data = torch.clamp(adv_data + delta, min=0, max=1).detach()
        
        for _ in range(self.steps):
            adv_data.requires_grad = True
            outputs = self.get_logits(adv_data)
            
            # Calculate loss
            if self.targeted:
                # cost = -loss(outputs, target_labels)
                raise NotImplementedError
            else:
                cost = loss(outputs, labels)
            
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
    
    def __str__(self):
        return("pgdl2")
            

    
