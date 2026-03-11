# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 16:11:42 2025

@author: ZML
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    """Dice Loss + BCE Loss"""
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = bce_weight
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, predict, target):
        """
        Args:
            predict: Predicted probability map (B, C, H, W) or (B, C, D, H, W), model output logits
            target: Ground truth labels (B, H, W) or (B, D, H, W)
        """
        num_classes = predict.shape[1]
        if num_classes == 1:
            #BCE Loss
            #predict (B, 1, H, W), (B, 1, D, H, W)
            #target (B, H, W), (B, D, H, W)
            ce = self.bce_loss(predict, target.float().unsqueeze(1)) #(B, 1, H, W), (B, 1, D, H, W)
            # Dice Loss
            predict = torch.sigmoid(predict).squeeze(1) #(B, H, W), (B, D, H, W)
            if predict.dim() == 3:
                intersection = torch.sum(predict * target, dim=(1, 2))#(B, H, W)
                union = torch.sum(predict, dim=(1, 2)) + torch.sum(target, dim=(1, 2)) #(B, H, W)
            if predict.dim() == 4:
                intersection = torch.sum(predict * target, dim=(1, 2, 3)) #(B, D, H, W)
                union = torch.sum(predict, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3))
        else:
            # CE Loss
            #predict (B, C, H, W), (B, C, D, H, W)
            #target (B, H, W), (B, D, H, W)
            ce = self.ce_loss(predict, target.long())
            # Dice Loss
            predict = torch.softmax(predict, dim=1) #(B, C, H, W), (B, C, D, H, W)
            if predict.dim() == 4:
                target_one_hot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()#(B, C, H, W)
                intersection = torch.sum(predict * target_one_hot, dim=(2, 3))
                union = torch.sum(predict, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
            if predict.dim() == 5:
                target_one_hot = F.one_hot(target.long(), num_classes).permute(0, 4, 1, 2, 3).float()#(B, C, D, H, W)
                intersection = torch.sum(predict * target_one_hot, dim=(2, 3, 4))
                union = torch.sum(predict, dim=(2, 3, 4)) + torch.sum(target_one_hot, dim=(2, 3, 4))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        # 组合损失
        return self.ce_weight * ce + self.dice_weight * dice_loss
    
    
class DiceCELoss(nn.Module):
    """Dice Loss + BCE Loss"""
    def __init__(self, dice_weight=0.5, ce_weight=0.5, smooth=1e-6):
        super(DiceCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, predict, target):
        """
        Args:
            predict: [N, C, H, W] model output logits
            target: [N, 1, H, W] ground truth labels
        """
        num_classes = predict.shape[1]
        
        if target.dim() == 4:
            target = target.squeeze(1)
            
        # BCE Loss
        ce = self.ce_loss(predict, target.long())
        
        # Dice Loss
        predict_softmax = torch.softmax(predict, dim=1)
        target_one_hot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()
            
        intersection = torch.sum(predict_softmax * target_one_hot, dim=(2, 3))
        union = torch.sum(predict_softmax, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        # 组合损失
        return self.ce_weight * ce + self.dice_weight * dice_loss
    
    
class DiceSoftmaxLoss(nn.Module):
    """Dice Loss + BCE Loss"""
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss() #CrossEntropyLoss()

    def forward(self, predict, target):
        """
        Args:
            predict: [N, 2, H, W] model output logits
            target: [N, 1, H, W] ground truth labels
        """
        # BCE Loss
        if target.dim() == 4:
            target = target.squeeze(1)
        
        bce = self.ce_loss(predict, target.long())
        
        # Dice Loss
        predict_softmax = torch.softmax(predict, dim=1)

        target_one_hot = F.one_hot(target.long(), num_classes=2).permute(0, 3, 1, 2).float()
            
        intersection = (predict_softmax * target_one_hot).sum(dim=(2, 3))
        union = predict_softmax.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice_per_class = (2. * intersection + self.smooth) / (union + self.smooth)
        
        dice_loss = 1 - dice_per_class.mean(dim=0)
        
        return self.bce_weight * bce + self.dice_weight * dice_loss
    
    
class BinaryKLLoss_(nn.Module):
    def __init__(self, temperature=4.0):
        super(BinaryKLLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, student_logits, teacher_logits):
        """
        student_logits: [B, 1, H, W] 
        teacher_logits: [B, 1, H, W] 道logits
        """
        student_probs = torch.sigmoid(student_logits / self.temperature)
        teacher_probs = torch.sigmoid(teacher_logits / self.temperature)
        
        kl_loss = teacher_probs * torch.log(teacher_probs / student_probs) + \
                 (1 - teacher_probs) * torch.log((1 - teacher_probs) / (1 - student_probs))
        
        return kl_loss.mean() * (self.temperature ** 2)
    
class BinaryKLLoss(nn.Module):
    def __init__(self, temperature=4.0):
        super(BinaryKLLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, student_logits, teacher_logits):
        """
        student_logits: [B, 1, H, W] 
        teacher_logits: [B, 1, H, W] 
        """
        student_log_probs = F.logsigmoid(student_logits / self.temperature)
        student_log_neg_probs = F.logsigmoid(-student_logits / self.temperature)
        
        teacher_probs = torch.sigmoid(teacher_logits / self.temperature)
        
        # KL(p||q) = p * log(p/q) + (1-p) * log((1-p)/(1-q))
        # = p * (log(p) - log(q)) + (1-p) * (log(1-p) - log(1-q))
        kl_loss = teacher_probs * (torch.log(teacher_probs.clamp(1e-8)) - student_log_probs) + \
                 (1 - teacher_probs) * (torch.log((1 - teacher_probs).clamp(1e-8)) - student_log_neg_probs)


        return kl_loss.mean() * (self.temperature ** 2)
    
    
class BinaryDistillationLoss(nn.Module):
    def __init__(self, alpha=0.7, temperature=4.0):
        super(BinaryDistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.dice_bce_loss = DiceBCELoss()
        self.kl_loss = BinaryKLLoss(temperature)
        
    def forward(self, student_logits, teacher_logits, labels):
        """
        student_logits: [B, 1, H, W] Student model output
        teacher_logits: [B, 1, H, W] Teacher model output
        labels: [B, 1, H, W] Ground truth labels (0 or 1)
        """
        # 基础二分类交叉熵损失
        bce_loss = self.dice_bce_loss(student_logits, labels.float())

        # 蒸馏损失
        distill_loss = self.kl_loss(student_logits, teacher_logits)
        # 总损失
        total_loss = (1 - self.alpha) * bce_loss + self.alpha * distill_loss
        
        return total_loss, bce_loss, distill_loss
    

    