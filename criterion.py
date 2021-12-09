import numpy as np
import torch
import pdb
import torch.nn as nn
from itertools import permutations
EPS = 1e-8

def loss_uPIT(num_spks, estimate,ref_1ch):

<<<<<<< HEAD
    """
    Get min L1Loss of each utterance (PIT)
    Args:
        estimate : [B,Spks,T,F]
        ref1_1ch = [B,T,F]
        ref2_1ch = [B,T,F]
        ref_1ch = [B,Spks,T,F]
    """
    B, Spks, T, F = estimate.size()
    L1Loss_ = torch.nn.L1Loss(reduction='none')
    if len(ref_1ch[0].shape) == 3:
        for spk_idx in range(num_spks):
            ref_1ch[spk_idx] = torch.unsqueeze(ref_1ch[spk_idx], dim=1)
        # ref1_1ch = torch.unsqueeze(ref1_1ch, dim=1)
        # ref2_1ch = torch.unsqueeze(ref2_1ch, dim=1)
    
    for spk_idx in range(num_spks):
        if spk_idx == 0:
            ref_ = ref_1ch[spk_idx]
        else:
            ref_ = torch.cat((ref_,ref_1ch[spk_idx]), dim=1)

    # ref_1ch  = torch.cat((ref1_1ch,ref2_1ch), dim=1)
    s_estimate = torch.unsqueeze(estimate, dim=2) #[B,Spks,1,T,F]
    s_ref_ = torch.unsqueeze(ref_, dim=1) #[B,1,Spks,T,F]
    # [B,Spks,Spks,T,F] -> pair_wise_Loss : [B,Spks,Spks] 
    L1_real = torch.sum(torch.abs(s_estimate.real-s_ref_.real),dim=[3,4])
    L1_imag = torch.sum(torch.abs(s_estimate.imag-s_ref_.imag),dim=[3,4])
    estimate_magnitude=  torch.abs(torch.sqrt(s_estimate.real**2 + s_estimate.imag**2 + EPS))
    L1_magnitude = torch.sum( torch.abs( estimate_magnitude - abs(s_ref_)),[3,4])
    
    """
    # L1_real = torch.sum(L1Loss_(s_estimate.real, s_ref_1ch.real),dim=[3,4])
    # L1_imag = torch.sum(L1Loss_(s_estimate.imag, s_ref_1ch.imag),dim=[3,4])
    # estimate_magnitude=  torch.abs(torch.sqrt(s_estimate.real**2 + s_estimate.imag**2 + EPS))
    # L1_magnitude = torch.sum(L1Loss_(estimate_magnitude, abs(s_ref_1ch)),dim=[3,4])
    """
    pair_wise_Loss = L1_real + L1_imag + L1_magnitude
    #perms:[Spks!,Spks]
    perms = ref_.new_tensor(list(permutations(range(Spks))), dtype=torch.long)
    index = torch.unsqueeze(perms,dim=2)
    #perms_one_hot:[Spks!,Spks,Spks]
    # perms_one_hot = ref_.new_zeros((*perms.size(), Spks),dtype=torch.double).scatter_(2, index, 1)
    perms_one_hot = ref_.new_zeros((*perms.size(), Spks),dtype=torch.float).scatter_(2, index, 1)

    #Loss:[B,Spks!]
    Loss = torch.einsum('bij,pij->bp', [pair_wise_Loss, perms_one_hot])
    #min_Loss_idx : [B]
    min_Loss_idx = torch.argmin(Loss,dim=1) 
    min_Loss,_ = torch.min(Loss,dim=1, keepdim=True)
    min_Loss = torch.mean(min_Loss)


    return min_Loss

def loss_uPIT_v1(num_spks, estimate,ref_1ch,zeros,alpha):
    """
    Get min L1Loss of each utterance (PIT)
    Args:
        estimate : [B,Spks,T,F]
        ref1_1ch = [B,T,F]
        ref2_1ch = [B,T,F]
        ref_1ch = [B,Spks,T,F]
    """
    B, Spks, T, F = estimate.size()
    L1Loss_ = torch.nn.L1Loss(reduction='none')

    if len(ref_1ch[0].shape) == 3:
        for spk_idx in range(num_spks):
            ref_1ch[spk_idx] = torch.unsqueeze(ref_1ch[spk_idx], dim=1)
        # ref1_1ch = torch.unsqueeze(ref1_1ch, dim=1)
        # ref2_1ch = torch.unsqueeze(ref2_1ch, dim=1)
    
    for spk_idx in range(num_spks):
        if spk_idx == 0:
            ref_ = ref_1ch[spk_idx]
        else:
            ref_ = torch.cat((ref_,ref_1ch[spk_idx]), dim=1)

    # ref_1ch  = torch.cat((ref1_1ch,ref2_1ch), dim=1)
    s_estimate = torch.unsqueeze(estimate, dim=2) #[B,Spks,1,T,F]
    s_ref_ = torch.unsqueeze(ref_, dim=1) #[B,1,Spks,T,F]
    # [B,Spks,Spks,T,F] -> pair_wise_Loss : [B,Spks,Spks] 
    L1_real = torch.sum(torch.abs(s_estimate.real-s_ref_.real),dim=[3,4])
    L1_imag = torch.sum(torch.abs(s_estimate.imag-s_ref_.imag),dim=[3,4])
    estimate_magnitude=  torch.abs(torch.sqrt(s_estimate.real**2 + s_estimate.imag**2 + EPS))
    L1_magnitude = torch.sum( torch.abs( estimate_magnitude - abs(s_ref_)),[3,4])
    L1_sub = alpha * torch.sum((torch.maximum(estimate_magnitude-abs(s_ref_),zeros)),[3,4])

    """
    # L1_real = torch.sum(L1Loss_(s_estimate.real, s_ref_1ch.real),dim=[3,4])
    # L1_imag = torch.sum(L1Loss_(s_estimate.imag, s_ref_1ch.imag),dim=[3,4])
    # estimate_magnitude=  torch.abs(torch.sqrt(s_estimate.real**2 + s_estimate.imag**2 + EPS))
    # L1_magnitude = torch.sum(L1Loss_(estimate_magnitude, abs(s_ref_1ch)),dim=[3,4])
    """
    pair_wise_Loss = L1_real + L1_imag + L1_magnitude + L1_sub
    #perms:[Spks!,Spks]
    perms = ref_.new_tensor(list(permutations(range(Spks))), dtype=torch.long)
    index = torch.unsqueeze(perms,dim=2)
    #perms_one_hot:[Spks!,Spks,Spks]
    perms_one_hot = ref_.new_zeros((*perms.size(), Spks),dtype=torch.float).scatter_(2, index, 1)
    #Loss:[B,Spks!]
    Loss = torch.einsum('bij,pij->bp', [pair_wise_Loss, perms_one_hot])
    #min_Loss_idx : [B]
    min_Loss_idx = torch.argmin(Loss,dim=1) 
    min_Loss,_ = torch.min(Loss,dim=1, keepdim=True)
    min_Loss = torch.mean(min_Loss)


    return min_Loss

def loss_Enhance(estimate,ref_1ch):

    """
    Args:
        estimate : [B,T,F]
        ref_1ch : [B,T,F]
    """

    B, Ch, T, F = estimate.size()
    L1Loss_ = torch.nn.L1Loss(reduction='none')
    L1_real = torch.sum(torch.abs(estimate.real-ref_1ch.real))
    L1_imag = torch.sum(torch.abs(estimate.imag-ref_1ch.imag))
    estimate_magnitude = torch.abs(torch.sqrt(estimate.real**2 + estimate.imag**2+ EPS))
    L1_magnitude = torch.sum(torch.abs(estimate_magnitude - abs(ref_1ch)))

    
    Loss = L1_real + L1_imag + L1_magnitude

    Loss = Loss / B
    
    return Loss
=======
        """
        Get min L1Loss of each utterance (PIT)
        Args:
            estimate : [B,Spks,T,F]
            ref1_1ch = [B,T,F]
            ref2_1ch = [B,T,F]
            ref_1ch = [B,Spks,T,F]
        """
        B, Spks, T, F = estimate.size()
        L1Loss_ = torch.nn.L1Loss(reduction='none')
        if len(ref_1ch[0].shape) == 3:
            for spk_idx in range(num_spks):
                ref_1ch[spk_idx] = torch.unsqueeze(ref_1ch[spk_idx], dim=1)
            # ref1_1ch = torch.unsqueeze(ref1_1ch, dim=1)
            # ref2_1ch = torch.unsqueeze(ref2_1ch, dim=1)
        
        for spk_idx in range(num_spks):
            if spk_idx == 0:
                ref_ = ref_1ch[spk_idx]
            else:
                ref_ = torch.cat((ref_,ref_1ch[spk_idx]), dim=1)
 
        # ref_1ch  = torch.cat((ref1_1ch,ref2_1ch), dim=1)
        s_estimate = torch.unsqueeze(estimate, dim=2) #[B,Spks,1,T,F]
        s_ref_ = torch.unsqueeze(ref_, dim=1) #[B,1,Spks,T,F]
        # [B,Spks,Spks,T,F] -> pair_wise_Loss : [B,Spks,Spks] 
        L1_real = torch.sum(torch.abs(s_estimate.real-s_ref_.real),dim=[3,4])
        L1_imag = torch.sum(torch.abs(s_estimate.imag-s_ref_.imag),dim=[3,4])
        estimate_magnitude=  torch.abs(torch.sqrt(s_estimate.real**2 + s_estimate.imag**2 + EPS))
        L1_magnitude = torch.sum( torch.abs( estimate_magnitude - abs(s_ref_)),[3,4])
        
        """
        # L1_real = torch.sum(L1Loss_(s_estimate.real, s_ref_1ch.real),dim=[3,4])
        # L1_imag = torch.sum(L1Loss_(s_estimate.imag, s_ref_1ch.imag),dim=[3,4])
        # estimate_magnitude=  torch.abs(torch.sqrt(s_estimate.real**2 + s_estimate.imag**2 + EPS))
        # L1_magnitude = torch.sum(L1Loss_(estimate_magnitude, abs(s_ref_1ch)),dim=[3,4])
        """
        pair_wise_Loss = L1_real + L1_imag + L1_magnitude
        #perms:[Spks!,Spks]
        perms = ref_.new_tensor(list(permutations(range(Spks))), dtype=torch.long)
        index = torch.unsqueeze(perms,dim=2)
        #perms_one_hot:[Spks!,Spks,Spks]
        # perms_one_hot = ref_.new_zeros((*perms.size(), Spks),dtype=torch.double).scatter_(2, index, 1)
        perms_one_hot = ref_.new_zeros((*perms.size(), Spks),dtype=torch.float).scatter_(2, index, 1)

        #Loss:[B,Spks!]
        Loss = torch.einsum('bij,pij->bp', [pair_wise_Loss, perms_one_hot])
        #min_Loss_idx : [B]
        min_Loss_idx = torch.argmin(Loss,dim=1) 
        min_Loss,_ = torch.min(Loss,dim=1, keepdim=True)
        min_Loss = torch.mean(min_Loss)


        return min_Loss
def loss_uPIT_v1(num_spks, estimate,ref_1ch,zeros,alpha):
        """
        Get min L1Loss of each utterance (PIT)
        Args:
            estimate : [B,Spks,T,F]
            ref1_1ch = [B,T,F]
            ref2_1ch = [B,T,F]
            ref_1ch = [B,Spks,T,F]
        """
        B, Spks, T, F = estimate.size()
        L1Loss_ = torch.nn.L1Loss(reduction='none')

        if len(ref_1ch[0].shape) == 3:
            for spk_idx in range(num_spks):
                ref_1ch[spk_idx] = torch.unsqueeze(ref_1ch[spk_idx], dim=1)
            # ref1_1ch = torch.unsqueeze(ref1_1ch, dim=1)
            # ref2_1ch = torch.unsqueeze(ref2_1ch, dim=1)
        
        for spk_idx in range(num_spks):
            if spk_idx == 0:
                ref_ = ref_1ch[spk_idx]
            else:
                ref_ = torch.cat((ref_,ref_1ch[spk_idx]), dim=1)
 
        # ref_1ch  = torch.cat((ref1_1ch,ref2_1ch), dim=1)
        s_estimate = torch.unsqueeze(estimate, dim=2) #[B,Spks,1,T,F]
        s_ref_ = torch.unsqueeze(ref_, dim=1) #[B,1,Spks,T,F]
        # [B,Spks,Spks,T,F] -> pair_wise_Loss : [B,Spks,Spks] 
        L1_real = torch.sum(torch.abs(s_estimate.real-s_ref_.real),dim=[3,4])
        L1_imag = torch.sum(torch.abs(s_estimate.imag-s_ref_.imag),dim=[3,4])
        estimate_magnitude=  torch.abs(torch.sqrt(s_estimate.real**2 + s_estimate.imag**2 + EPS))
        L1_magnitude = torch.sum( torch.abs( estimate_magnitude - abs(s_ref_)),[3,4])
        L1_sub = alpha * torch.sum((torch.maximum(estimate_magnitude-abs(s_ref_),zeros)),[3,4])

        """
        # L1_real = torch.sum(L1Loss_(s_estimate.real, s_ref_1ch.real),dim=[3,4])
        # L1_imag = torch.sum(L1Loss_(s_estimate.imag, s_ref_1ch.imag),dim=[3,4])
        # estimate_magnitude=  torch.abs(torch.sqrt(s_estimate.real**2 + s_estimate.imag**2 + EPS))
        # L1_magnitude = torch.sum(L1Loss_(estimate_magnitude, abs(s_ref_1ch)),dim=[3,4])
        """
        pair_wise_Loss = L1_real + L1_imag + L1_magnitude + L1_sub
        #perms:[Spks!,Spks]
        perms = ref_.new_tensor(list(permutations(range(Spks))), dtype=torch.long)
        index = torch.unsqueeze(perms,dim=2)
        #perms_one_hot:[Spks!,Spks,Spks]
        perms_one_hot = ref_.new_zeros((*perms.size(), Spks),dtype=torch.float).scatter_(2, index, 1)
        #Loss:[B,Spks!]
        Loss = torch.einsum('bij,pij->bp', [pair_wise_Loss, perms_one_hot])
        #min_Loss_idx : [B]
        min_Loss_idx = torch.argmin(Loss,dim=1) 
        min_Loss,_ = torch.min(Loss,dim=1, keepdim=True)
        min_Loss = torch.mean(min_Loss)


        return min_Loss

def loss_Enhance(estimate,ref_1ch):

        """
        Args:
            estimate : [B,T,F]
            ref_1ch : [B,T,F]
        """
        B, Ch, T, F = estimate.size()
        L1Loss_ = torch.nn.L1Loss(reduction='none')

        L1_real = torch.sum(torch.abs(estimate.real-ref_1ch.real))
        L1_imag = torch.sum(torch.abs(estimate.imag-ref_1ch.imag))
        estimate_magnitude = torch.abs(torch.sqrt(estimate.real**2 + estimate.imag**2+ EPS))
        L1_magnitude = torch.sum(torch.abs(estimate_magnitude - abs(ref_1ch)))

        
        Loss = L1_real + L1_imag + L1_magnitude
        
        return Loss
>>>>>>> 7431d9618a519d5bf78594445b6810a1a197388d
