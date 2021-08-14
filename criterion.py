import numpy as np
import torch
import pdb
import torch.nn as nn
from itertools import permutations
EPS = 1e-8

def loss_uPIT(estimate,ref1_1ch, ref2_1ch):
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

        if len(ref1_1ch.shape) == 3:
            ref1_1ch = torch.unsqueeze(ref1_1ch, dim=1)
            ref2_1ch = torch.unsqueeze(ref2_1ch, dim=1)
        ref_1ch  = torch.cat((ref1_1ch,ref2_1ch), dim=1)
        s_estimate = torch.unsqueeze(estimate, dim=2) #[B,Spks,1,T,F]
        s_ref_1ch = torch.unsqueeze(ref_1ch, dim=1) #[B,1,Spks,T,F]
        # [B,Spks,Spks,T,F] -> pair_wise_Loss : [B,Spks,Spks] 
        L1_real = torch.sum(torch.abs(s_estimate.real-s_ref_1ch.real),dim=[3,4])
        L1_imag = torch.sum(torch.abs(s_estimate.imag-s_ref_1ch.imag),dim=[3,4])
        estimate_magnitude=  torch.abs(torch.sqrt(s_estimate.real**2 + s_estimate.imag**2 + EPS))
        L1_magnitude = torch.sum( torch.abs( estimate_magnitude - abs(s_ref_1ch)),[3,4])

        """
        # L1_real = torch.sum(L1Loss_(s_estimate.real, s_ref_1ch.real),dim=[3,4])
        # L1_imag = torch.sum(L1Loss_(s_estimate.imag, s_ref_1ch.imag),dim=[3,4])
        # estimate_magnitude=  torch.abs(torch.sqrt(s_estimate.real**2 + s_estimate.imag**2 + EPS))
        # L1_magnitude = torch.sum(L1Loss_(estimate_magnitude, abs(s_ref_1ch)),dim=[3,4])
        """
        pair_wise_Loss = L1_real + L1_imag + L1_magnitude
        #perms:[Spks!,Spkss]
        perms = ref_1ch.new_tensor(list(permutations(range(Spks))), dtype=torch.long)
        index = torch.unsqueeze(perms,dim=2)
        #perms_one_hot:[Spks!,Spks,Spks]
        perms_one_hot = ref_1ch.new_zeros((*perms.size(), Spks),dtype=torch.double).scatter_(2, index, 1)
        #Loss:[B,Spks!]
        Loss = torch.einsum('bij,pij->bp', [pair_wise_Loss, perms_one_hot])
        #min_Loss_idx : [B]
        min_Loss_idx = torch.argmin(Loss,dim=1) 
        min_Loss,_ = torch.min(Loss,dim=1, keepdim=True)
        min_Loss = torch.mean(min_Loss)


        return min_Loss

# def loss_uPIT(estimate, ref1_1ch, ref2_1ch):
#     """
#     Building upon utterance-level PIT(uPIT),
#     The loss function is defined on the predicted RI components and the resulting magnitude
#     estimate : [B,2,T,F]
#     ref_1ch : [ref1_1ch, ref2_1ch] == [B,2,T,F]
#     """
#     if len(ref1_1ch.shape) == 3:
#         ref1_1ch = torch.unsqueeze(ref1_1ch, dim=1)
#         ref2_1ch = torch.unsqueeze(ref2_1ch, dim=1)
#     ref_1ch  = torch.cat((ref1_1ch,ref2_1ch), dim=1)
#     B, Spk, T, F = estimate.size()
#     Loss1 = 0
#     pdb.set_trace()
#     for Spk_idx in range(Spk):
#         """
#         Sum of loss by the number of speakers
#         """
#         L1_real
        # L1_real = torch.sum( torch.abs(estimate[:,Spk_idx,:,:].real - ref_1ch[:,Spk_idx,:,:].real), [0,1,2]) 
#         # L1_imag = torch.sum( torch.abs(estimate[:,Spk_idx,:,:].imag - ref_1ch[:,Spk_idx,:,:].imag), [0,1,2])
#         # L1_magnitude = torch.sum( torch.abs( torch.sqrt(estimate[:,Spk_idx,:,:].real**2 + estimate[:,Spk_idx,:,:].imag**2+1e-10) \
#                                                 # - abs(ref_1ch[:,Spk_idx,:,:])),[0,1,2])
#         Loss1 += L1_real + L1_imag + L1_magnitude
#     # Loss2= 0
#     # for Spk_idx in range(Spk):
#     #     """
#     #     Sum of loss by the number of speakers
#     #     """
#     #     if Spk_idx == 1:
#     #         ref_idx = 0
#     #     else:
#     #         ref_idx = 1

#     #     L2_real = torch.sum( torch.abs(estimate[:,Spk_idx,:,:].real - ref_1ch[:,ref_idx,:,:].real), [0,1,2]) 
#     #     L2_imag = torch.sum( torch.abs(estimate[:,Spk_idx,:,:].imag - ref_1ch[:,ref_idx,:,:].imag), [0,1,2])
#     #     L2_magnitude = torch.sum( torch.abs( torch.sqrt(estimate[:,Spk_idx,:,:].real**2 + estimate[:,Spk_idx,:,:].imag**2+1e-10) \
#     #                                             - torch.abs(ref_1ch[:,ref_idx,:,:])),[0,1,2])
#     #     Loss2 += L2_real + L2_imag + L2_magnitude    
    
#     # return torch.min(Loss1,Loss2)
#     return Loss1

    

