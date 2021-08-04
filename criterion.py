import numpy as np
import torch
import pdb

def loss_uPIT(estimate, ref1_1ch, ref2_1ch):
    """
    Building upon utterance-level PIT(uPIT),
    The loss function is defined on the predicted RI components and the resulting magnitude
    estimate : [B,Spk,T,F]
    ref_1ch : [ref1_1ch, ref2_1ch] == [B,2,T,F]
    """
    if len(ref1_1ch.shape) == 3:
        ref1_1ch = torch.unsqueeze(ref1_1ch, dim=1)
        ref2_1ch = torch.unsqueeze(ref2_1ch, dim=1)
    ref_1ch  = torch.cat((ref1_1ch,ref2_1ch), dim=1)
    B, Spk, T, F = estimate.size()
    Loss = 0
    for Spk_idx in range(Spk):
        """
        Sum of loss by the number of speakers
        """
        L1_real = torch.sum( torch.abs(estimate[:,Spk_idx,:,:].real - ref_1ch[:,Spk_idx,:,:].real), [0,1,2]) 
        L1_imag = torch.sum( torch.abs(estimate[:,Spk_idx,:,:].imag - ref_1ch[:,Spk_idx,:,:].imag), [0,1,2])
        L1_magnitude = torch.sum( torch.abs( torch.sqrt(estimate[:,Spk_idx,:,:].real**2 + estimate[:,Spk_idx,:,:].imag**2) \
                                                - abs(ref_1ch[:,Spk_idx,:,:])),[0,1,2])
        Loss += L1_real + L1_imag + L1_magnitude


    
    return Loss