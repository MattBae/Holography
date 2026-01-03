# we want to do 2d and 3d conjugate gradient
# write the 2d loss function given target and prediction return mse loss (no need to write this??)
# we have the ideal amplitude and target amplitude 
# loss should take phase,target distribution as input 
# write a function for forward propagation
import torch
from slmsuite.holography.algorithms import FeedbackHologram
from torchmin import minimize
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from src.plots import fg

def periodic_bb(xdata, A, w, xc, yc, offset,n,p):
    x = xdata[0]
    y = xdata[1]
    theta = np.arctan2(y - yc, x - xc)
    I = A * ((x - xc)**2 + (y - yc)**2) * np.exp(-((x - xc)**2 + (y - yc)**2) / w**2)*(4 + 1*np.cos(n* theta + p)) + offset
    return I

def build_pbb_target(pad_factor=4,n=0,H = 1024,W = 1280,width = 7.07):
    """ Build a target tensor for the SLM."""
    y = torch.arange(H*pad_factor)
    x = torch.arange(W*pad_factor)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    return periodic_bb((xx, yy), A=0.38526713, w=width*pad_factor/4, xc = W*pad_factor / 2, yc =H*pad_factor / 2, offset=0,n=n,p=0)



def angular_modulation_mask(H, W, center_y, center_x, a=0.5, b=0.5, n=3, device='cpu'):
    y = torch.arange(H, device=device).reshape(-1, 1) - center_y
    x = torch.arange(W, device=device).reshape(1, -1) - center_x
    theta = torch.atan2(y.float(), x.float())  # (H, W)
    modulation = a + b * torch.sin(n * theta)
    return modulation

# modulation_mask = angular_modulation_mask(
#     H=m.pad_H,
#     W=m.pad_W,
#     center_y=m.pad_H // 2,
#     center_x=m.pad_W // 2,
#     a=2,
#     b=0.4,
#     n=5,
#     device=a_out.device
# )

def build_input(H = 1024, W = 1280, width = 2.53538014e+02):
    """ Build an input tensor for the SLM."""
    y = torch.arange(H)
    x = torch.arange(W)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    return fg((xx, yy), 4.27754945e-01, width, W / 2, H / 2, 0)

class SLM():
    def __init__(self,H=1024,W=1280):
        self.H = H
        self.W = W
        self.xc = self.W // 2
        self.yc = self.H // 2
class paddedSLM(SLM):
    """ Class for SLM with padding and cropping.
    Args:
        pad_factor (int): Factor by which to pad the SLM dimensions.
    """
    def __init__(self, pad_factor=4,H=1024,W=1280):
        super().__init__(H=H,W=W)
        self.pad_factor = pad_factor
        self.pad_H = self.H * pad_factor
        self.pad_W = self.W * pad_factor
        self.start_y = (self.pad_H - self.H) // 2
        self.start_x = (self.pad_W - self.W) // 2
        self.crop_y = slice(self.start_y, self.start_y + self.H)
        self.crop_x = slice(self.start_x, self.start_x + self.W)
        self.yc = self.pad_H // 2
        self.xc = self.pad_W // 2
        self.q_tensor = torch.tensor([[((i-self.yc)**2 + (j-self.xc)**2) for j in range(self.pad_W)] for i in range(self.pad_H)], dtype=torch.float32)
    def pad(self, tensor):
        """ Pad the input tensor to the padded SLM dimensions.
        Args:
            tensor (torch.Tensor): Input tensor, shape (H, W).
            Returns: padded_tensor (torch.Tensor): Padded tensor, shape (pad_H, pad_W)"""
        return F.pad(tensor, (self.start_x, self.pad_W - self.start_x - self.W, self.start_y, self.pad_H - self.start_y - self.H), mode='constant', value=0)
    def unpad(self, tensor):
        """ Unpad the input tensor to the original SLM dimensions.
        Args:
            tensor (torch.Tensor): Input tensor, shape (pad_H, pad_W).
            Returns: unpadded_tensor (torch.Tensor): Unpadded tensor, shape (H, W)"""
        return tensor[self.crop_y, self.crop_x]
    def build_target_from_phase(self,  phase):
        """ Build the target tensor for the SLM from a phase tensor.
        Args:
            phase (torch.Tensor): Phase tensor, shape (H, W).
            Returns: target_tensor (torch.Tensor): Padded target tensor, shape (pad_H, pad_W)"""
        target = torch.abs(self.forward_propagate(torch.exp(1j * self.pad(phase))*self.A_in))  # Convert phase to complex field
        self.A_t = torch.tensor(target, dtype=torch.float32)
        return target
    def lens(self,z):
        """ create a lens phase with a quadratic phase near focus
        Args:
            z (float): Propagation distance.
            Returns: lens (torch.Tensor): Lens phase, shape (H, W)"""
        
        return torch.exp(1j * z * self.q_tensor)
    def forward_propagate(self,F_in,z=0):
        """ forward propagation of the field F_in through a lens with a quadratic phase near focus
        Args:
            F_in (torch.Tensor): Input field, shape (H, W).
            z (float): Propagation distance.
            Returns: farfield (torch.Tensor): Output field after propagation, shape (H, W)"""
        if z == 0:
            field_nf = F_in
        else:
            field_nf = F_in * self.lens(z)
        return torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(field_nf), norm='ortho'))

class SLMphaseRetrievalCG(paddedSLM):
    """ Class for SLM phase retrieval using conjugate gradient method 2d.
    Args:
        target (torch.Tensor): Target field distribution, shape (H, W).
        max_iter (int): Maximum number of iterations for the CG algorithm.
        tol (float): Tolerance for convergence.
    """
    def __init__(self,A_in, A_t,pad_factor = 4,H=1024,W=1280):
        super().__init__(pad_factor=pad_factor,H=H,W=W)
        # input Amplitude
        if A_in is not None:
            self.A_in = self.pad(torch.tensor(A_in / FeedbackHologram._norm(A_in), dtype=torch.float32))
        else:
            self.A_in = self.build_input()
        
        # target Amplitude
        if A_t is not None:
            self.A_t = torch.tensor(A_t / FeedbackHologram._norm(A_t), dtype=torch.float32)
    def loss(self, phase):
        """ compute the mse loss """
        # forward propagation
        F_in = torch.exp(1j * self.pad(phase)) * self.A_in
        F_out = self.forward_propagate(F_in,z=0)
        # compute the loss
        F_out_amp = torch.abs(F_out)
        return torch.sum((F_out_amp**2 - self.A_t**2) ** 2)*1e12
    def solve(self,max_iter=10,x0 = None):
        """ solves the phase using conjugate gradient method. """
        phase_rand = torch.rand((self.H, self.W), dtype=torch.float32) * 2 * torch.pi
        if x0 is None:
            x0 = phase_rand
        result = minimize(self.loss, x0 = x0, method='cg',max_iter = max_iter,tol = 1e-20,disp = 1)
        out = torch.abs(self.forward_propagate(torch.exp(1j * self.pad(result.x)) * self.A_in)).detach().cpu().numpy()
        return result ,self.unpad(out)
    
class SLMphaseRetrievalCG3D(SLMphaseRetrievalCG):
    """ Class for SLM phase retrieval using conjugate gradient method 3d.
    Args:
        target (torch.Tensor): Target field distribution, shape (H, W).
        max_iter (int): Maximum number of iterations for the CG algorithm.
        tol (float): Tolerance for convergence.
    """
    def __init__(self,A_in, A_t,pad_factor = 4,H=1024,W=1280):
        super().__init__(A_in=A_in, A_t=A_t,pad_factor=pad_factor,H=H,W=W)
        self.z_vals = torch.linspace(-3e-5, 3e-5, steps=20)
        # self.z_vals = torch.linspace(-1e-3, 1e-3, steps=20)
    def build_target_from_phase(self, phase):
        """ Build the target tensor for the SLM from a phase tensor.
        Args:
            phase (torch.Tensor): Phase tensor, shape (H, W).
            Returns: target_tensor (torch.Tensor): Padded target tensor, shape (pad_H, pad_W)"""
        outputs = []
        for z in self.z_vals:
            output = torch.abs(self.forward_propagate(self.A_in * self.pad(torch.exp(1j * phase)), z=z))
            outputs.append(output)
        self.A_t = torch.stack(outputs, dim=0)
        return self.A_t
    def loss(self, phase):
        """ compute the mse loss """
        # forward propagation
        
        F_in = torch.exp(1j * self.pad(phase)) * self.A_in
        outputs = []
        for z in self.z_vals:
            output = self.forward_propagate(F_in,z=z)
            outputs.append(output)
        F_out = torch.stack(outputs, dim=0)
        # compute the loss
        F_out_amp = torch.abs(F_out)
        return torch.sum((F_out_amp**2 - self.A_t**2) ** 2)*1e12




