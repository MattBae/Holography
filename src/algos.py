import torch
import numpy as np
from slmsuite.holography.algorithms import FeedbackHologram
from slmsuite.holography import toolbox
from slmsuite.holography.toolbox.phase import zernike
from lmfit.models import Gaussian2dModel
from lmfit.lineshapes import gaussian2d
from src.utils import *

def create_target_spots(spot_vectors,fs,spot_weights=None):
    "Creates the target Amplitude image in fourier domain (superposition of deltas ) using the spot locations at camera domain"
    target = np.zeros(fs.slm.shape)
    target_shape = target.shape
    if spot_weights is None:
        spot_weights = np.ones(spot_vectors.shape[0])
    for i in range(spot_vectors.shape[0]):
        spot_location = spot_vectors[i]
        xc, yc = toolbox.convert_vector(fs.ijcam_to_kxyslm(spot_location),from_units='kxy',to_units='knm',hardware = fs)
        target[int(yc),int(xc)] = spot_weights[i]
    return target/FeedbackHologram._norm(target)

def gs(A_target,fs,n_iters = 100):
    A_t = A_target
    phi_nf = torch.zeros(fs.slm.shape)
    A_input = torch.tensor(np.ones(fs.slm.shape)/FeedbackHologram._norm(np.ones(fs.slm.shape)),dtype=torch.float32)
    F_ff = None
    for iter in range(n_iters):
        F_ff = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(A_input*torch.exp(1j * phi_nf)),norm ='ortho'))
        A_ff = torch.abs(F_ff)
        phi_ff = torch.angle(F_ff)
        F_nf = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(A_t * torch.exp(1j * phi_ff)),norm ='ortho'))
        A_nf = torch.abs(F_nf)
        phi_nf = torch.angle(F_nf)
        print(torch.nn.functional.mse_loss(A_ff*A_ff, A_t*A_t).item()*fs.slm.shape[0]*fs.slm.shape[1])
    return phi_nf,phi_ff

def gs_spots(spot_vectors,fs,n_iters = 100):
    "for the given spot locations in camera domain runs gerchberg saxton algorithm to create phase mask in simulation"
    "return the nearfield phase and farfield phase which will be kept fixed for other feed back corrections"
    A_t = torch.tensor(create_target_spots(spot_vectors,fs))
    return gs(A_t,fs,n_iters = 100)



def get_phi_nf(fs,spot_vectors,phi_ff,spot_weights):
    "creates a nearfield phase given spot params spot_vectors and feedback from spot_weights :- positions shifts : 3,4 intensities weightage: 0 shape apodisation : 1,2 and farfield phases computed from gs algo"
    "a method based on https://arxiv.org/pdf/2407.20699"
    thetas = []
    F_nf = torch.zeros(phi_ff.shape)
    for i in range(spot_vectors.shape[0]):
        spot_vector = spot_vectors[i]
        spot_amp = spot_weights[i][0]
        spot_x = spot_weights[i][1]
        spot_y = spot_weights[i][2]
        shift_x = spot_weights[i][3]
        shift_y = spot_weights[i][4]
        zernikex = torch.tensor(zernike(fs.slm, 2))
        zernikey = torch.tensor(zernike(fs.slm, 1))
        yc, xc = toolbox.convert_vector(fs.ijcam_to_kxyslm(spot_vector),from_units='kxy',to_units='knm',hardware = fs)
        theta = phi_ff[int(xc),int(yc)]
        x,y = torch.meshgrid(torch.arange(phi_ff.shape[0]), torch.arange(phi_ff.shape[1]), indexing='ij')
        F_nf = F_nf + spot_amp*(torch.exp((-(np.sign(spot_y))*(spot_y**2)*((x-phi_ff.shape[0]/2)**2)-(np.sign(spot_x))*(spot_x**2)*((y-phi_ff.shape[1]/2)**2) )/1))*torch.exp(1j*(zernikex*shift_x + zernikey*shift_y))*(torch.exp((2j*np.pi*((int(xc[0]-phi_ff.shape[0]/2)*(x - phi_ff.shape[0]/2)/phi_ff.shape[0]) + (int(yc[0]-phi_ff.shape[1]/2)*(y - phi_ff.shape[1]/2)/phi_ff.shape[1])))+ 1j*theta))
    return torch.angle(F_nf/np.sqrt(phi_ff.shape[0]*phi_ff.shape[1]))


def get_fit_params_lm(target):
    "fits a gaussian and returns the fitted params"
    xdata = np.array([[i,j] for i in range(target.shape[0]) for j in range(target.shape[1])]).T
    model = Gaussian2dModel()
    params = model.guess(target.ravel(),xdata[0].ravel(),xdata[1].ravel())
    result = model.fit(target.ravel(),params,x = xdata[0].ravel(),y = xdata[1].ravel(),max_nfev = 10000)
    return [result.params['height'].value,result.params['sigmax'].value,result.params['sigmay'].value,result.params['centerx'].value,result.params['centery'].value]


def update_weights(fs,phase,spot_vectors,spot_weights,only_amp = False):
    "updates the spot weights :- positions shifts : 3,4 intensities weightage: 0 shape apodisation : 1,2 based on the camera feedback and gaussian fits"
    out = plot_cam(fs=fs,phase = phase,plot = False,sleep=3)
    params = []
    for spot_location in spot_vectors:
        params.append(get_fit_params_lm(out[spot_location[1]-50:spot_location[1]+50,spot_location[0]-50:spot_location[0]+50]))
    spot_w = np.array(params)
    spot_mean = np.mean(spot_w,axis=0)
    spot_std = np.std(spot_w,axis=0)
    print(f"variation is {np.std(spot_w,axis = 0)/np.mean(spot_w,axis = 0) * 100}")
    spot_weights[:,0] = spot_weights[:,0]*np.power(spot_mean[0]/spot_w.T[0],0.2)
    if only_amp == False:
        spot_weights[:,1] = spot_weights[:,1] + (- spot_w[:,1] + spot_mean[1])/1000
        spot_weights[:,2] = spot_weights[:,2] + (- spot_w[:,2] + spot_mean[2])/1000
        spot_weights[:,3] = spot_weights[:,3] + (- spot_w[:,3] + spot_mean[3])/3
        spot_weights[:,4] = spot_weights[:,4] + (- spot_w[:,4] + spot_mean[4])/3
    return spot_weights


def ultraprecise_spot_correction(spot_vectors,fs,phi_ff_gs,n_iters = 100):
    "corrects the spots using the method as in https://arxiv.org/pdf/2407.20699"
    spot_weights = np.zeros((spot_vectors.shape[0],5))
    spot_weights[:,0] = 1
    phis = []
    for iter in range(n_iters):
        phi_nf_new = get_phi_nf(fs,spot_vectors,phi_ff_gs,spot_weights)
        spot_weights = update_weights(fs,phi_nf_new.numpy(),spot_vectors,spot_weights,only_amp = False)
        phis.append(phi_nf_new)
    return phis

def ultraprecise_spots(spot_vectors,fs,n_iters_gs = 100,n_iters_corr = 100):
    "creates the spots using gs and corrects them using the method as in https://arxiv.org/pdf/2407.20699"
    phi_nf_gs,phi_ff_gs = gs(spot_vectors,fs,n_iters = n_iters_gs)
    return ultraprecise_spot_correction(spot_vectors,fs,phi_ff_gs,n_iters = n_iters_corr)