from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from slmsuite.holography.toolbox.phase import zernike_sum,zernike,blaze
import torch
from src.config import base_path
from src.utils import save_cam,get_cam,calibrate,sp_calibrate
from src.slm_initialise import slm_initialise
import pickle
from skimage.feature import peak_local_max
from lmfit.models import Gaussian2dModel
import time
from src import global_store

# slm test cases

def is_slm_active():
    image = get_cam(fs=global_store.fs,phase = blaze(global_store.fs,vector = (-0.003,-0.003)))
    if image[330:360,490:520].sum() > 10000:
        image2 = get_cam(fs=global_store.fs,phase = blaze(global_store.fs,vector = (0.003,-0.003)))
        if image2[750:790,510:540].sum() > 10000:
            return True
    return False
def slm_restart( max_retries=5, delay=2):
    """
    Try restarting SLM if it is unresponsive.

    Args:
        fs: current SLM session/handle.
        max_retries (int): number of times to try restarting.
        delay (int): seconds to wait between retries.

    Returns:
        fs (updated SLM handle if successful), or None if failed.
    """
    if not is_slm_active():
        print("SLM stopped, attempting restart...")
        for attempt in range(1, max_retries + 1):
            global_store.fs = slm_initialise(force=True, verbose=True)
            calibrate(global_store.fs,vendor = False,refresh = False)
            sp_calibrate(global_store.fs,refresh=False)
            if is_slm_active():
                print(f"SLM restarted successfully on attempt {attempt}")
                return global_store.fs
            print(f"Attempt {attempt} failed, retrying in {delay}s...")
            time.sleep(delay)

        print("SLM could not be restarted after retries.")
        return None

    return global_store.fs




#Fitting functions
def parabola(x, x0, a, c):
    return c + a * np.square(x - x0)
def fit_metric(sweep,result):
    "fits a parabola to the given w and values and returns the fitted params"
    "sweep : the x values"
    "result: the y values"
    ddy = np.diff(result, n=2, axis=0)
    a0 = .5 * np.mean(ddy, axis=0) / np.square(np.mean(np.diff(sweep)))
    if a0>0:
        a0 = -a0
    guess = (sweep[np.argmax(result)], a0, np.argmax(result))
    try:
        popt, pcov = optimize.curve_fit(
            parabola,
            sweep,
            result,
            ftol=1e-5,
            p0=guess,
            bounds=(
                [-np.inf,  -np.inf, -np.inf],
                [np.inf, 0, np.inf]
            )
        )
        perr = np.sqrt(np.diag(pcov))   # Single sigma error, which can be multiplied later.
    except Exception as e:
        print(e)
        popt = guess
        perr = np.zeros_like(guess)
    # print("guess",guess)
    if np.any(popt[0] < np.min(sweep)) or np.any(popt[0] > np.max(sweep)):
        print("A Zernike aberration optimized out of the range of the measurement.")
    if perr[0] > 0.1:
        print("very high error in the fit, setting values to 0")
        popt = np.zeros_like(guess)
    
    return popt,perr
def fit_metric_udf(sweep,result):
    "udf for pandas which returns mean,error"
    popt,perr = fit_metric(sweep,result)
    mean = popt[0]
    limit = 0.2
    if popt[0] < -limit:
        mean = -limit
    if popt[0] > limit:
       mean = limit
    return pd.Series({"w_opt": mean, "w_stddev": perr[0]})
def plot_parabola(popt,sweep,result):
    # Your fitted parameters
    x0, a, c = popt

    # Define smooth x range for plotting fitted curve
    x_fit = np.linspace(sweep.min(), sweep.max(), 500)
    y_fit = parabola(x_fit, x0, a, c)

    plt.figure(figsize=(7,5))

    # Plot measured data (actual parabola samples)
    plt.scatter(sweep, result, label="Data (actual)", color="blue")

    # Plot fitted parabola
    plt.plot(x_fit, y_fit, label="Fitted parabola", color="red", linewidth=2)

    # If you know the *true parabola* function (say with known params):
    # y_true = parabola(x_fit, x0_true, a_true, c_true)
    # plt.plot(x_fit, y_true, '--', label="True parabola", color="green")

    plt.xlabel("w")
    plt.ylabel('metric')
    plt.title("Parabola Fit")
    plt.legend()
    plt.grid(True)
    plt.show()



"metrics functions for tweezers"



def refine_centers(image, expected_centers, patch_size=40, min_distance=1, threshold_abs=None):
    half = patch_size // 2
    refined_centers = []

    for (x_exp, y_exp) in expected_centers:
        x_exp, y_exp = int(round(x_exp)), int(round(y_exp))
        patch = image[y_exp - half:y_exp + half, x_exp - half:x_exp + half]

        # Detect all peaks in this patch
        peaks = peak_local_max(
            patch,
            min_distance=min_distance,
            threshold_abs=threshold_abs,
            num_peaks=1  # adjust if needed
        )

        if len(peaks) == 0:
            refined_centers.append((x_exp, y_exp))
            continue

        # Find the peak closest to the expected center (in patch coordinates)
        patch_center = np.array([half, half])
        distances = np.linalg.norm(peaks - patch_center, axis=1)
        closest_peak = peaks[np.argmin(distances)]

        # Convert local peak coords to global image coords
        y_actual = y_exp - half + closest_peak[0]
        x_actual = x_exp - half + closest_peak[1]

        refined_centers.append((x_actual, y_actual))

    return refined_centers


def get_tweezer_metrics(target, waist_multiplier=7):
    ny, nx = target.shape
    y, x = np.mgrid[:ny, :nx]
    x_flat = x.ravel()
    y_flat = y.ravel()
    z_flat = target.ravel()

    # Fit 2D Gaussian
    model = Gaussian2dModel()
    params = model.guess(z_flat, x=x_flat, y=y_flat)
    result = model.fit(z_flat, params, x=x_flat, y=y_flat, max_nfev=10000)

    # Best-fit parameters
    height = result.params["height"].value
    sigmax = result.params["sigmax"].value
    sigmay = result.params["sigmay"].value
    centerx = result.params["centerx"].value
    centery = result.params["centery"].value

    # Fit quality
    fitted = result.best_fit
    residuals = z_flat - fitted
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((z_flat - np.mean(z_flat))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # FWHM and ellipticity
    fwhm_x = 2 * np.sqrt(2 * np.log(2)) * sigmax
    fwhm_y = 2 * np.sqrt(2 * np.log(2)) * sigmay
    ellipticity = sigmax / sigmay if sigmay != 0 else np.inf

    # === Strehl-style metric ===
    # Use window size based on waist multiplier (e.g., 5x or 7x waist)
    sigma_avg = (sigmax + sigmay) / 2
    window_size = int(round(waist_multiplier * sigma_avg))

    # Make sure it's odd (centered box)
    if window_size % 2 == 0:
        window_size += 1

    half = window_size // 2
    cx, cy = int(round(centerx)), int(round(centery))

    # Crop patch around fitted center
    y_start = max(cy - half, 0)
    y_end   = min(cy + half + 1, ny)
    x_start = max(cx - half, 0)
    x_end   = min(cx + half + 1, nx)

    patch = target[y_start:y_end, x_start:x_end]
    if patch.size > 0 and patch.sum() > 0:
        strehl_prime = patch.max() / patch.sum()
    else:
        strehl_prime = 0.0  # fallback in case of zero patch

    return {
        "height": height,
        "sigmax": sigmax,
        "sigmay": sigmay,
        "centerx": centerx,
        "centery": centery,
        "fwhm_x": fwhm_x,
        "fwhm_y": fwhm_y,
        "ellipticity": ellipticity,
        "strehl_prime": strehl_prime,
        "rmse": rmse,
        "r_squared": r_squared,
        "window_px": window_size
    }




"functions for zernike calibrations"

def get_zernike_data(base_phase,zernike_indices,file_name = "zernike_data_temp"):
    "iterates the top 20 zernike polynomials over the given phase mask and saves the data to the file_name"
    data = {}
    # zernike_indices = [i for i in range(1,nz+1)]
    for zernike_index in zernike_indices:
        # slm_restart()
        zernike_term_phase = zernike(global_store.fs.slm, zernike_index,use_mask = False)
        data[zernike_index] = {
            "w": [],
            "images": []
        }
        for w in np.linspace(-0.5, 0.5, 20):  
            phase = base_phase + w * zernike_term_phase
            if np.isnan(phase).any() == True:
                print("na's exist")
            image = save_cam(global_store.fs,phase = phase,sleep = 4,exposure=80)
            data[zernike_index]["w"].append(w)
            data[zernike_index]["images"].append(image)
    save_path = base_path / "data" / "zernike" / "raw" / f"{file_name}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    return data

def get_zernike_metrics_df(loaded,spot_locations,file_name = "zernike_data_temp"):
    "Computes tweezer level metrics like strehl ratio etc into a dataframe using the get_tweezer_metrics and expected tweezer locations and the data from iterating zernike modes"
    results = []
    crop_pixels = 30

    for zernike_index, zernike_data in loaded.items():
        images = zernike_data["images"]  # list of images at different w

        for w_index, image in enumerate(images):
            refined_centres = refine_centers(image, spot_locations, 200)
            spot_index = 0
            for (expected, refined) in zip(spot_locations, refined_centres):
                cx, cy = refined
                crop_details = (
                    slice(cy - crop_pixels, cy + crop_pixels),
                    slice(cx - crop_pixels, cx + crop_pixels)
                )

                cropped_image = image[crop_details]
                metrics = get_tweezer_metrics(cropped_image)

                results.append({
                    "zernike_index": zernike_index,
                    "w_index": w_index,
                    "w":zernike_data['w'][w_index],
                    "spot_index":spot_index,
                    "expected_x": expected[0],
                    "expected_y": expected[1],
                    "refined_x": cx,
                    "refined_y": cy,
                    **metrics
                })
                spot_index+=1
    
    pdf = pd.DataFrame(results)
    load_path = base_path / "data" / "zernike" / "raw" / f"{file_name}_df.csv"
    pdf.to_csv(load_path)
    return pdf


def get_zernike_dict_from_metrics_df(pdf,spot_locations):
    "takes the metrics pdf and returns list of zernike dicts with mean best aberration for each"
    result = pdf.groupby(["zernike_index","spot_index"]).apply(lambda g : fit_metric_udf(g['w'].to_numpy(),g['strehl_prime'].to_numpy())).reset_index()
    result.loc[result["zernike_index"].isin([1,2]), "w_opt"] = 0.0
    return result.groupby("spot_index").apply(
        lambda g: dict(zip(g["zernike_index"], g["w_opt"]))
    ).reset_index(name="zernike_dict").sort_values(by = "spot_index" )['zernike_dict'].to_list()


def get_zernike_calibrations(phase,spot_locations,zernike_indices,file_name):
    "iterates the zenike polynomials takes the spot level zernike metric dataframe fits parabolas and returns the top zernike modes with aberration weights as a dict"
    data = get_zernike_data(base_phase=phase,zernike_indices=zernike_indices,file_name = file_name)
    zernike_metrics_df = get_zernike_metrics_df(loaded=data,spot_locations = spot_locations,file_name=file_name)
    zernike_calibrations = get_zernike_dict_from_metrics_df(pdf = zernike_metrics_df,spot_locations=spot_locations)  
    return zernike_calibrations




#zernike kernel holography 

class CompressedHolo:
    def __init__(self,zernike_calibrations,fs):
        """Pass a zernike calibration a list of dicts with keys zernike_index and weights"""
        self.zernike_calibrations = zernike_calibrations
        self.N = len(zernike_calibrations)
        self.H = fs.slm.shape[0]
        self.W = fs.slm.shape[1]
        self.kernel = self.compute_kernel(zernike_calibrations,fs)
        self.A_in = torch.ones(self.H,self.W)/np.sqrt(self.H*self.W)
        # self.A_in = torch.tensor(fs.slm.source['amplitude']/np.sqrt((fs.slm.source['amplitude']**2).sum()))
    def zt(self,F_in):
        """zernike transform of the nearfield to farfield 
        Takes input the nearfield and outputs a array of size N
        NxHxW,HxW to Nx1
        """
        return torch.einsum('nhw,hw->n', self.kernel, F_in)
    def izt(self,F_out):
        """inverse zernike transform of the farfield to nearfield 
        Takes input the farfield spot array complex amplitudes and outputs a array of size HxW
        NxHxW,Nx1 to HxW
        """
        return torch.einsum('nhw,n->hw', torch.conj(self.kernel), F_out)
    def compute_kernel(self,zernike_calibrations,fs):
        """Compute the zernike kernel"""
        kernels = []
        for zernike_calibration in zernike_calibrations:
            kernels.append(zernike_sum(fs,tuple(zernike_calibration.keys()),tuple(zernike_calibration.values()),use_mask = False))
        return torch.exp(1j*torch.from_numpy(np.stack(kernels)).float())
    def optimize_gs(self,initial_phase,iters = 10):
        """Optimize the Gerchberg-Saxton algorithm using the zernike transform"""
        if initial_phase == None:
            initial_phase = torch.zeros(self.H,self.W)
        else :
            initial_phase = torch.tensor(initial_phase)
        phi_in = initial_phase
        for iter in range(iters):
            F_in = self.A_in*torch.exp(1j*phi_in)
            phi_out = torch.angle(self.zt(F_in))
            F_out = torch.exp(1j*phi_out)/np.sqrt(self.N)
            phi_in = torch.angle(self.izt(F_out))
        return phi_in.numpy()
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
    
class CompressedHolo_bb_g(CompressedHolo):
    def __init__(self,zernike_calibrations,fs):
        """Pass a zernike calibration a list of dicts with keys zernike_index and weights"""
        super().__init__(zernike_calibrations,fs)
    def izt_new(self,F_out,bb_phase,n):
        """inverse zernike transform of the farfield to nearfield 
        Takes input the farfield spot array complex amplitudes and outputs a array of size HxW
        NxHxW,Nx1 to HxW
        """
        out = torch.einsum('n,nhw->hw', F_out[:n], torch.conj(self.kernel[:n])) + 1*torch.einsum('n,nhw->hw', F_out[n:], torch.conj(self.kernel[n:]))*torch.exp(1j*torch.tensor(bb_phase))
        return out
    def optimize_gs(self,initial_phase,bb_phase,n,iters = 20):
        """Optimize the Gerchberg-Saxton algorithm using the zernike transform"""
        if initial_phase == None:
            initial_phase = torch.zeros(self.H,self.W)
        else :
            initial_phase = torch.tensor(initial_phase)
        phi_in = initial_phase
        for iter in range(iters):
            F_in = self.A_in*torch.exp(1j*phi_in)
            phi_out = torch.angle(self.zt(F_in))
            F_out = torch.exp(1j*phi_out)/np.sqrt(self.N)
            phi_in = torch.angle(self.izt(F_out))
        phi_in = torch.angle(self.izt_new(F_out,bb_phase,n))
        return phi_in.numpy()

def get_cholo_phase(zernike_calibrations,fs):
    "get the phase mask using gerchberg saxton algorithm in zernike space"
    cholo = CompressedHolo(zernike_calibrations = zernike_calibrations,fs=fs)
    phase = cholo.optimize_gs(initial_phase=None,iters = 100)
    return phase

def get_cholo_bb_g_phase(zernike_calibrations,fs,bb_phase,n):
    cholo = CompressedHolo_bb_g(zernike_calibrations,fs=fs)
    phase = cholo.optimize_gs(initial_phase=None,n=n,bb_phase = bb_phase,iters = 100)
    return phase



zernike_names = {
    0: "Piston",
    1: "Tilt X",
    2: "Tilt Y",
    3: "Defocus",
    4: "Astigmatism 45°",
    5: "Astigmatism 0°",
    6: "Coma X",
    7: "Coma Y",
    8: "Trefoil X",
    9: "Trefoil Y",
    10: "Primary Spherical",
    11: "Secondary Astigmatism 0°",
    12: "Secondary Astigmatism 45°",
    13: "Secondary Coma X",
    14: "Secondary Coma Y",
    15: "Quadrafoil 0°",
    16: "Quadrafoil 45°",
    17: "Secondary Spherical",
    18: "Tertiary Astigmatism 0°",
    19: "Tertiary Astigmatism 45°",
    20: "Tertiary Coma X",
    21: "Tertiary Coma Y",
    22: "Tertiary Trefoil X",
    23: "Tertiary Trefoil Y",
    24: "Tertiary Spherical"
}