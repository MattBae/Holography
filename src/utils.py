import matplotlib.pyplot as plt
from .config import calibrations_path,vendor_path,figures_path
from slmsuite.holography.analysis.files import load_h5, save_h5, generate_path, latest_path
from slmsuite.holography.algorithms import SpotHologram
import numpy as np
from PIL import Image
import time

def get_cam(fs,exposure = 1,figsize = None,centre = False,phase = None,limits = None,save=False,sleep = 0):
    "gets an image from camera "  
    if isinstance(phase, np.ndarray):
        print("writing phase")
        fs.slm.write(phase, settle=True)
    elif phase == "zero":
        phase = np.zeros(fs.slm.shape)
        fs.slm.write(phase, settle=True)
    else:
        "do nothing"
    fs.cam.set_exposure(exposure*0.064e-3)
    time.sleep(sleep)
    img = fs.cam.get_image()
    if limits != None:
        img = img[limits]
    return img


def plot_cam(fs,exposure = 1,figsize = None,centre = False,phase = None,limits = None, save=False,plot = True,sleep = 0):
    "gets an image from camera and plots it" 
    img = get_cam(fs,exposure,figsize,centre,phase,limits,save)
    if plot == True:
        plt.figure(figsize=figsize)
        plt.imshow(img)
    if centre == True:
        plt.scatter(720,540,color = 'blue',s = 10)
    return img


def save_cam(fs,exposure = 1,figsize = None,centre = False,phase = None,limits = None,file_name="temp"):
    "gets an image from camera and saves it" 
    img = get_cam(fs,exposure,figsize,centre,phase,limits)
    np.save(f"{figures_path/'raw'/file_name}.npy", img)



def calibrate(fs,vendor = True,refresh = True):
    "does defalult calibrations of vendor, fourier and settle time"
    
    "vendor"
    if vendor == True:
        fs.slm.load_vendor_phase_correction(
        file_path=vendor_path
        )
    
    if refresh == True:
        "fourier"
        fs.cam.set_exposure(30*0.064e-3)
        fc = fs.fourier_calibrate(
            array_shape=(12,16),
            array_pitch=20,
            plot=True,
        )
        "settle time"
        sc = fs.settle_calibrate()

        fs.save_calibration(calibration_type = "fourier", path=calibrations_path)
        fs.save_calibration(calibration_type = "settle", path=calibrations_path)
        return sc,fc
    else:
        for c in ["fourier","settle"]:
            name = fs.name_calibration(c)
            file_path = latest_path(calibrations_path, name, extension="h5")
            fs.load_calibration(calibration_type = c, file_path=file_path)
    
def sp_calibrate(fs,refresh = True):
    "does wavefront super pixel calibration to remove aberrations"
    if refresh == True:
        fs.cam.set_exposure(10*0.064e-3)
        movie = fs.wavefront_calibrate(
            calibration_points=np.array([[1180], [230]]),
            field_point=(0.25, 0),
            field_point_units="freq",
            superpixel_size=128,
            plot=0
        )
        fs.save_calibration(calibration_type = "wavefront_superpixel",path=calibrations_path)
    else:
        name = fs.name_calibration("wavefront_superpixel")
        file_path = latest_path(calibrations_path, name, extension="h5")
        fs.load_calibration(calibration_type = "wavefront_superpixel", file_path=file_path)
    fs.wavefront_calibration_superpixel_process(r2_threshold=.3, plot=True,smooth=True)
    fs.slm.fit_source_amplitude(force=True)

def create_tweezers(fs,spot_locations = None,maxiter = 30):
    "creates optical tweezers using wgs-kim at the given spot locations"

    "create the target spot array"
    # spot_locations = np.array([[900+(i+1)*50 for i in range(8) for j in range(8)],
    #    [(j+1)*50 for i in range(8) for j in range(8)]])
    if spot_locations is None:
        spot_locations = np.array([[900+(i+1)*100 for i in range(4) for j in range(4)],
        [(j+1)*100 for i in range(4) for j in range(4)]])
    plt.scatter(spot_locations[0,:], spot_locations[1,:])
    plt.xlim([0, fs.cam.shape[1]]); plt.ylim([fs.cam.shape[0], 0])
    plt.show()

    "Initialise and optimise the hologram"
    hologram = SpotHologram(shape=fs.slm.shape, spot_vectors=spot_locations, basis='ij', cameraslm=fs)
    hologram.optimize('WGS-Kim', feedback='computational_spot', stat_groups=[ 'computational_spot'], maxiter=maxiter)
    hologram.optimize('WGS-Kim', feedback='experimental_spot', stat_groups=['computational_spot','experimental_spot'], maxiter=maxiter,fixed_phase=True)
    hologram.plot_stats()
    plot_cam(fs)
    return hologram

def bottle_beam_phase(fs):
    ro = 240
    return np.array([[np.pi if ((i-fs.slm.get_source_center()[0])**2 + (j-fs.slm.get_source_center()[1])**2) <= ro**2 else 0 for j in range(1280)] for i in range(1024)])

def random_masking(phi1,phi2):
    "Superposes two holograms by allocating slm pixels randomly to each of them"
    mask = np.random.choice([True,False],size = phi1.shape,p = [0.42,0.58])
    return np.where(mask,phi1,phi2)

def lens_phase(fs,shift):
    xc,yc = fs.slm.get_source_center()
    return np.array([((i-xc)**2 + (j-yc)**2)*-1*shift for i in range(fs.slm.shape[0]) for j in range(fs.slm.shape[1])]).reshape(fs.slm.shape)

def get_3d_images(fs,phase,N = 100):
    imgs = []
    for i in np.linspace(-25,25,N):
        imgs.append(get_cam(fs,phase = phase +lens_phase(fs,-0.000001*i)))
    return imgs



def plot_axial(plane='XZ', index=50, images=None):
    """
    Plots either XZ or YZ plane from a 3D image stack.
    
    Parameters:
    - plane: str, either 'XZ' or 'YZ', determines the plane to plot
    - index: int, the fixed index for Y (for XZ) or X (for YZ) plane
    - image_stack: 3D numpy array, the image stack (Z, Y, X)
    """
    image_stack = np.stack(images,axis=0)
    # Define slicing logic for each plane type
    slicing_dict = {
        'XZ': (slice(None), index, slice(None)),  # XZ -> All Z, fixed Y, All X
        'YZ': (slice(None), slice(None), index)   # YZ -> All Z, All Y, fixed X
    }
    
    if plane not in slicing_dict:
        print("Invalid plane. Please choose 'XZ' or 'YZ'.")
        return
    # Apply the correct slicing
    slice_z, slice_y, slice_x = slicing_dict[plane]
    plane_data = image_stack[slice_z, slice_y, slice_x]
    print(plane_data.shape)

    # Create labels and title based on plane type
    if plane == 'XZ':
        xlabel, ylabel = "X", "Z"
        title = f"XZ Plot (Y = {index})"
        extent = [0, plane_data.shape[1], 0, image_stack.shape[0]]
    elif plane == 'YZ':
        xlabel, ylabel = "Y", "Z"
        title = f"YZ Plot (X = {index})"
        extent = [0, plane_data.shape[0], 0, image_stack.shape[0]]
        plane_data = plane_data

    # Plot the selected plane
    plt.figure(figsize=(10, 5))
    plt.imshow(plane_data, aspect='auto', cmap='viridis')
    plt.colorbar(label="Intensity")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
