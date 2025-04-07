import matplotlib.pyplot as plt
from .config import calibrations_path,vendor_path
from slmsuite.holography.analysis.files import load_h5, save_h5, generate_path, latest_path
from slmsuite.holography.algorithms import SpotHologram
import numpy as np
def plot_cam(fs,exposure = 1,figsize = None,centre = False,phase = None,limits = None):
    "gets an image from camera and plots it "  
    if isinstance(phase, np.ndarray):
        fs.slm.write(phase, settle=True)
    elif phase == "zero":
        phase = np.zeros(fs.slm.shape)
        fs.slm.write(phase, settle=True)
    else:
        "do nothing"
    fs.cam.set_exposure(exposure*0.064e-3)
    img = fs.cam.get_image()
    if limits != None:
        img = img[limits]
    plt.figure(figsize=figsize)
    plt.imshow(img)
    if centre == True:
        plt.scatter(720,540,color = 'blue',s = 10)
    return img


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

def create_tweezers(fs):
    "creates optical tweezers using wgs-kim at the given spot locations"

    "create the target spot array"
    # spot_locations = np.array([[900+(i+1)*50 for i in range(8) for j in range(8)],
    #    [(j+1)*50 for i in range(8) for j in range(8)]])
    spot_locations = np.array([[900+(i+1)*100 for i in range(4) for j in range(4)],
       [(j+1)*100 for i in range(4) for j in range(4)]])
    plt.scatter(spot_locations[0,:], spot_locations[1,:])
    plt.xlim([0, fs.cam.shape[1]]); plt.ylim([fs.cam.shape[0], 0])
    plt.show()

    "Initialise and optimise the hologram"
    hologram = SpotHologram(shape=fs.slm.shape, spot_vectors=spot_locations, basis='ij', cameraslm=fs)
    hologram.optimize('WGS-Kim', feedback='experimental_spot', stat_groups=[ 'experimental_spot'], maxiter=30,fixed_phase=False)
    hologram.plot_stats()
    plot_cam(fs)
    return hologram

def bottle_beam_phase(fs):
    ro = 240
    return np.array([[np.pi if ((i-fs.slm.get_source_center()[0])**2 + (j-fs.slm.get_source_center()[1])**2) <= ro**2 else 0 for j in range(1280)] for i in range(1024)])


