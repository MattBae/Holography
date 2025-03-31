from time import sleep
import numpy as np
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.holography.algorithms import Hologram, SpotHologram, FeedbackHologram
from slmsuite.hardware.slms.screenmirrored import ScreenMirrored
from slmsuite.hardware.cameras.thorlabs import ThorCam

slm = None
cam = None
def slm_initialise(force = False,verbose = False):
    "initialises the slm and camera and returns as fourierslm(slm+camera) object with a zero phase mask written on the slm"
    global slm
    global cam
    if force == True:
        try:
            slm.close()
        except Exception as e:
            print(f"An error occurred with slm: {e}")

    if force == True:
        try:
            cam.close()
        except Exception as e:
            print(f"An error occurred with cam: {e}")
    ScreenMirrored.info(verbose=verbose);
    slm = ScreenMirrored(display_number=0, wav_um=1.064, settle_time_s=3)

    ThorCam.info(verbose=verbose)
    cam = ThorCam(serial="12975")
    cam.setup("free")

    fs = FourierSLM(cam, slm)

    # phase = np.zeros(slm.shape)
    # fs.slm.write(phase, settle=True)
    return fs

