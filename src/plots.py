from scipy.optimize import curve_fit
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize, ndimage
def g(x,y,A,w,xc,yc):
    "gaussian function"
    r = ((x-xc)**2 + (y-yc)**2)**0.5
    return (A)*(np.e**(-(r**2)/(w**2)))
def fg(xdata,A,w,xc,yc,offset):
    "gaussian function for scipy curve fit"
    x = xdata[0]
    y = xdata[1]
    return g(x,y,A,w,xc,yc) + offset
def bb(x,y,A1,A2,w1,w2,xc,yc):
    "bottle beam function"
    I = (g(x,y,A1,w1,xc,yc) - g(x,y,A2,w2,xc,yc))**2
    return I
def fbb(xdata,A1,A2,w1,w2,xc,yc,offset):
    "bottle function for scipy curve fit"
    x = xdata[0]
    y = xdata[1]
    return bb(x,y,A1,A2,w1,w2,xc,yc) + offset


def get_fit_params(target,trap = "none",plot = False,verbose = True):
    if trap == "g":
        f_fit = fg
        initial_guess = (target.max() - target.min(),50,target.shape[0]/2,*np.unravel_index(np.argmax(target),target.shape))
    elif trap== "bb":
        f_fit = fbb
        initial_guess = ((target.max() - target.min())/2,(target.max() - target.min())/2,20,10,target.shape[0]/2,target.shape[1]/2,target.min())
    else:
        print("provide valid trap type")
        return
    "function which fits an optical tweezer and plots it"
    target.shape[0],target.shape[1]
    xdata = np.array([[i,j] for i in range(target.shape[0]) for j in range(target.shape[1])]).T

    
    popt, pcov = curve_fit(f_fit,xdata,target[*xdata],p0=initial_guess,maxfev=10000)
    # popt = initial_guess
    if verbose == True:
        print(f"optimised params are {popt}")

    pred = f_fit(xdata, *popt).reshape(target.shape[0],target.shape[1])
    if plot == True:
        "matplotlib 2d plot"
        plt.figure()
        plt.imshow(pred)
    return popt

def plot_fit(target,trap = "none"):
    if trap == "g":
        f_fit = fg
        initial_guess = (target.max() - target.min(),5,target.shape[0]/2,target.shape[1]/2,target.min())
    elif trap== "bb":
        f_fit = fbb
        initial_guess = ((target.max() - target.min())/2,(target.max() - target.min())/2,20,10,target.shape[0]/2,target.shape[1]/2,target.min())
    else:
        print("provide valid trap type")
        return
    "function which fits an optical tweezer and plots it"
    target.shape[0],target.shape[1]
    xdata = np.array([[i,j] for i in range(target.shape[0]) for j in range(target.shape[1])]).T

    
    popt, pcov = curve_fit(f_fit,xdata,target[*xdata],p0=initial_guess,maxfev=10000)
    # popt = initial_guess
    print(f"optimised params are {popt}")

    "fitted image"
    pred = f_fit(xdata, *popt).reshape(target.shape[0],target.shape[1])

    "matplotlib 2d plot"
    plt.imshow(pred)
    
    "plotly 3d plot"
    "original bb as scatter plot"
    scatter_plot = go.Scatter3d(
        x=xdata[0],
        y=xdata[1],
        z=target[*xdata],
        mode='markers',
        marker=dict(size=1, color='blue'),
        name='Target bottle beam'
    )
    "fitted bb as surface plot"
    surface_plot = go.Surface(
    x=xdata[0].reshape(target.shape[0],target.shape[1]),
    y=xdata[1].reshape(target.shape[0],target.shape[1]),
    z=pred,
    colorscale='YlOrRd',
    opacity=0.7,
    name='Predicted Surface'
    )

    "Combine plots"
    fig = go.Figure(data=[scatter_plot, surface_plot])
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        title='3D Curve Fit Plot in Plotly'
    )

    fig.show()



# Bottle beam line scans 
#usage : - 

        # fitres = fit_bottle_beam(bottle_beam_image, patch_size=80)
        # print("Params:", fitres["params"])
        # print("Uncertainties:", fitres["param_err"])
        # print("RMSE:", fitres["rmse"], "R^2:", fitres["r2"])

        # plot_fit_overview(patch, fitres)
        # plot_line_scans(fitres)



def bb(x, y, A, w, xc, yc, B):
    r = np.sqrt((x - xc)**2 + (y - yc)**2)
    return B + A * (r**4) * np.exp(-(r**2)/(w**2))

def _initial_guess(roi, xg, yg, approx_center=None):
    # Background ~ median; amplitude ~ (max - median)
    B0 = float(np.median(roi))
    A0 = float(np.max(roi) - B0)
    A0 = max(A0, 1e-6)

    # Center: use provided approx if any; else intensity COM (fine for a ring)
    if approx_center is not None:
        xc0, yc0 = approx_center
    else:
        # Center-of-mass in image coordinates
        cy, cx = ndimage.center_of_mass(roi - B0 + 1e-6)
        yc0 = float(yg.min() + cy)
        xc0 = float(xg.min() + cx)

    # Width: rough radial scale from second moment
    r2 = (xg - xc0)**2 + (yg - yc0)**2
    w0 = np.sqrt(max(np.average(r2, weights=np.clip(roi - B0, 0, None)), 1.0))

    return (A0, w0, xc0, yc0, B0)

def fit_bottle_beam(image, approx_center=None, patch_size=None):
    """
    image: 2D numpy array
    approx_center: (xc, yc) in pixel coords (x=cols, y=rows). If given with patch_size,
                   we fit a crop around it for speed/robustness.
    patch_size: half-size of ROI box (int). If None, fit the whole image.
    """
    H, W = image.shape

    # Choose ROI
    if approx_center is not None and patch_size is not None:
        xc_a, yc_a = approx_center
        xc_a = int(round(xc_a))
        yc_a = int(round(yc_a))
        x0 = max(0, xc_a - patch_size)
        x1 = min(W, xc_a + patch_size)
        y0 = max(0, yc_a - patch_size)
        y1 = min(H, yc_a + patch_size)
        roi = image[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]  # absolute coords so fit returns global xc,yc
    else:
        roi = image
        yy, xx = np.mgrid[0:H, 0:W]

    # Initial guess
    A0, w0, xc0, yc0, B0 = _initial_guess(roi, xx, yy, approx_center)

    # Flatten for curve_fit
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    z_flat = roi.ravel().astype(np.float64)

    # Bounds: A >= 0, w > 0, xc,yc within image, B free (>= min-abs-margin)
    eps = 1e-6
    lower = [0.0, eps, 0.0, 0.0, -np.inf]
    upper = [np.inf, np.inf, W-1.0, H-1.0, np.inf]

    p0 = [A0, max(w0, 2.0), np.clip(xc0, 0, W-1), np.clip(yc0, 0, H-1), B0]

    popt, pcov = optimize.curve_fit(
        lambda XY, A, w, xc, yc, B: bb(XY[0], XY[1], A, w, xc, yc, B),
        (x_flat, y_flat), z_flat, p0=p0, bounds=(lower, upper), maxfev=20000
    )
    perr = np.sqrt(np.diag(pcov)) if pcov.size else np.full_like(popt, np.nan, dtype=float)

    # Predictions on ROI for metrics
    z_pred = bb(x_flat, y_flat, *popt)
    resid = z_flat - z_pred
    rmse = float(np.sqrt(np.mean(resid**2)))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((z_flat - np.mean(z_flat))**2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    # Line scans through fitted center on FULL image
    xc, yc = popt[2], popt[3]
    x_idx = int(np.clip(round(xc), 0, W-1))
    y_idx = int(np.clip(round(yc), 0, H-1))

    x_axis = np.arange(W)
    y_axis = np.arange(H)

    scan_h_data = image[y_idx, :]                   # horizontal (y fixed = yc)
    scan_v_data = image[:, x_idx]                   # vertical (x fixed = xc)

    scan_h_model = bb(x_axis, np.full_like(x_axis, yc, dtype=float), *popt)
    scan_v_model = bb(np.full_like(y_axis, xc, dtype=float), y_axis, *popt)

    out = {
        "params": {"A": popt[0], "w": popt[1], "xc": popt[2], "yc": popt[3], "B": popt[4]},
        "param_err": {"A": perr[0], "w": perr[1], "xc": perr[2], "yc": perr[3], "B": perr[4]},
        "rmse": rmse,
        "r2": r2,
        "center_indices": {"x_idx": x_idx, "y_idx": y_idx},
        "line_scans": {
            "x": x_axis, "y": y_axis,
            "horizontal_data": scan_h_data, "horizontal_model": scan_h_model,
            "vertical_data": scan_v_data,   "vertical_model":   scan_v_model,
        },
        "roi_bbox": ((yy.min(), yy.max()), (xx.min(), xx.max()))  # (y0,y1), (x0,x1)
    }
    return out

# ---------- Example plotting helpers ----------

def plot_fit_overview(image, fitres):
    A, w, xc, yc, B = (fitres["params"][k] for k in ("A","w","xc","yc","B"))
    H, W = image.shape
    yy, xx = np.mgrid[0:H, 0:W]
    model_full = bb(xx, yy, A, w, xc, yc, B)

    fig, axs = plt.subplots(1, 3, figsize=(14,4))
    im0 = axs[0].imshow(image, origin="upper")
    axs[0].set_title("Image")
    axs[0].plot([xc],[yc],'r+', ms=12)
    plt.colorbar(im0, ax=axs[0], fraction=0.046)

    im1 = axs[1].imshow(model_full, origin="upper")
    axs[1].set_title("Fitted model")
    axs[1].plot([xc],[yc],'r+', ms=12)
    plt.colorbar(im1, ax=axs[1], fraction=0.046)

    im2 = axs[2].imshow(image - model_full, origin="upper")
    axs[2].set_title(f"Residual (RMSE={fitres['rmse']:.3g}, R²={fitres['r2']:.3f})")
    plt.colorbar(im2, ax=axs[2], fraction=0.046)

    for ax in axs: ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_line_scans(fitres):
    L = fitres["line_scans"]
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    axes[0].plot(L["x"], L["horizontal_data"], '.', label="data")
    axes[0].plot(L["x"], L["horizontal_model"], '-', label="model")
    axes[0].set_title("Horizontal scan (y = yc)")
    axes[0].set_xlabel("x (px)"); axes[0].set_ylabel("Intensity"); axes[0].legend()

    axes[1].plot(L["y"], L["vertical_data"], '.', label="data")
    axes[1].plot(L["y"], L["vertical_model"], '-', label="model")
    axes[1].set_title("Vertical scan (x = xc)")
    axes[1].set_xlabel("y (px)"); axes[1].set_ylabel("Intensity"); axes[1].legend()

    plt.tight_layout(); plt.show()
