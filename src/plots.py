from scipy.optimize import curve_fit
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import numpy as np
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
    I = g(x,y,A1,w1,xc,yc) - g(x,y,A2,w2,xc,yc)
    return I
def fbb(xdata,A1,A2,w1,w2,xc,yc,offset):
    "bottle function for scipy curve fit"
    x = xdata[0]
    y = xdata[1]
    return bb(x,y,A1,A2,w1,w2,xc,yc) + offset


def get_fit_params(target,trap = "none"):
    if trap == "g":
        f_fit = fg
        initial_guess = (target.max() - target.min(),10,target.shape[0]/2,target.shape[1]/2,target.min())
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

    pred = f_fit(xdata, *popt).reshape(target.shape[0],target.shape[1])
    "matplotlib 2d plot"
    plt.figure()
    plt.imshow(pred)
    return popt

def plot_fit(target,trap = "none"):
    if trap == "g":
        f_fit = fg
        initial_guess = (target.max() - target.min(),10,target.shape[0]/2,target.shape[1]/2,target.min())
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