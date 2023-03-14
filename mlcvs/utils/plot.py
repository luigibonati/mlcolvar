##########################################################################
## FESSA COLOR PALETTE
##########################################################################
#  https://github.com/luigibonati/fessa-color-palette/blob/master/fessa.py

from matplotlib.colors import LinearSegmentedColormap, ColorConverter
from matplotlib.cm import register_cmap

__all__ = ["paletteFessa"]

paletteFessa = [
    '#1F3B73', # dark-blue
    '#2F9294', # green-blue
    '#50B28D', # green
    '#A7D655', # pisello
    '#FFE03E', # yellow
    '#FFA955', # orange
    '#D6573B', # red
]

cm_fessa = LinearSegmentedColormap.from_list('fessa', paletteFessa)
register_cmap(cmap=cm_fessa)
register_cmap(cmap=cm_fessa.reversed())

for i in range(len(paletteFessa)):
    ColorConverter.colors[f'fessa{i}'] = paletteFessa[i]

### To set it as default
# import fessa
# plt.set_cmap('fessa')
### or the reversed one
# plt.set_cmap('fessa_r')
### For contour plots
# plt.contourf(X, Y, Z, cmap='fessa')
### For standard plots
# plt.plot(x, y, color='fessa0')

##########################################################################
## HELPER FUNCTIONS FOR 2D SYSTEMS
##########################################################################

import matplotlib.pyplot as plt
import numpy as np
import torch 

def muller_brown_potential(x,y):
    """Muller-Brown analytical potential"""
    prefactor = 0.15
    A=(-200,-100,-170,15)
    a=(-1,-1,-6.5,0.7)
    b=(0,0,11,0.6)
    c=(-10,-10,-6.5,0.7)
    x0=(1,0,-0.5,-1)
    y0=(0,0.5,1.5,1)
    offset = -146.7

    v = -prefactor*offset
    for i in range(4):
        v += prefactor * A[i]*np.exp( a[i]*(x-x0[i])**2 + b[i]*(x-x0[i])*(y-y0[i]) + c[i]*(y-y0[i])**2 )
    return v

def plot_isolines_2D(function, component=None, 
                     limits=((-1.8,1.2),(-0.4,2.1)),num_points=(100,100),
                     mode='contourf',levels=12,vmax=None,cmap=None,colorbar=None,
                     ax=None):
    """Plot isolines of a function/model in a 2D space."""

    # Define grid where to evaluate function
    if type(num_points) == int:
        num_points = (num_points,num_points)
    xx = np.linspace(limits[0][0],limits[0][1],num_points[0])
    yy = np.linspace(limits[1][0],limits[1][1],num_points[1])
    xv, yv = np.meshgrid(xx, yy)

    # if torch module 
    if isinstance(function,torch.nn.Module):
        z = np.zeros_like(xv)
        for i in range(num_points[0]):
            for j in range(num_points[1]):
                xy = torch.Tensor([xv[i,j], yv[i,j]])
                with torch.no_grad():
                    train_mode = function.training
                    function.eval()
                    s = function(xy).numpy()
                    function.training = train_mode
                    if component is not None:
                        s = s[component]
                z[i,j] = s
    # else apply function directly to grid points
    else:
        z = function(xv,yv)

    if vmax is not None:
        z[z>vmax] = vmax

    # Setup plot
    return_axs = False
    if ax is None:
        return_axs = True
        _, ax = plt.subplots(figsize=(6,4.), dpi=100)

    # Color scheme
    if cmap is None:
        if mode == 'contourf':
            cmap = 'fessa'
        elif mode == 'contour':
            cmap = 'Greys_r'

    # Colorbar 
    if colorbar is None:
        if mode == 'contourf':
            colorbar = True
        elif mode == 'contour':
            colorbar = False

    # Plot
    if mode == 'contourf':
        pp = ax.contourf(xv,yv,z,levels=levels,cmap=cmap)
        if colorbar:
            plt.colorbar(pp,ax=ax)
    else:
        pp =  ax.contour(xv,yv,z,levels=levels,cmap=cmap)
    
    if return_axs:
        return ax

def plot_metrics(metrics, 
                keys = ['train_loss_epoch','valid_loss'],
                x = None, # 'epoch'
                labels = None, #['Train','Valid'],
                linestyles = None, #['-','--']
                colors = None, # ['fessa0','fessa1']
                xlabel='Epoch', ylabel='Loss', title='Learning curves', 
                yscale=None, ax = None):
    """ Plot logged metrics. """

    # Setup axis
    return_axs = False
    if ax is None:
        return_axs = True
        _, ax = plt.subplots(figsize=(5,4),dpi=100)

    # Plot metrics 
    auto_x = True if x is None else False
    for i,key in enumerate(keys):
        y = metrics[key]
        lstyle=linestyles[i] if linestyles is not None else None
        label=labels[i] if labels is not None else key
        color=colors[i] if colors is not None else None
        if auto_x:
            x = np.arange( len(y) )
        ax.plot(x,y,linestyle=lstyle,label=label,color=color)

    # Plot settings
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if yscale is not None:
        ax.set_yscale(yscale)

    ax.legend(ncol=1,frameon=False)

    if return_axs:
        return ax