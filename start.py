import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mplcyberpunk
import numpy as np
import scipy.stats as stats

def gradient_background(ax, cmap='gray', gmin=0.3, gmax=0.9):
    xmin, xmax = xlim = ax.get_xlim()
    ymin, ymax = ylim = ax.get_ylim()
    
    grad = np.atleast_2d(np.linspace(gmin, gmax,256)).T # Gradient of your choice

    ax.set(xlim=xlim, ylim=ylim, autoscale_on = False)
    ax.imshow(grad, interpolation='bicubic', cmap=cmap, vmin=0.0, vmax=1.0,
              extent=(xmin, xmax, ymin, ymax), alpha=0.7, aspect="auto", zorder=-1)
    #ax.axis([xlim[0], xlim[1], ylim[0], ylim[1]]) 

def gradient_bars(bars, cmap='cool', vmin=0.0, vmax=1.0):
    xlim = bars.get_xlim()
    ylim = bars.get_ylim()
    
    grad = np.atleast_2d(np.linspace(0.0,1.0,256)).T # Gradient of your choice
    from matplotlib.colors import LinearSegmentedColormap

    colors = [(0.6, 0.76, 0.98), (0, 0.21, 0.46)] # Experiment with this
    cm = LinearSegmentedColormap.from_list('gradient_bar', colors, N=256)

    rectangles = bars.containers[0]

    xList = []
    yList = []
    for rectangle in rectangles:
        x0 = rectangle.get_x()
        x1 = rectangle.get_x()+rectangle.get_width()
        y0 = rectangle.get_y()
        y1 = rectangle.get_y()+rectangle.get_height()

        xList.extend([x0,x1])
        yList.extend([y0,y1])

        bars.imshow(grad, extent=[x0,x1,y0,y1], cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", zorder=100)

    bars.axis([xlim[0], xlim[1], ylim[0], ylim[1]]) # *1.1 to add some buffer to top of plot

    return bars


def use_style(plt, style):
    if style == 'dark':
        plt.style.use('cyberpunk')

        params = {
            'font.family' : 'sans-serif', 
            'font.sans-serif': 'SimHei',
        }

        plt.rcParams.update(params)
    elif style == 'light':
        plt.style.use('fivethirtyeight')
        params = {
            'text.color':'k',
            'axes.labelcolor':'k',
            'xtick.color':'k',
            'ytick.color':'k'
        }
        plt.rcParams.update(params)

def show_grid(ax, xstep, ystep, minor_xstep=None, minor_ystep=None):
    # 以下用于显示网格
    xlim = ax.get_xlim()
    xlim1 = int(xlim[0]/xstep)*xstep
    ylim = ax.get_ylim()
    ylim1 = int(ylim[0]/ystep)*ystep

    if minor_xstep==None:
        minor_xstep = xstep/2
    if minor_ystep == None:
        minor_ystep = ystep/2

    ax.set_xticks(np.arange(xlim1,xlim[1]+xstep, xstep), minor=False)
    ax.set_xticks(np.arange(xlim1,xlim[1]+xstep, minor_xstep), minor=True)

    ax.set_yticks(np.arange(ylim1,ylim[1]+ystep, ystep), minor=False)
    ax.set_yticks(np.arange(ylim1,ylim[1]+ystep, minor_ystep), minor=True)

    #ax.grid(which="major",alpha=0.9)
    ax.grid(which="minor",alpha=0.3)

def normal_pdf(mu, sigma, nx=100):
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, nx)
    return x, stats.norm.pdf(x, mu, sigma)

def exponential_pdf(mean, nx=100):
    loc = 0

    xvalues = np.linspace(stats.expon.ppf(0.01, loc, mean), stats.expon.ppf(0.999, loc, mean), nx)
    return xvalues, stats.expon.pdf(xvalues, loc, mean)

def ecdf(data):
    """计算一维数组的ECDF。"""
 
    # 对输入的数组进行排序
    x = np.sort(data)
    
    # 计算x值的个数
    n = len(data)

    # 计算对应x值累积的个数
    
    y = np.arange(0, n)
    
    # x值对应的小于该值的记录数的累计百分比
    y = y / (n-1)

    return x, y

COLOR = 'k'

LARGEFONT=36
MEDIUMFONT=32
SMALLFONT=24
params = {
    'figure.figsize': (20, 10),
    "figure.autolayout": True,
    'font.family' : 'sans-serif', 
    'font.sans-serif': 'SimHei',
    'axes.labelsize': MEDIUMFONT,
    'axes.titlesize': MEDIUMFONT,
    'figure.titlesize':LARGEFONT,
    'xtick.labelsize': SMALLFONT,
    'ytick.labelsize': SMALLFONT,
    'legend.fontsize': SMALLFONT,
    'legend.title_fontsize': SMALLFONT,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.titleweight': 'bold',
    'font.weight': 'normal',
    'text.color':COLOR,
    'axes.labelcolor':COLOR,
    'xtick.color':COLOR,
    'ytick.color':COLOR
}

plt.rcParams.update(params)
plt.rc('axes', unicode_minus=False)

