import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import gridspec
from mpl_toolkits import axes_grid1
import itertools

import spyalg

params = {'font.size': 10,
          'font.family': 'sans-serif',
          'font.sans-serif': ['Linux Biolinum O', 'Arial', 'sans-serif'],
          'pdf.fonttype': 42,
          'ps.fonttype': 42,
          'axes.titlesize': 10,
          'axes.labelsize': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'lines.markersize': 2, 
          'lines.linewidth': 0.45}
plt.rcParams.update(params)

savedpi = 72.27*4

def plotit(ax, x, thresh=None):
    markerline, stemlines, baseline = ax.stem(np.arange(len(x)), x)
    plt.setp(baseline, 'color', 'k')
    if thresh is not None:
        ax.plot([0, len(x)], [thresh, thresh], 'r-')
        ax.plot([0, len(x)], [-thresh, -thresh], 'r-')

def plotz(ax, z):
    ax.hlines(np.arange(len(z)), 0, z, color='green')
    ax.plot(z, np.arange(len(z)), 'go')
    ax.plot([0, 0], [0, len(z)], 'k-')

lmbda = 3
M = 20
N = 100

# sparse vector to measure
x = np.zeros(N)
x[25] = 10
x[60] = 100
x[65] = 70

# get measurement matrix
np.random.seed(0)
A = np.random.randn(M, N)
# normalize columns
A = A/np.sqrt(np.sum(A**2, axis=0))

# measurements
y = np.dot(A, x)

# set up figure
fig = plt.figure(figsize=(1.85, 2.7))
gs = gridspec.GridSpec(5, 1)
v = [axes_grid1.Size.Fixed(0.5)]
h1 = [axes_grid1.Size.Fixed(1.75)]
h2 = [axes_grid1.Size.Fixed(0.7), axes_grid1.Size.Fixed(0.35), axes_grid1.Size.Fixed(0.7)]
divs = []
locs = []
axs = []
for spec, h in zip(gs, [h1, h2, h1, h1, h1]):
    div = axes_grid1.SubplotDivider(fig, spec, horizontal=h, vertical=v)
    divs.append(div)
    if len(h) == 1:
        loc = div.new_locator(nx=0, ny=0)
    else:
        loc = div.new_locator(nx=1, ny=0)
    locs.append(loc)
    ax = fig.add_axes(loc(None, None))
    axs.append(ax)
    # locate the axes in the divider
    ax.set_axes_locator(loc)
    # also have to override get_subplotspec after setting locator
    # so tight_layout works
    ax.get_subplotspec = loc.get_subplotspec

def clear_frame():
    for ax in axs:
        ax.cla()
    for ax in [axs[0]] + axs[2:]:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_ylim(-105, 105)
    axs[1].axes.get_xaxis().set_visible(False)
    axs[1].axes.get_yaxis().set_visible(False)
    axs[1].set_xlim(-50, 50)

clear_frame()
plt.tight_layout(0.1)

plt.draw() # need draw to update axes position
# need the resolution to be multiples of 2 for libx264
savesize = np.floor(savedpi*fig.get_size_inches())
if np.any(np.mod(savesize, 2)):
    newsize = np.mod(savesize, 2)/savedpi + fig.get_size_inches()
    fig.set_size_inches(newsize, forward=True)
    savesize = np.floor(savedpi*newsize)

# iterative soft thresholding
xhat = np.zeros_like(x)
def ist():
    global xhat
    for k in xrange(1000):
        z = y - np.dot(A, xhat)
        Asz = np.dot(A.T, z)
        thresh = lmbda*spyalg.thresholding.medestnoise(Asz)
        xnew = spyalg.thresholding.softthresh(xhat + Asz, thresh).real
        
        yield k, xhat, z, Asz, thresh, xnew
        
        if np.max(np.abs(xnew - xhat)) < 1e-1:
            xnew = np.zeros_like(x)
            return
        
        xhat = xnew

def gen():
    it = ist()
    for kslow in xrange(5):
        state = next(it)
        for krepeat in xrange(16):
            yield state
    for state in it:
        yield state

def animate(framedata):
    clear_frame()
    k, xhat, z, Asz, thresh, xnew = framedata
    plotit(axs[0], xhat)
    axs[0].text(5, 95, '{0}'.format(k), ha='left', va='top')
    plotz(axs[1], z)
    plotit(axs[2], Asz)
    plotit(axs[3], xhat + Asz, thresh)
    plotit(axs[4], xnew)
    
anim = animation.FuncAnimation(fig, animate, init_func=clear_frame, 
                               frames=gen, interval=100, repeat_delay=1000, 
                               blit=False)

anim.save('ist_animation.mp4', 
          fps=8, dpi=savedpi, extra_args=['-vcodec', 'libx264', 
                                          '-g', '1'])

#plt.show()
plt.close('all')