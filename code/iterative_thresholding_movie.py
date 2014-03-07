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

for ax in [axs[0]] + axs[2:]:
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_ylim(-105, 105)
axs[1].axes.get_xaxis().set_visible(False)
axs[1].axes.get_yaxis().set_visible(False)
axs[1].set_xlim(-50, 50)

# x stem
ml0, sl0, bl0 = axs[0].stem(np.arange(len(x)), x)
plt.setp(bl0, 'color', 'k')
# z stem
sl1 = axs[1].hlines(np.arange(len(y)), 0, y, color='green')
ml1, = axs[1].plot(y, np.arange(len(y)), 'go')
bl1, = axs[1].plot([0, 0], [0, len(y)], 'k-')
# Asz stem
ml2, sl2, bl2 = axs[2].stem(np.arange(len(x)), x)
plt.setp(bl2, 'color', 'k')
# x + Asz stem
ml3, sl3, bl3 = axs[3].stem(np.arange(len(x)), x)
plt.setp(bl3, 'color', 'k')
tul, = axs[3].plot([0, len(x)], [0, 0], 'r-')
tll, = axs[3].plot([0, len(x)], [0, 0], 'r-')
# xnew stem
ml4, sl4, bl4 = axs[4].stem(np.arange(len(x)), x)
plt.setp(bl4, 'color', 'k')
txt = axs[0].text(5, 95, '', ha='left', va='top')

artists = [ml0, ml1, ml2, ml3, ml4, sl1, tul, tll, txt] + sl0 + sl2 + sl3 + sl4

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
        
        print('{0}: {1}'.format(k, np.max(np.abs(xnew - xhat))))
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

def clear_frame():
    for a in artists:
        a.set_visible(False)

def animate(framedata):
    for a in artists:
        a.set_visible(True)
    k, xhat, z, Asz, thresh, xnew = framedata
    
    txt.set_text('{0}'.format(k))
    ml0.set_ydata(xhat)
    for a, d in zip(sl0, xhat):
        a.set_ydata([0, d])
    verts = [ ((thisxmin, thisy), (thisxmax, thisy))
              for thisxmin, thisxmax, thisy in zip(np.zeros(len(z)), z, np.arange(len(z)))]
    sl1.set_segments(verts)
    ml1.set_xdata(z)
    ml2.set_ydata(Asz)
    for a, d in zip(sl2, Asz):
        a.set_ydata([0, d])
    ml3.set_ydata(xhat + Asz)
    for a, d in zip(sl3, xhat + Asz):
        a.set_ydata([0, d])
    tul.set_ydata([thresh, thresh])
    tll.set_ydata([-thresh, -thresh])
    ml4.set_ydata(xnew)
    for a, d in zip(sl4, xnew):
        a.set_ydata([0, d])
    
anim = animation.FuncAnimation(fig, animate, init_func=clear_frame, 
                               frames=gen, interval=100, repeat_delay=1000, 
                               save_count=200, blit=False)

anim.save('ist_animation.mp4', 
          fps=8, dpi=savedpi, 
          extra_args=['-vcodec', 'libx264', 
                      '-g', '1'])

#plt.show()
plt.close('all')