import numpy as np
import matplotlib.pyplot as plt

import prx

params = {'lines.markersize': 2,
          'lines.linewidth': 0.45}
plt.rcParams.update(params)

def plotit(x, name, thresh=None):
    plt.figure(figsize=(1.75, 0.5))
    markerline, stemlines, baseline = plt.stem(np.arange(len(x)), x)
    plt.setp(baseline, 'color', 'k')
    if thresh is not None:
        plt.plot([0, len(x)], [thresh, thresh], 'r-')
        plt.plot([0, len(x)], [-thresh, -thresh], 'r-')
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.ylim(-105, 105)
    plt.tight_layout(0.1)
    plt.savefig(name, dpi=180, bbox_inches='tight', 
                pad_inches=0.01, transparent=True)

def plotz(z, name):
    plt.figure(figsize=(0.35, 0.5))
    plt.hlines(np.arange(len(z)), 0, z, color='green')
    plt.plot(z, np.arange(len(z)), 'go')
    plt.plot([0, 0], [0, len(z)], 'k-')
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.xlim(-50, 50)
    plt.tight_layout(0.1)
    plt.savefig(name, dpi=180, bbox_inches='tight', 
                pad_inches=0.01, transparent=True)

lmbda = 3
M = 20
N = 100

# sparse vector to measure
x = np.zeros(N)
x[25] = 10
x[60] = 100
x[65] = 70
plotit(x, 'ist/ist_x.png')

# get measurement matrix
np.random.seed(0)
A = np.random.randn(M, N)
# normalize columns
A = A/np.sqrt(np.sum(A**2, axis=0))

# measurements
y = np.dot(A, x)

# iterative soft thresholding
xhat = np.zeros_like(x)
plotit(xhat, 'ist/ist_x000.png')
for k in xrange(1000):
    z = y - np.dot(A, xhat)
    Asz = np.dot(A.T, z)
    thresh = lmbda*prx.thresholding.medestnoise(Asz)
    xnew = prx.thresholding.softthresh(xhat + Asz, thresh).real
    if k % 1 == 0:
        plotz(z, 'ist/ist_z{0:03d}.png'.format(k))
        plotit(Asz, 'ist/ist_Asz{0:03d}.png'.format(k))
        plotit(xhat + Asz, 'ist/ist_xpt{0:03d}'.format(k), thresh)
        plotit(xnew, 'ist/ist_x{0:03d}.png'.format(k+1))
    if np.max(np.abs(xnew - xhat)) < 1e-1:
        break
    xhat = xnew

plt.close('all')
plt.show()