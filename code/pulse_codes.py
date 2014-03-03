import numpy as np
from matplotlib import pyplot as plt

params = {'lines.linewidth': 0.45}
plt.rcParams.update(params)

N = 100
f0 = 1.0

def plotbinarypulse(s):
    t = np.arange(len(s))
    sN = np.repeat(s, N)
    tN = np.arange(len(sN)).astype(np.float_)/N
    y = -sN*np.cos(2*np.pi*f0*tN)
    
    fig = plt.figure(figsize=(4, 0.3))
    plt.bar(t, s, width=1, linewidth=0.45, color=(0.5, 0.8, 0.5))
    plt.plot(tN, y)
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    plt.xlim(-0.5, (len(t) + 0.5))
    plt.ylim(-1.1*np.max(np.abs(s)), 1.1*np.max(np.abs(s)))
    plt.tight_layout(0.1)
    
    return fig

def plotlfmpulse(m, bw):
    slope = float(bw)/m
    t = np.arange(m + 2)
    tN = np.arange((m + 2)*N).astype(np.float_)/N - 1
    f = f0 + (slope*tN - bw/2.)
    f[:N] = f0
    f[-N:] = f0
    y = -np.cos(2*np.pi*f*tN)
    y[:N] = 0
    y[-N:] = 0
    
    fig = plt.figure(figsize=(4, 0.3))
    plt.fill_between(tN, (f - f0)/(bw/2.), color=(0.8, 0.8, 0.5))
    plt.plot(tN, y)
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    plt.xlim(-1.5, (len(t) - 0.5))
    plt.ylim(-1.1, 1.1)
    plt.tight_layout(0.1)
    
    return fig

b13 = np.array([0, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 0]).astype(np.float_)
msl = np.array([0, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 0]).astype(np.float_)

figb13 = plotbinarypulse(b13)
figmsl = plotbinarypulse(msl)
figlfm = plotlfmpulse(10, 1.0)

figb13.savefig('barker13pulse.pdf', transparent=True)
figmsl.savefig('mslpulse.pdf', transparent=True)
figlfm.savefig('lfmpulse.pdf', transparent=True)

plt.show()