import numpy as np
from matplotlib import pyplot as plt

params = {'lines.linewidth': 0.45}
plt.rcParams.update(params)

def plotpulse(t, s, f0):
    y = s*np.cos(2*np.pi*f0*t)
    
    fig = plt.figure(figsize=(2, 0.3))
    plt.bar(t, 2*np.abs(s), width=tstep, bottom=-np.abs(s), 
            linewidth=0, color=(0.5, 0.8, 0.5))
    plt.plot(t, y)
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    plt.xlim(-0.5*tstep, (len(t) + 0.5)*tstep)
    plt.ylim(-1.1, 1.1)
    plt.tight_layout(0.1)
    
    return fig

f0 = 4e5
tstep = 1e-7
t = np.arange(0, 50e-6, tstep)
ss = np.zeros(len(t), np.float_)
ss[25:75] = 1
sl = np.zeros_like(ss)
sl[25:325] = 1

figs = plotpulse(t, ss, f0)
figl = plotpulse(t, sl, f0)

figs.savefig('shortpulse.pdf', transparent=True)
figl.savefig('longpulse.pdf', transparent=True)

plt.show()