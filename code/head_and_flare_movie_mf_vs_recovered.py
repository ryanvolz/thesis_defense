import numpy as np
import scipy as sp
import scipy.constants
from matplotlib import pyplot as plt
from matplotlib import animation
import cPickle
import copy
import os

import echolect as el

params = {#'figure.subplot.left': 0.01,
          #'figure.subplot.bottom': 0.01,
          #'figure.subplot.right': .99,
          #'figure.subplot.top': .99,
          #'figure.subplot.wspace': .025,
          #'figure.subplot.hspace': .025,
          'font.size': 10,
          'font.family': 'sans-serif',
          'font.sans-serif': ['Linux Biolinum O', 'Arial', 'sans-serif'],
          'pdf.fonttype': 42,
          'ps.fonttype': 42,
          #'ps.usedistiller': 'pdftk',
          'axes.titlesize': 10,
          'axes.labelsize': 10,
          'text.fontsize': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'lines.markersize': 1,
          'lines.linewidth': 0.45,
          'axes.linewidth': 0.45,
          'xtick.major.size': 2,
          'ytick.major.size': 2,
          'text.usetex': False}
          #'text.latex.preamble': ['\usepackage{amsmath}']}
plt.rcParams.update(params)

basefilename = 'head_and_flare'
with open(basefilename + '.pkl', 'rb') as f:
    data = cPickle.load(f)

with open(basefilename + '_recovered.pkl', 'rb') as f:
    cs = cPickle.load(f)

n = 128
m = data.vlt.shape[-1]
freqs = np.fft.fftfreq(int(n), data.ts/np.timedelta64(1, 's'))
v = freqs/data.f0*sp.constants.c/2

filts = []
for code in data.codes:
    s = (code/np.linalg.norm(code)).astype(data.vlt.dtype)
    filt = el.filtering.MatchedDoppler(s, n, m, xdtype=data.vlt.dtype)
    filts.append(filt)

imgdpi = 225 # should be sized to match font size
savedpi = imgdpi*2 # should be a multiple of imgdpi
xstretch = 3
ystretch = 2
pixelaspect = float(xstretch)/ystretch

basedir = 'movies'
if not os.path.exists(basedir):
    os.makedirs(basedir)

cmap = copy.copy(plt.cm.coolwarm)
cmap.set_bad(cmap(0))

for kp in xrange(5):
    pslc = slice(kp, None, 5)
    vlt = data.vlt[pslc]
    cs_sig = cs.vlt_sig[pslc]
    cs_noise = cs.vlt_noise[pslc]
    
    filt = filts[kp]

    zs = np.zeros((n, m), np.float_)
    
    xinches = len(v)/float(imgdpi)*xstretch
    yinches = len(data.r)/float(imgdpi)*ystretch
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(2*xinches + 1, yinches + 0.55))
    el.make_axes_fixed(axes[0], xinches, yinches)
    el.make_axes_fixed(axes[1], xinches, yinches)
    mfimg = el.implot(zs, 
                      np.fft.fftshift(v)/1e3, data.r/1e3,
                      xlabel='Doppler range rate (km/s)', ylabel='Range (km)',
                      cbar=False, title='Matched Filter', 
                      exact_ticks=False, xbins=5,
                      vmin=0, vmax=40, 
                      cmap=cmap, csize=0.0625, cpad=0.05,
                      pixelaspect=pixelaspect, ax=axes[0])
    csimg = el.implot(zs, 
                      np.fft.fftshift(cs.v)/1e3, cs.r/1e3,
                      xlabel='Doppler range rate (km/s)',
                      clabel='SNR (dB)', title='Waveform Inversion',
                      exact_ticks=False, xbins=5,
                      vmin=0, vmax=40, 
                      cmap=cmap, csize=0.0625, cpad=0.05,
                      pixelaspect=pixelaspect, ax=axes[1])
    plt.tight_layout(0.1)
    plt.draw() # need draw to update axes position
    # need the resolution to be multiples of 2 for libx264
    savesize = np.floor(savedpi*fig.get_size_inches())
    if np.any(np.mod(savesize, 2)):
        newsize = np.mod(savesize, 2)/savedpi + fig.get_size_inches()
        fig.set_size_inches(newsize, forward=True)
        savesize = np.floor(savedpi*newsize)

    def init_frame():
        mfimg.set_data(zs.T)
        csimg.set_data(zs.T)
        return mfimg, csimg

    def animate(kf):
        vlt_mf = filt(vlt[kf])[:, filt.validsame]
        mfimg.set_data(20*np.log10(np.abs(np.fft.fftshift(vlt_mf, 
                                                          axes=0))/data.noise_sigma).T)
        csimg.set_data(20*np.log10(np.abs(np.fft.fftshift(cs_sig[kf] + cs_noise[kf], 
                                                          axes=0))/cs.noise_sigma).T)
        return mfimg, csimg

    anim = animation.FuncAnimation(fig, animate, init_func=init_frame, 
                                   frames=vlt.shape[0], interval=100, blit=False)
    anim.save(os.path.join(basedir, basefilename + '_mf_vs_recovered_{0}.mp4').format(kp), 
              dpi=savedpi, extra_args=['-vcodec', 'libx264'])

#plt.show()
plt.close('all')