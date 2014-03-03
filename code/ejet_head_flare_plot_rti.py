import numpy as np
import matplotlib.pyplot as plt
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
          'lines.markersize': 4,
          'lines.linewidth': 0.45,
          'text.usetex': False}
          #'text.latex.preamble': ['\usepackage{amsmath}']}
plt.rcParams.update(params)

basefilename = 'ejet_head_flare'
with open(basefilename + '_mf.pkl', 'rb') as f:
    mf = cPickle.load(f)

dpi = 110 # should be sized to match font size
savedpi = dpi*4 # should be a multiple of dpi
def plotter(z, t, r, **kwargs):
    xinches = len(t)/float(dpi)
    yinches = len(r)/float(dpi)
    figsize = (xinches + 1.5, yinches + 1)
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    el.make_axes_fixed(ax, xinches, yinches)
    img = el.rtiplot(z, t, r, ax=ax, csize=0.0625, cpad=0.05, **kwargs)
    plt.tight_layout(0.1)
    return fig

basedir = 'figures'
if not os.path.exists(basedir):
    os.makedirs(basedir)
cmap = copy.copy(plt.cm.coolwarm)
cmap.set_bad(cmap(0))

rslc = el.slice_by_value(mf.r, 85000, 115000)
for pslc in [slice(0, None, 5), slice(1, None, 5), slice(2, None, 5), 
             slice(3, None, 5), slice(4, None, 5), slice(None)]:
    fig4 = plotter(20*np.log10(np.abs(mf.vlt[pslc, rslc])/mf.noise_sigma), 
                   mf.t[pslc], mf.r[rslc]/1e3,
                   exact_ticks=False, vmin=0, vmax=40,
                   ylabel='Range (km)', clabel='SNR (dB)')
    path = os.path.join(basedir, basefilename + '_mf_rti_{0}.png')
    if pslc.start is not None:
        path = path.format(pslc.start)
    else:
        path = path.format('all')
    fig4.savefig(path, dpi=savedpi, bbox_inches='tight', pad_inches=0.05,
                 transparent=True)
    
    if pslc.start is not None:
        plt.close('all')

plt.show()