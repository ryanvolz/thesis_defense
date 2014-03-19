import numpy as np
import matplotlib.pyplot as plt
import cPickle

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
          'xtick.major.pad': 2,
          'ytick.major.size': 2,
          'ytick.major.pad': 3,
          'text.usetex': False}
          #'text.latex.preamble': ['\usepackage{amsmath}']}
plt.rcParams.update(params)

with open('three_targets_recovered.pkl', 'rb') as f:
    store = cPickle.load(f)

h_matched = store['h_matched']
h_cs = store['h_cs']
velocity_idx = store['velocity_idx']
range_idx = store['range_idx']

def plot_recovery(x):
    img = el.implot(np.abs(np.fft.fftshift(x, axes=0)), 
                    np.fft.fftshift(velocity_idx/1e3), 
                    range_idx/1e3,
                    xlabel='Range rate (km/s)',
                    ylabel='Range (km)',
                    clabel='Reflectivity',
                    exact_ticks=False,
                    xbins=4, ybins=5,
                    vmin=0, vmax=20,
                    csize=0.0625, cpad=0.05)
    img.axes.set_xlim(-15, 60)
    img.axes.set_ylim(98, 102)
    plt.tight_layout(0.1)
    return img

savedpi = 180

plt.figure(figsize=(2.25, 2.5))
img = plot_recovery(h_matched)
plt.savefig('three_targets_matched.pdf', dpi=savedpi, bbox_inches='tight', 
            pad_inches=0.025, transparent=True)

plt.figure(figsize=(2.25, 2.5))
img = plot_recovery(h_cs)
plt.savefig('three_targets_cs.pdf', dpi=savedpi, bbox_inches='tight', 
            pad_inches=0.025, transparent=True)

plt.show()