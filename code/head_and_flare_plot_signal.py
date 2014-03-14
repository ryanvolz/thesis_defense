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

basefilename = 'head_and_flare'
with open(basefilename + '.pkl', 'rb') as f:
    data = cPickle.load(f)

vlt = data.vlt[22].real

plt.figure(figsize=(0.35, 1.1))
plt.plot([0, 0], [0, -len(vlt)], color=(0.5, 0.5, 0.5))
plt.plot(vlt, -np.arange(len(vlt)))
ax = plt.gca()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.tight_layout(0)

plt.savefig(basefilename + '_example_signal.pdf', dpi=180, 
            bbox_inches='tight', pad_inches=0, transparent=True)

plt.show()
plt.close('all')