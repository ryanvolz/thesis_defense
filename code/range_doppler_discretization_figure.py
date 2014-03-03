import numpy as np
import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

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

delta = 0.1
x = y = np.arange(0, 100, delta)
X, Y = np.meshgrid(x,y)
Z1 = mlab.bivariate_normal(X, Y, 5, 10, 25, 30)
Z1 = Z1/np.max(Z1)
Z2 = mlab.bivariate_normal(X, Y, 1, 1, 71, 61)
Z2 = Z2/np.max(Z2)

f = x/100. - 0.5
d = y

fig = plt.figure(figsize=(2,2))
img = el.implot(10*(Z1 + 2*Z2).T, f, d, 
                csize=0.0625, cpad=0.05, 
                exact_ticks=False, xbins=6, ybins=6, 
                interpolation='bilinear',
                xlabel='Normalized Frequency', ylabel='Delay (samples)', 
                clabel='Reflectivity')
ax = img.axes
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.02))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2.0))
ax.grid(which='minor', color='m', linewidth=0.45, linestyle='-')

#plt.show()
fig.savefig('range_doppler_discretization.pdf', bbox_inches='tight', 
            pad_inches=0.01, transparent=True)
