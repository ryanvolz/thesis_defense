import numpy as np
import matplotlib.pyplot as plt

params = {#'figure.figsize': (9.4, 5.4),
          #'figure.subplot.left': 0.01,
          #'figure.subplot.bottom': 0.01,
          #'figure.subplot.right': .99,
          #'figure.subplot.top': .99,
          #'figure.subplot.wspace': .025,
          #'figure.subplot.hspace': .025,
          'font.size': 16,
          'font.family': 'sans-serif',
          'font.sans-serif': ['Arial', 'sans-serif'],
          'pdf.fonttype': 42,
          'ps.fonttype': 42,
          #'ps.usedistiller': 'pdftk',
          'axes.titlesize': 16,
          'axes.labelsize': 16,
          'text.fontsize': 16,
          'legend.fontsize': 16,
          'xtick.labelsize': 14,
          'ytick.labelsize': 14,
          'lines.markersize': 4,
          'lines.linewidth': 0.75,
          'text.usetex': False}
          #'text.latex.preamble': ['\usepackage{amsmath}']}
plt.rcParams.update(params)

b13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]).astype(np.float_)

b13acorr = np.correlate(b13, b13, 'full')
d = np.fft.fftshift(np.fft.fftfreq(2*13 - 1)*(2*13 - 1))

plt.figure(figsize=(4, 3))
plt.plot(d, b13acorr)
plt.xlim(-12.5, 12.5)
plt.xlabel('Delay (samples)')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.savefig('b13acorr_analog.png', dpi=200, bbox_inches='tight', 
            pad_inches=0.05)

plt.figure(figsize=(4, 3))
plt.plot(d, b13acorr, 'bo')
plt.bar(d, b13acorr, width=0.1, color='blue', edgecolor='blue', align='center')
plt.xlim(-12.5, 12.5)
plt.xlabel('Delay (samples)')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.savefig('b13acorr_discrete.png', dpi=200, bbox_inches='tight', 
            pad_inches=0.05)

plt.show()