import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

params = {#'figure.subplot.left': 0.01,
          #'figure.subplot.bottom': 0.01,
          #'figure.subplot.right': .99,
          #'figure.subplot.top': .99,
          #'figure.subplot.wspace': .025,
          #'figure.subplot.hspace': .025,
          'font.size': 8,
          'font.family': 'sans-serif',
          'font.sans-serif': ['Linux Biolinum O', 'Arial', 'sans-serif'],
          'pdf.fonttype': 42,
          'ps.fonttype': 42,
          #'ps.usedistiller': 'pdftk',
          'axes.titlesize': 8,
          'axes.labelsize': 8,
          'text.fontsize': 8,
          'legend.fontsize': 8,
          'xtick.labelsize': 6,
          'ytick.labelsize': 6,
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

N = 16
f = np.fft.fftshift(np.fft.fftfreq(N*100))
h = np.abs(np.sinc(N*f)/np.sinc(f))

fig = plt.figure(figsize=(1.5,0.6))
plt.plot(f, h, color=(0, 0.5, 0))
plt.xlim(-0.5, 0.5)
plt.xlabel('Normalized Frequency')
plt.text(-0.45, 0.8, r'$|b_k(f)|$', va='top')
plt.text(0.45, 0.8, r'$N={0}$'.format(N), va='top', ha='right')
ax = plt.gca()
ax.xaxis.tick_bottom()
ax.xaxis.labelpad = 1
ax.yaxis.set_visible(False)
ax.set_frame_on(False)
plt.tight_layout(0.1)

fig.savefig('wrapped_sinc.pdf', bboz_inches='tight', 
            pad_inches=0.01, transparent=True)

plt.show()
plt.close('all')