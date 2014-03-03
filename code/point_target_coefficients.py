import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools

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

ts = 1e-6
A = 1

ft_vals = [1e5, 1.5e5]
n_vals = [10.0, 25.0, 50.0]
nr = len(n_vals)
nc = len(ft_vals)
fig, axs = plt.subplots(nr, nc, sharex=True, sharey=True, figsize=(4.25,1.75))
#fig.subplots_adjust(left=0.075, bottom=0.115, right=0.915, top=0.93, wspace=0.05, hspace=0.05)
for r, c in itertools.product(xrange(nr), xrange(nc)):
    n = n_vals[r]
    ft = ft_vals[c]
    f = np.fft.fftshift(np.fft.fftfreq(int(n), ts))
    h = np.abs(A*np.sinc(n*ts*(f - ft))/np.sinc(ts*(f - ft)))
    ffun = np.arange(-1/ts/2, 1/ts/2, 1/ts/1000)
    hfun = np.abs(A*np.sinc(n*ts*(ffun - ft))/np.sinc(ts*(ffun - ft)))
    ax = axs[r, c]
    ax.plot(f/1e3, h, 'ko')
    ax.hold(True)
    ax.plot(ffun/1e3, hfun, 'k', 
            dashes=(params['lines.linewidth'],2*params['lines.linewidth']))
    
    if r == 0:
        ax.set_title(r'$f_t = {0:n}$ kHz'.format(ft/1e3))
    if r != (nr - 1):
        plt.setp(ax.get_xticklabels(), visible=False)
    else:
        ax.set_xlabel('Frequency (kHz)')
    if c == 0:
        if r == 1:
            ax.set_ylabel(r'$\left| h[n,p_t] \right|$')
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
    if c == (nc - 1):
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(r'$N = {0:n}$'.format(n), rotation=0)

plt.xlim(-1/ts/2/1e3, 1/ts/2/1e3)
plt.ylim(-0.1*A**2, 1.1*A**2)

plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)

fig.savefig('point_target_coefficients.pdf', bboz_inches='tight', 
            pad_inches=0.01, transparent=True)