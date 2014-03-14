import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import copy

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
          'lines.markersize': 2,
          'lines.linewidth': 0.45,
          'axes.linewidth': 0.45,
          'xtick.major.size': 2,
          'xtick.major.pad': 2,
          'ytick.major.size': 2,
          'ytick.major.pad': 3,
          'text.usetex': False}
          #'text.latex.preamble': ['\usepackage{amsmath}']}
plt.rcParams.update(params)

# f(x) = sum(exp(A*x + b))
A = np.array([[1, 3], [1, -3], [-1, 0]], np.float_)
b = -0.1*np.ones(3, np.float_)
def F(x):
    return np.sum(np.exp(np.dot(A, x)) + b)
def gradF(x):
    return np.dot(A.T, np.exp(np.dot(A, x) + b))
def hessF(x):
    return np.dot(np.dot(A.T, np.diag(np.dot(A, x) + b)), A)

# backtracking parameters
alpha = 0.1
beta = 0.7

# starting point
x0 = np.array([-1, 1])
f0 = F(x0)

# calculate the minimum using Newton steps
x = x0
f = f0
#for i in xrange(10):
    #grad = gradF(x)
    #hess = hessF(x)
    #v = -np.linalg.solve(hess, grad)
    
    #if np.dot(grad, -v)/2 < 1e-6:
        #break
    
    #s = 1
    #for k in xrange(100):
        #xnew = x + s*v
        #fxnew = F(xnew)
        #if fxnew < (f + s*alpha*np.dot(grad, v)):
            #break
        #else:
            #s = s*beta
    
    #x = x + s*v
    #f = F(x)
maxits = 100
x = x0
for i in xrange(maxits):
    f = F(x)
    
    g = gradF(x)
    v = -g
    
    s = 1
    for k in xrange(10):
        xnew = x + s*v
        fxnew = F(xnew)
        if fxnew < (f + s*alpha*np.dot(g, v)):
            break
        else:
            s = s*beta
    
    x = x + s*v
xmin = x
fmin = F(xmin)

# calculate the contour lines at f0 + k*delta
X = np.linspace(-3, 2, 100)
Y = np.linspace(-1.5, 1.5, 100)
XX, YY = np.meshgrid(X, Y)
XY = np.concatenate((XX[:, :, np.newaxis], YY[:, :, np.newaxis]), -1)
def Farr(X):
    return np.sum(np.exp(np.tensordot(X, A, axes=([-1], [1])) + b), axis=-1)
Z = Farr(XY)
K = 5
delta = (f0 - fmin)/(K - 1)
Zlevels = f0 + (np.arange(K) - K + 2)*delta

# run the gradient method
maxits = 100
xs = np.zeros((maxits, 2), np.float_)
fs = np.zeros(maxits, np.float_)
x = x0
for i in xrange(maxits):
    f = F(x)
    xs[i, :] = x
    fs[i] = f
    
    if (f - fmin) < 1e-4:
        break
    
    g = gradF(x)
    v = -g
    
    s = 1
    for k in xrange(10):
        xnew = x + s*v
        fxnew = F(xnew)
        if fxnew < (f + s*alpha*np.dot(g, v)):
            break
        else:
            s = s*beta
    
    x = x + s*v

xs = xs[:(i+1), :]
fs = fs[:(i+1)]

fig = plt.figure(figsize=(4, 2))
cmap = copy.deepcopy(plt.cm.Blues_r)
cmap.set_over((1, 1, 1))
# make alpha fade out with high values
#cmap._lut[:-3, -1] = 1 - np.linspace(0, 1, cmap._lut.shape[0] - 3)
plt.pcolormesh(X, Y, Z, vmax=Zlevels[-1], shading='gouraud', cmap=cmap)
plt.contour(X, Y, Z, levels=Zlevels, colors=(0.25,0.25,0.25), linestyles='dashed')
line, = plt.plot([], [], 'wo-', mec='w')
ax = plt.gca()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.set_frame_on(False)
plt.tight_layout(0)

def init_frame():
    line.set_data([], [])
    
    return line

def animate(k):
    line.set_data(xs[:k, 0], xs[:k, 1])
    
    return line

anim = animation.FuncAnimation(fig, animate, init_func=init_frame,
                               frames=xs.shape[0], interval=500, repeat_delay=500, 
                               blit=False)

anim.save('gradient_descent_animation.mp4', 
          fps=2, dpi=300, extra_args=['-vcodec', 'libx264', 
                                          '-g', '5', # key frame interval
                                          '-r', '6']) # output framerate (>=6 or else vlc won't play)

#plt.show()
plt.close('all')