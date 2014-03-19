import numpy as np
import scipy as sp
import scipy.constants
import cPickle

import echolect as el
import radarmodel
import spyalg

with open('three_targets_data.pkl', 'rb') as f:
    store = cPickle.load(f)

y = store['y']
s = store['s']
ts = store['ts']
t0 = store['t0']
f0 = store['f0']
noise_sigma = store['noise_sigma']
m = len(y)

r = 1
n = 200#len(s)

filt = el.filtering.MatchedDoppler(s, n, m, xdtype=np.complex_)
A = radarmodel.point.Forward(s, n, m, r)
Astar = radarmodel.point.Adjoint(s, n, m, r)

# matched filter recovery
h_matched = filt(y)[:, filt.nodelay]

# compressed sensing recovery
x0 = np.zeros(A.inshape, A.indtype)
x1 = spyalg.l1rls(A, Astar, y, lmbda=.125, x0=x0, printrate=10)

x = x1/np.sqrt(n) + Astar(y - A(x1))*np.sqrt(n)

h_cs = x[:, Astar.delays >= 0]
    
range_idx = (t0 + np.arange(m)*ts)*sp.constants.c/2
velocity_idx = np.fft.fftfreq(int(n), ts)/f0*sp.constants.c/2

recovered = dict(h_matched=h_matched, h_cs=h_cs,
                 range_idx=range_idx, velocity_idx=velocity_idx)
with open('three_targets_recovered.pkl', 'wb') as f:
    cPickle.dump(recovered, f, protocol=-1)
