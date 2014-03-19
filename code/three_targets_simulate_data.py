import numpy as np
import scipy as sp
import scipy.constants
import cPickle

import radarmodel

msl = np.array([-1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1]).astype(np.float_)

m = 128
r = 1
s = msl/np.sqrt(len(msl))
n = 200#len(s)
ts = 1e-6
t0 = np.floor(2*90e3/sp.constants.c/ts)*ts
f0 = 440.2e6#49.92e6

lmbda_t = 2*100e3/sp.constants.c
f1 = 0
f2 = 2*50e3/sp.constants.c*f0
f3 = 2*45e3/sp.constants.c*f0

def target_signal(A, lt, ft):
    y = np.zeros(m, np.complex)

    for q in xrange(m):
        t = q*ts + t0
        #FIXME: why do I have to add 1 here to match timing of model?
        s_idx = int(np.floor((t - lt)/ts)) + 1
        if (s_idx < 0) or (s_idx >= len(s)):
            sq = 0
        else:
            sq = s[s_idx]
        #FIXME: use incorrect phase term to match old Matlab simulation
        y[q] = A*sq*np.exp(2*np.pi*1j*ft*(t - t0))*np.exp(-2*np.pi*1j*f0*lt)
        #y[q] = A*sq*np.exp(2*np.pi*1j*ft*t)*np.exp(-2*np.pi*1j*f0*lt)

    return y

y_calc1 = target_signal(20, lmbda_t, f1)
y_calc2 = target_signal(10, lmbda_t, f2)
y_calc3 = target_signal(10, lmbda_t, f3)
y_calc = y_calc1 + y_calc2 + y_calc3

A = radarmodel.point.Forward(s, n, m, r)
fs = np.fft.fftfreq(int(n), ts)

def g_N(f, N):
    return np.exp(-np.pi*1j*(2*N + n - 1)*ts*f)*np.sinc(n*ts*f)/np.sinc(ts*f)

def h_p(f, A, lt, ft):
    N = int(np.floor(lt/ts)) + 1
    return A*g_N(f - ft, N)*np.exp(-2*np.pi*1j*f0*lt)

h1 = h_p(fs, 20, lmbda_t, f1)
h2 = h_p(fs, 10, lmbda_t, f2)
h3 = h_p(fs, 10, lmbda_t, f3)
h = h1 + h2 + h3

k = int(np.floor((lmbda_t - t0)/ts)) + np.searchsorted(A.delays, 0)
hmat1 = np.zeros(A.inshape, h1.dtype)
hmat1[:, k] = h1
hmat2 = np.zeros(A.inshape, h2.dtype)
hmat2[:, k] = h2
hmat3 = np.zeros(A.inshape, h3.dtype)
hmat3[:, k] = h3
hmat = np.zeros(A.inshape, h.dtype)
hmat[:, k] = h

y_model = A(n*hmat)

# y_calc should equal y_model, but they do not for some values of n
# don't know where the problem is or which one is correct, so go with calc for now
y_noiseless = y_calc

noise_sigma = 5/np.sqrt(len(s))
noise = noise_sigma/np.sqrt(2)*np.random.randn(m) + 1j*noise_sigma/np.sqrt(2)*np.random.randn(m)
y = y_noiseless + noise

store = dict(y=y, y_noiseless=y_noiseless, h=hmat, s=s, ts=ts, t0=t0, f0=f0, noise_sigma=noise_sigma)
with open('three_targets_data.pkl', 'wb') as f:
    cPickle.dump(store, f, protocol=-1)
