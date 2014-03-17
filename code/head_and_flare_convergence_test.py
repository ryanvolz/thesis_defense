import numpy as np
import scipy as sp
import scipy.constants
import cPickle
from bunch import Bunch

import echolect as el
import radarmodel
import spyalg

basefilename = 'head_and_flare'
with open(basefilename + '.pkl', 'rb') as f:
    data = cPickle.load(f)

n = 128
m = data.vlt.shape[-1]
freqs = np.fft.fftfreq(int(n), data.ts/np.timedelta64(1, 's'))
v = freqs/data.f0*sp.constants.c/2

lmbda = data.noise_sigma/np.sqrt(n)*2

As = []
Astars = []
nodelays = []
for k, code in enumerate(data.codes):
    s = (code/np.linalg.norm(code)).astype(data.vlt.dtype)
    A = radarmodel.point.fastest_forward(s, n, m, 1)
    Astar = radarmodel.point.fastest_adjoint(s, n, m, 1)
    
    As.append(A)
    Astars.append(Astar)
    
    try:
        code_delay = data.code_delays[k]
    except:
        code_delay = 0
    
    delay = Astar.delays + code_delay
    nodelays.append(slice(np.searchsorted(delay, 0), np.searchsorted(delay, m)))

k = 201
y = data.vlt[k, :]
A = As[k % len(As)]
Astar = Astars[k % len(Astars)]
x0 = np.zeros(A.inshape, A.indtype)

sol0 = spyalg.standard_probs.l1rls_proxgrad(A, Astar, y, lmbda, x0, stepsize=1.0, backtrack=None, moreinfo=True)
sol1 = spyalg.standard_probs.l1rls_proxgradaccel(A, Astar, y, lmbda, x0, stepsize=1.0, backtrack=None, restart=False, moreinfo=True)
sol2 = spyalg.standard_probs.l1rls_proxgradaccel(A, Astar, y, lmbda, x0, stepsize=1.0, backtrack=None, restart=True, moreinfo=True)
sol3 = spyalg.standard_probs.l1rls_proxgradaccel(A, Astar, y, lmbda, x0, stepsize=1.0, backtrack=0.5, expand=1.25, restart=True, moreinfo=True)

sol4 = spyalg.standard_probs.l1rls_admmlin(A, Astar, y, lmbda, x0, stepsize=1.0, backtrack=0.5, expand=1.25, pen=1.0, residgap=2, penfactor=1.5, relax=1.0, moreinfo=True)
sol5 = spyalg.standard_probs.l1rls_pdhg(A, Astar, y, lmbda, x0, step_p=1.0, step_d=1.0, moreinfo=True)

print('proxgrad: {0} iterations'.format(sol0['numits']))
print('proxgradaccel: {0} iterations'.format(sol1['numits']))
print('proxgradaccel+restart: {0} iterations'.format(sol2['numits']))
print('proxgradaccel+restart+stepsize: {0} iterations and backtracks'.format(sol3['numits'] + sol3['backtracks']))
print('admmlin (adaptive step): {0} iterations and backtracks'.format(sol4['numits'] + sol4['backtracks']))
print('pdhg: {0} iterations'.format(sol5['numits']))