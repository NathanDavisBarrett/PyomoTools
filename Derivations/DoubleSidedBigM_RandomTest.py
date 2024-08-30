import numpy as np
import matplotlib.pyplot as plt

n = 10000
betaMin = -2
betaMax = 5

Bvals = np.random.uniform(betaMin,betaMax,n)
Xvals = np.random.uniform(0,1,n)
Avals = np.random.uniform(betaMin,betaMax,n)

for i in range(n):
    feasible = (Avals[i] >= betaMin * Xvals[i]) and (Avals[i] >= Bvals[i] + betaMax * (Xvals[i] - 1)) and (Avals[i] <= betaMax * Xvals[i]) and (Avals[i] <= Bvals[i] + betaMin*(Xvals[i] - 1))

    if not feasible:
        Bvals[i] = np.nan
        Xvals[i] = np.nan
        Avals[i] = np.nan


fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection="3d")

ax.scatter(Bvals,Xvals,Avals)
ax.set_xlabel("$B$")
ax.set_ylabel("$X$")
ax.set_zlabel("$A$")
ax.view_init(elev=10., azim=110)
ax.invert_xaxis()
ax.dist = 11

plt.show()