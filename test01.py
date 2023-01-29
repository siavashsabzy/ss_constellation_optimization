import numpy as np
import matplotlib.pyplot as plt
from ss_astro_lib import ss_j2_gravity,ss_rk4


t,y = ss_rk4(ss_j2_gravity,np.array([[0,6000]]),np.array([[5956.64397078590,3439.070,0.000,-2.017033,3.493604,6.455856]]),5000)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot3D(y[:,0],y[:,1],y[:,2])

plt.show()