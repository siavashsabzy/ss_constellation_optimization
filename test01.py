import numpy as np
import matplotlib.pyplot as plt
from ss_astro_lib import *
epoch = np.array([2022,11,1,0,3,24])
refsat = np.append(epoch,np.array([7078,0,60,0,0,0]))
eci = ss_coe2eci_multi([refsat])
ecef = ss_eci2ecef_multi(eci)
aer = ss_ecef_aer([6378.137,0,0],ecef[0][6:9])
print(aer[1]*180/np.pi)
'''
epoch = np.array([2020,1,1,20,10,15.500])
eci = np.array([5956.64397078590,3439.070,0.000,-2.017033,3.493604,6.455856])
coe = ss_eci2coe(eci)
refsat = np.append(epoch,np.array([7078.137,0,30,0,0,0]))
e = ss_walker(5, 5, 4, refsat)
h = ss_coe2eci_multi(e)
d = ss_eci2ecef_multi(h)
t,y = ss_rk4(ss_j2_gravity,np.array([[0,6000]]),eci,5000)

u = np.linspace(0, np.pi, 30)
v = np.linspace(0, 2 * np.pi, 30)
xx = np.outer(np.sin(u), np.sin(v))
yy = np.outer(np.sin(u), np.cos(v))
zz = np.outer(np.cos(u), np.ones_like(v))
#ax.plot_wireframe(xx*6378.137, yy*6378.137, zz*6378.137)
ax.plot3D(y[:,0],y[:,1],y[:,2],'.r')

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(d[:,0],d[:,1],d[:,2],'*k')
plt.show()
'''