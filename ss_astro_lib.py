import numpy as np
def ss_rk4(func,tspan,y0,n):
    #y0 = np.transpose(y0)
    m = np.size(y0,1)
    t = np.zeros([n+1,1])
    y = np.zeros([n+1,m])
    tfirst = tspan[0,0]
    tlast = tspan[0,1]
    dt = (tlast - tfirst) / n
    t[0,0] = tspan[0,0]
    y[0,:] = y0
    for i in range(n) :
            f1 = func ( t[i,0] , y[i,:])
            f2 = func ( t[i,0] + dt / 2.0, y[i,:] + dt * np.transpose ( f1 ) / 2.0 )
            f3 = func ( t[i,0] + dt / 2.0, y[i,:] + dt * np.transpose ( f2 ) / 2.0 )
            f4 = func ( t[i,0] + dt,       y[i,:] + dt * np.transpose ( f3 ) )
            t[i+1,0] = t[i,0] + dt
            y[i+1,:] = y[i,:] + dt * ( np.transpose ( f1 ) + 2.0 * np.transpose ( f2 ) + 2.0 * np.transpose ( f3 ) + np.transpose ( f4 ) ) / 6.0      
    return t,y

def ss_j2_gravity(t,xx):
    x = np.zeros([1,6])
    x[0,:] = xx
    xdot = np.zeros([6,1])
    Re  = 6378.137
    j2  = 1.08262668e-3
    mu  = 398600.4415
    r = (x[0,0] ** 2 + x[0,1] ** 2 + x[0,2] ** 2) ** 0.5
    xdot[0,0] = x[0,3]
    xdot[1,0] = x[0,4]
    xdot[2,0] = x[0,5]
    xdot[3,0] = -mu*x[0,0]/r**3*  (1+1.5*j2* (Re/r)**2 * (1-5*(x[0,3]/r)**2))
    xdot[4,0] = -mu*x[0,1]/r**3*  (1+1.5*j2* (Re/r)**2 * (1-5*(x[0,3]/r)**2))
    xdot[5,0] = -mu*x[0,2]/r**3*  (1+1.5*j2* (Re/r)**2 * (3-5*(x[0,3]/r)**2))
    return xdot