import numpy as np

Re = 6378.137  # the earth radii in [km]
e_ecc = 8.181919084261345e-2  # the earth eccentricity [ndim]
e_f = 1 / 298.257223563  # The earth flattering parameter [ndim]
j2 = 1.08262668e-3  # the earth J2 coefficient [ndim]
mu = 398600.4415  # gravitational constant [km^3/s^2]


def ss_rk4(func, tspan, y0, n):  # simple Runge–Kutta method for integration
    # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    # y0 = np.transpose(y0)
    m = np.size(y0) # determining the size of output
    t: np.ndarray = np.zeros([n + 1, 1])   # building time vector
    y: np.ndarray = np.zeros([n + 1, m])   # building the state vector
    dt = (tspan[0, 1] - tspan[0, 0]) / n   # finding time step value
    t[0, 0] = tspan[0, 0]   # specifying the first time value
    y[0, :] = y0   # specifying the first state value
    for i in range(n):
        f1 = func(t[i, 0], y[i, :])
        f2 = func(t[i, 0] + dt / 2.0, y[i, :] + dt * np.transpose(f1) / 2.0)
        f3 = func(t[i, 0] + dt / 2.0, y[i, :] + dt * np.transpose(f2) / 2.0)
        f4 = func(t[i, 0] + dt, y[i, :] + dt * np.transpose(f3))
        t[i + 1, 0] = t[i, 0] + dt
        y[i + 1, :] = y[i, :] + dt * (
                    np.transpose(f1) + 2.0 * np.transpose(f2) + 2.0 * np.transpose(f3) + np.transpose(f4)) / 6.0
    return t, y


def ss_j2_gravity(t, xx):   # equations of motion for satellite under j2 perturbation
    x = np.zeros([1, 6])   # state vector
    x[0, :] = xx
    xdot = np.zeros([6, 1])
    r = (x[0, 0] ** 2 + x[0, 1] ** 2 + x[0, 2] ** 2) ** 0.5
    xdot[0, 0] = x[0, 3]
    xdot[1, 0] = x[0, 4]
    xdot[2, 0] = x[0, 5]
    xdot[3, 0] = -mu * x[0, 0] / r ** 3 * (1 + 1.5 * j2 * (Re / r) ** 2 * (1 - 5 * (x[0, 3] / r) ** 2))
    xdot[4, 0] = -mu * x[0, 1] / r ** 3 * (1 + 1.5 * j2 * (Re / r) ** 2 * (1 - 5 * (x[0, 3] / r) ** 2))
    xdot[5, 0] = -mu * x[0, 2] / r ** 3 * (1 + 1.5 * j2 * (Re / r) ** 2 * (3 - 5 * (x[0, 3] / r) ** 2))
    return xdot


def ss_coe2eci(coe):  # transformation from classical orbital elemens to inertial vectors
    ssdeg2rad_local = np.pi / 180  # defining a local variable for degree to radians
    a = coe[0]
    e = coe[1]
    i = coe[2] * ssdeg2rad_local  # inclination
    raan = coe[3] * ssdeg2rad_local  # Right ascension of the ascending node
    aop = coe[4] * ssdeg2rad_local  # argument of pregee
    ta = coe[5] * ssdeg2rad_local  # true anomaly
    r = (a * (1 - e ** 2)) / (1 + e * np.cos(ta))  # Get position from orbit formula
    h = np.sqrt(mu * a * (1 - e ** 2))  # Magnitude of specific angular momentum
    x = r * (np.cos(raan) * np.cos(aop + ta) - np.sin(raan) * np.sin(aop + ta) * np.cos(i))  # Position X-component
    y = r * (np.sin(raan) * np.cos(aop + ta) + np.cos(raan) * np.sin(aop + ta) * np.cos(i))  # Position Y-component
    z = r * (np.sin(i) * np.sin(aop + ta))  # Position Z-component
    p = a * (1 - e ** 2)  # Semilatus Rectum
    x_dot = ((x * h * e) / (r * p)) * np.sin(ta) - ((h / r) * (
                np.cos(raan) * np.sin(aop + ta) + np.sin(raan) * np.cos(aop + ta) * np.cos(i)))  # Velocity X-component
    y_dot = ((y * h * e) / (r * p)) * np.sin(ta) - ((h / r) * (
                np.sin(raan) * np.sin(aop + ta) - np.cos(raan) * np.cos(aop + ta) * np.cos(i)))  # Velocity Y-component
    z_dot = ((z * h * e) / (r * p)) * np.sin(ta) + ((h / r) * (np.sin(i) * np.cos(aop + ta)))  # Velocity Z-component
    eci_vector = np.array(
        [x, y, z, x_dot, y_dot, z_dot])  # Put X,Y,Z, X_dot, Y_dot, Z_dot into an array to create a vector
    return (eci_vector)


def ss_eci2coe(eci):
    r_vector = eci[0:3]
    v_vector = eci[3:6]
    # Used "Orbital Mechanics for Engineering Students"  3rd Edition by Howard D. Curtis; Also used our class notes
    r = np.linalg.norm(r_vector)  # Distance
    v = np.linalg.norm(v_vector)  # MAgnitude of Velocity or Speed
    v_r = np.dot(v_vector, r_vector) / r  # radial velocity
    h_vector = np.cross(r_vector, v_vector)  # specific angular momentum vector
    h = np.linalg.norm(h_vector)  # magnitude of specific angular momentum
    i = np.arccos(h_vector[2] / h)  # inclination
    n_vector = np.cross([0, 0, 1], h_vector)  # vector pointing to ascending node
    n = np.linalg.norm(n_vector)  # magnitude of n
    if n_vector[1] > 0:
        raan = np.arccos(n_vector[0] / n)  # Right Ascension of the Ascending node
    if n_vector[1] < 0:
        raan = 2 * np.pi - np.arccos(n_vector[0] / n)
    e_vector = (1 / mu) * ((((v ** 2) - (mu / r)) * (r_vector)) - ((r * v_r) * v_vector))  # eccentricity vector
    e = np.linalg.norm(e_vector)  # eccentricity
    if e_vector[2] > 0:
        aop = np.arccos(np.dot(n_vector, e_vector) / (n * e))  # Argument of periapse
    if e_vector[2] < 0:
        aop = 2 * np.pi - np.arccos(np.dot(n_vector, e_vector) / (n * e))
    if v_r > 0:
        ta = np.arccos((np.dot(e_vector, r_vector) / (e * r)))  # True anomaly
    if v_r < 0:
        ta = 2 * np.pi - np.arccos((r_vector / r) * (e_vector / e))
    Energy = (v ** 2 / 2) - (mu / r)
    if e == 1:
        p = h ** 2 / mu  # Semilatus Rectum
        return "Orbit is parabolic. Eccentricity = infinity"
    else:
        a = -mu / (2 * Energy)  # Semi-major Axis
        p = a * (1 - e ** 2)
    return [a, e, i * (180 / np.pi), raan * (180 / np.pi), aop * (180 / np.pi), ta * (180 / np.pi)]


def ss_dcm_eci_ecef(UTC):
    #   Gets the Direction Cosine Matrix for conversion between ECI and ECEF
    #   Input: UTC time (6x1)
    #   Output: DCM matrix (3x3)
    year = UTC[0]
    mes = UTC[1]
    dia = UTC[2]
    hora = UTC[3]
    minuto = UTC[4]
    segundo = UTC[5]
    # get modified julian day (mjd)
    mjd = 367 * year + dia - 712269 + np.fix(275 * mes / 9) - np.fix(7 * (year + np.fix((mes + 9) / 12)) / 4)
    # get fraction of the day (dfra)
    dfra = segundo + 60 * (minuto + 60 * hora)
    # get Greenwich Sidereal Time (GWST)
    tsj = (mjd - 18262.5) / 36525
    tsgo = (24110.54841 + (8640184.812866 + 9.3104e-2 * tsj - 6.2e-6 * tsj * tsj) * tsj) * np.pi / 43200
    tetp = 7.292116e-5;  # Earth angular velocity (rad/s)
    gwst = np.mod(tsgo + dfra * tetp, 2 * np.pi)
    # get DCM from ECI to ECEF
    coan = np.cos(gwst)
    sian = np.sin(gwst)
    DCM_eci2ecef = np.array([[coan, sian, 0], [-sian, coan, 0], [0, 0, 1]])
    return DCM_eci2ecef


def ss_walker(num_plane, num_sat, F, refsat):
    # allsat = [[0 for i in range(num_sat)] for i in range(num_plane)]
    sats = np.zeros([num_plane * num_sat, 12])
    raan0 = refsat[9]
    ta0 = refsat[11]
    for i in range(num_plane):
        for j in range(num_sat):
            raan = raan0 + i * 360 / num_plane
            ta = ta0 + j * 360 / num_sat + i * 360 * F / (num_sat * num_plane)
            ta = ta % 360
            if ta >= 180:
                ta -= 360
            sats[(i * num_sat) + j][:] = np.append(refsat[0:6],
                                                   np.array([refsat[6], refsat[7], refsat[8], raan, refsat[10], ta]))
    return sats


def ss_coe2eci_multi(sats):   
    # converting classical orbital elements to earth centered inertial vector (multiple inputs)
    no_sats = np.size(sats, 0)
    eci_single = np.zeros([1, 6])
    eci = np.zeros([no_sats, 12])
    for i in range(no_sats):
        eci_single = ss_coe2eci(sats[i][6:12])
        eci[i][:] = np.append(sats[i][0:6], eci_single)
    return eci


def ss_eci2ecef_multi(sats):
    no_sats = len(sats)
    pos_temp = np.zeros([3, 1])
    vel_temp = np.zeros([3, 1])
    ecef_single = np.zeros([1, 6])
    ecef = np.zeros([no_sats, 12])
    for i in range(no_sats):
        pos_temp = np.matmul(ss_dcm_eci_ecef(sats[i][0:6]), np.transpose(sats[i][6:9]))
        vel_temp = np.matmul(ss_dcm_eci_ecef(sats[i][0:6]), np.transpose(sats[i][9:12]))
        ecef_single = np.append(np.transpose(pos_temp), np.transpose(vel_temp))
        ecef[i][:] = np.append(sats[i][0:6], ecef_single)
    return ecef


def ss_ecef2eci_multi(sats):
    no_sats = len(sats)
    pos_temp = np.zeros([3, 1])
    vel_temp = np.zeros([3, 1])
    eci_single = np.zeros([1, 6])
    eci = np.zeros([no_sats, 12])
    for i in range(no_sats):
        pos_temp = np.matmul(np.transpose(ss_dcm_eci_ecef(sats[i][0:6])), np.transpose(sats[i][6:9]))
        vel_temp = np.matmul(np.transpose(ss_dcm_eci_ecef(sats[i][0:6])), np.transpose(sats[i][9:12]))
        eci_single = np.append(np.transpose(pos_temp), np.transpose(vel_temp))
        eci[i][:] = np.append(sats[i][0:6], eci_single)
    return eci


def ss_gridsphere(n):
    a = (4 * np.pi) / n
    d = np.sqrt(a)
    m = np.round(np.pi / d)
    dt = np.pi / m
    dp = a / dt
    for i in range(int(m)):
        t = np.pi * (i + 0.5) / m
        mp = np.round(2 * np.pi * np.sin(t) / dp)
        for j in range(int(mp)):
            p = 2 * np.pi * j / mp
            if j == 0 and i == 0:
                r = [np.array([Re * np.sin(t) * np.cos(p), Re * np.sin(t) * np.sin(p), Re * np.cos(t)])]
            else:
                r = np.append(r, [np.array([Re * np.sin(t) * np.cos(p), Re * np.sin(t) * np.sin(p), Re * np.cos(t)])],
                              axis=0)
    return r


def ss_ecef_lla_no_use(pos):
    x = pos[0]
    y = pos[1]
    z = pos[2]
    b = np.sqrt(Re ** 2 * (1 - e_ecc ** 2))
    ep = np.sqrt((Re ** 2 - b ** 2) / b ** 2)
    p = np.sqrt(x ** 2 + y ** 2)
    th = np.arctan2(Re * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan((z + ep ** 2 * b * (np.sin(th)) ** 3) / (p - e_ecc ** 2 * Re * (np.cos(th)) ** 3))
    n = Re / np.sqrt(1 - e_ecc ** 2 * (np.sin(lat)) ** 2)
    alt = p / np.cos(lat) - n
    k = np.abs(x) < 1 & abs(y) < 1
    if k:
        alt = np.abs(z) - b
    GPS = [lat, lon, alt]


def ss_ecef_lla(pos):
    x = pos[0]
    y = pos[1]
    z = pos[2]
    a = Re
    a_sq = a ** 2
    e_sq = e_ecc ** 2
    b = a * (1 - e_f)
    # calculations:
    r = np.sqrt(x ** 2 + y ** 2)
    ep_sq = (a ** 2 - b ** 2) / b ** 2
    ee = (a ** 2 - b ** 2)
    f = (54 * b ** 2) * (z ** 2)
    g = r ** 2 + (1 - e_sq) * (z ** 2) - e_sq * ee * 2
    c = (e_sq ** 2) * f * r ** 2 / (g ** 3)
    s = (1 + c + np.sqrt(c ** 2 + 2 * c)) ** (1 / 3.)
    p = f / (3. * (g ** 2) * (s + (1. / s) + 1) ** 2)
    q = np.sqrt(1 + 2 * p * e_sq ** 2)
    r_0 = -(p * e_sq * r) / (1 + q) + np.sqrt(
        0.5 * (a ** 2) * (1 + (1. / q)) - p * (z ** 2) * (1 - e_sq) / (q * (1 + q)) - 0.5 * p * (r ** 2))
    u = np.sqrt((r - e_sq * r_0) ** 2 + z ** 2)
    v = np.sqrt((r - e_sq * r_0) ** 2 + (1 - e_sq) * z ** 2)
    z_0 = (b ** 2) * z / (a * v)
    h = u * (1 - b ** 2 / (a * v))
    phi = np.arctan((z + ep_sq * z_0) / r)
    lambd = np.arctan2(y, x)
    return np.array([phi, lambd, h])


def ss_ecef_aer(station, sat):
    r = sat - station
    station_lla = ss_ecef_lla(station)
    t = np.cos(station_lla[1]) * r[0] + np.sin(station_lla[1]) * r[1]
    e = -np.sin(station_lla[1]) * r[0] + np.cos(station_lla[1]) * r[1]
    u = np.cos(station_lla[0]) * t + np.sin(station_lla[0]) * r[2]
    n = -np.sin(station_lla[0]) * t + np.cos(station_lla[0]) * r[2]
    try:
        e[abs(e) < 1e-3] = 0.0
        n[abs(n) < 1e-3] = 0.0
        u[abs(u) < 1e-3] = 0.0
    except TypeError:
        if abs(e) < 1e-3:
            e = 0.0  # type: ignore
        if abs(n) < 1e-3:
            n = 0.0  # type: ignore
        if abs(u) < 1e-3:
            u = 0.0  # type: ignore 
    enu = np.sqrt(e ** 2 + n ** 2 + u ** 2)
    slantRange = np.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
    elev = np.arcsin(u / enu)
    az = np.arctan2(e / enu, n / enu)
    return np.array([az, elev, slantRange])


def ss_single_user_access(pos, sats, el_mask):  # input pos in ecef coordinate, sats in [n*12] type, elmask in degrees
    eci = ss_coe2eci_multi(sats)
    ecef = ss_eci2ecef_multi(eci)
    no_access = 0
    for i in range(np.size(ecef, 0)):
        aer = []
        aer = ss_ecef_aer(pos, ecef[i][6:9])
        if aer[1] * 180 / np.pi >= el_mask:
            no_access += 1
        else:
            pass
    return no_access


def ss_multi_user_access(grid, sats, el_mask):
    no_access = np.zeros([1, np.size(grid, 0)])
    for i in range(np.size(grid, 0)):
        no_access[0][i] = ss_single_user_access(grid[i][:], sats, el_mask)
    return no_access


def ss_get_dops_single(site,sats,elmask):
    lat,lon,alt = ss_ecef_lla(site)
    ecef2enu = np.array([[-np.sin(lon),np.cos(lon),0]\
                         ,[-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)]\
                            ,[np.cos(lat)*np.cos(lon),  np.cos(lat)*np.sin(lon), np.sin(lat)]])
    eci = ss_coe2eci_multi(sats)
    ecef = ss_eci2ecef_multi(eci)
    no_sv = 0
    for i in range(np.size(ecef, 0)):
        aer = []
        aer = ss_ecef_aer(site, ecef[i,6:9])
        if aer[1] * 180 / np.pi >= elmask:
            no_sv += 1
            los_norm = ( (ecef[i,6] - site[0]) ** 2 +\
                      (ecef[i,7] - site[1]) ** 2 +\
                          (ecef[i,8] - site[2]) ** 2 ) ** 0.5
            try:
                los = np.append(los,[ecef[i,6:9] - site] / los_norm,axis=0)
            except: 
                los = [ecef[i,6:9] - site] / los_norm
        else:
            pass
    
    if no_sv >= 4:
        g_mat = np.append(los,np.ones([no_sv,1]),axis=1)
        a = np.append(ecef2enu,[[0],[0],[0]],axis=1)
        r_tilde = np.append( np.append(ecef2enu,[[0],[0],[0]],axis=1),\
                            [[0,0,0,1]],axis=0 )
        h_mat = np.linalg.pinv(np.matmul(np.matmul(r_tilde,np.matmul(np.transpose(g_mat),g_mat)),np.transpose(r_tilde)))
        dops = np.diag(h_mat)
    else:
        dops = [25e4,25e4,25e4,25e4]
    gdop = np.sqrt(np.sum(dops))
    pdop = np.sqrt(np.sum(dops[0:3]))
    tdop = np.sqrt(np.sum(dops[3]))
    hdop = np.sqrt(np.sum(dops[0:2]))
    vdop = np.sqrt(np.sum(dops[2]))
    return gdop, pdop, tdop, hdop, vdop


def ss_get_dops_multi(grid, sats, el_mask):
    dops = np.zeros([np.size(grid, 0),5])
    for i in range(np.size(grid, 0)):
        dops[i,0:5] = ss_get_dops_single(grid[i,:], sats, el_mask)
    return dops