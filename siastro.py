
from ss_astro_lib import *

class satellite:
    def __init__(self,name = 'satellite',\
                 epoch = [2023,1,11,7,30,00.00],\
                    eph_type = 'coe',\
                        eph_vec = [7078,0,60,0,0,0]):
        self.name = name
        self.epoch = epoch
        if eph_type == 'coe':
            self.coe = eph_vec
            self.eci = ss_coe2eci(self.coe)
            self.eci2ecef_dcm = ss_dcm_eci_ecef(self.epoch)
            self.ecef =  np.append(np.matmul( self.eci2ecef_dcm , np.transpose(self.eci[0:3])),\
                                   np.matmul( self.eci2ecef_dcm , np.transpose(self.eci[3:6])))
            self.lla = ss_ecef_lla(self.ecef)

        
