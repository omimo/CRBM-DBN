# coding: utf-8
import crbm as CO
import pickle
import scipy

crbm, batchdata = CO.train_crbm()

# Just dumping the model for further use in Python
with open('crbmconfig_sean_100h_3p.pkl', 'wb') as output:
    pickle.dump(crbm, output, pickle.HIGHEST_PROTOCOL)
    

#scipy.io.savemat('crbmconfig_sean_100h_3p.mat',{'A':crbm.A, 'B':crbm.B, 'W':crbm.W, 'hbias':crbm.hbias,'vbias':crbm.vbias})
scipy.io.savemat('crbmconfig_sean_100h_3p.mat',{'A':crbm.A.get_value(), 'B':crbm.B.get_value(), 'W':crbm.W.get_value(), 'hbias':crbm.hbias.get_value(),'vbias':crbm.vbias.get_value()})
