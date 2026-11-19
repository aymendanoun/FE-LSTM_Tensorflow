#derive de ConstitutiveLaw
#This law should be used with an InternalForce WeakForm

from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw

#from fedoo.libUtil.StrainOperator import *
#from fedoo.libUtil.ModelingSpace  import GetDimension
from fedoo.libUtil.PostTreatement import listStressTensor, listStrainTensor

import numpy as np
import copy 
from simcoon import simmit as sim

class LSTM_DenseMaterial(ConstitutiveLaw):
    def __init__(self, ID=""):
        #props is a nparray containing all the material variables
        #nstatev is a nparray containing all the material variables
        ConstitutiveLaw.__init__(self, ID) # heritage
    
        
        # self.__InitialStatev = statev #statev may be an int or an array        
        
        self.__currentGradDisp = self.__initialGradDisp = 0        
        self.__currentStress = None

        # Variable("DispX")
        # Variable("DispY")
        """
        if GetDimension() == "3D":
            # Variable("DispZ")
            ndi = 3 ; nshr = 3
        elif GetDimension() in ['2Dstress']:
            # ndi = 2 ; nshr = 1
            ndi = 3 ; nshr = 3###shoud be modified?
        elif GetDimension() in ['2Dplane']:
             ndi = 3 ; nshr = 3 # the constitutive law is treated in a classical way
        """
        #statev = ??? require to get te number of gauss point
        
        ### initialization of the simcoon UMAT
        # sim.Umat_fedoo.__init__(self, umat_name, np.atleast_2d(props), corate, ndi, nshr)
        # sim.Umat_fedoo.__init__(self, umat_name, np.atleast_2d(props), statev, corate, ndi, nshr, 0.)
            
    def GetPKII(self):
        return listStressTensor(self.__currentStress.T)
    
    def GetKirchhoff(self):
        return listStressTensor(self.Kirchhoff.T)        
    
    def GetCauchy(self):
        return listStressTensor(self.Cauchy.T)        
    
    def GetStrain(self, **kargs):
        #print(len(listStrainTensor(self.__currentStrain.T)[0]))
        return listStrainTensor(self.__currentStrain.T)
           
    def GetStatev(self):
        return self.statev.T
    
    
    
    def GetCurrentStress(self):
        #print(len(listStressTensor(self.__currentStress.T)))
        return listStressTensor(self.__currentStress.T)

    def GetStress(self, **kargs): #same as GetPKII (used for small def)
        #print(len(listStressTensor(self.__currentStress.T)[0]))
        return listStressTensor(self.__currentStress.T)

    # def GetHelas (self):
    #     # if self.__L is None:                
    #     #     self.RunUmat(np.eye(3).T.reshape(1,3,3), np.eye(3).T.reshape(1,3,3), time=0., dtime=1.)

    #     return np.squeeze(self.L.transpose(1,2,0)) 
    
    # def GetCurrentGradDisp(self):
    #     if self.__currentGradDisp is 0: return 0
    #     else: return self.__currentGradDisp
        
    def GetH(self):
        # if self.__Lt is None:
        #     if self.__L is None:                
        #         self.RunUmat(np.eye(3).T.reshape(1,3,3), np.eye(3).T.reshape(1,3,3), time=0., dtime=1.)
        #         self.__Statev = None
        #         self.__currentStress = None

        #     return np.squeeze(self.__L.transpose(1,2,0))            
        # else: 
        #     return np.squeeze(self.__Lt.transpose(1,2,0))
        #print(np.squeeze(self.Lt.transpose(1,2,0))[:,:,0])             
        return np.squeeze(self.Lt.transpose(1,2,0))
                    
    def __ChangeBasisH(self, H):       
        #Change of basis capability for laws on the form : StressTensor = H * StrainTensor
        #StressTensor and StrainTensor are column vectors based on the voigt notation 
        if self._ConstitutiveLaw__localFrame is not None:
            localFrame = self._ConstitutiveLaw__localFrame
            #building the matrix to change the basis of the stress and the strain
#            theta = np.pi/8
#            np.array([[np.cos(theta),np.sin(theta),0], [-np.sin(theta),np.cos(theta),0], [0,0,1]]) 
            R_epsilon = np.empty((len(localFrame), 6,6))
            R_epsilon[:,  :3,  :3] = localFrame**2 
            R_epsilon[:,  :3, 3:6] = localFrame[:,:,[0,2,1]]*localFrame[:,:,[1,0,2]]
            R_epsilon[:, 3:6,  :3] = 2*localFrame[:,[0,2,1]]*localFrame[:,[1,0,2]] 
            R_epsilon[:, 3:6, 3:6] = localFrame[:,[[0],[2],[1]], [0,2,1]]*localFrame[:,[[1],[0],[2]],[1,0,2]] + localFrame[:,[[1],[0],[2]],[0,2,1]]*localFrame[:,[[0],[2],[1]],[1,0,2]] 
            R_sigma_inv = R_epsilon.transpose(0,2,1)    # np.transpose(R_epsilon,[0,2,1])        
            
            if len(H.shape) == 3: H = np.rollaxis(H,2,0)
            H = np.matmul(R_sigma_inv, np.matmul(H,R_epsilon))
            if len(H.shape) == 3: H = np.rollaxis(H,0,3)  
        
        print(H[:,:,0])
        return H
    
    def GetStressOperator(self, localFrame=None):
        H = self.__ChangeBasisH(self.GetH())
                      
        eps, eps_vir = GetStrainOperator()
        sigma = [sum([0 if eps[j] is 0 else eps[j]*H[i][j] for j in range(6)]) for i in range(6)]

        return sigma # list de 6 objets de type OpDiff
    
    def NewTimeIncrement(self):
        self.memory_start = copy.deepcopy(self.memory)
        # self.set_start() #in set_start -> set tangeant matrix to elastic
        
        # #save variable at the begining of the Time increment
        # self.__initialGradDisp = self.__currentGradDisp

        
    def ResetTimeIncrement(self):
        self.memory = copy.deepcopy(self.memory_start)
    
    def Reset(self):
        """
        Reset the constitutive law (time history)
        """
        self.memory = []
        for i in range(2): #lstm layer            
            self.memory.append(np.zeros(nb_points, units))
        
        self.memory_start = copy.deepcopy(self.memory)
    
    def __sigmoid(self,x):  
            return 1 / (1 + np.exp(-x))
    
    
    def __relu(self,x):  
            return np.maximum(0,x)
            
    def __transform(self, X_input, minX, maxX):
        return (X_input-minX)/(maxX-minX)

    def __inv_transform(self, Y, minX, maxX):
        return Y*(maxX-minX) + minX

    def Initialize(self, assembly, pb, initialTime = 0., nlgeom=True):

        #if the number of material points is not defined (=0) we need to initialize statev
        nb_points = assembly.GetNumberOfGaussPoints() * assembly.GetMesh().GetNumberOfElements()
        self.nb_points = nb_points
        self.PKII = np.zeros((self.nb_points, 6))


        """
        self.Lstm_layer1  = np.load('./weights_scaler_3D_V1/weights_scaler_stress/LSTM_layer_1.npz')
        self.Lstm_layer2  = np.load('./weights_scaler_3D_V1/weights_scaler_stress/LSTM_layer_2.npz')
        self.Dense_layer1  = np.load('./weights_scaler_3D_V1/weights_scaler_stress/Dense_layer_3.npz')
        self.Dense_layer2  = np.load('./weights_scaler_3D_V1/weights_scaler_stress/Dense_layer_4.npz')
        self.scaling_stress = np.load('./weights_scaler_3D_V1/weights_scaler_stress/scaling_stress.npz')
        
        self.Lstm_layer1_Lt  = np.load('./weights_scaler_3D_V1/weights_scaler_Lt/LSTM_layer_1.npz')
        self.Lstm_layer2_Lt  = np.load('./weights_scaler_3D_V1/weights_scaler_Lt/LSTM_layer_2.npz')
        self.Dense_layer1_Lt  = np.load('./weights_scaler_3D_V1/weights_scaler_Lt/Dense_layer_3.npz')
        self.Dense_layer2_Lt  = np.load('./weights_scaler_3D_V1/weights_scaler_Lt/Dense_layer_4.npz')
        self.scaling_Lt = np.load('./weights_scaler_3D_V1/weights_scaler_Lt/scaling_Lt.npz')        
        
        """
        self.Lstm_layer1  = np.load('./weights_scaler_3D_DenseMaterial/weights_scaler_stress/LSTM_layer_1.npz')
        self.Lstm_layer2  = np.load('./weights_scaler_3D_DenseMaterial/weights_scaler_stress/LSTM_layer_2.npz')
        self.Dense_layer1  = np.load('./weights_scaler_3D_DenseMaterial/weights_scaler_stress/Dense_layer_3.npz')
        self.Dense_layer2  = np.load('./weights_scaler_3D_DenseMaterial/weights_scaler_stress/Dense_layer_4.npz')
        self.scaling_stress = np.load('./weights_scaler_3D_DenseMaterial/weights_scaler_stress/scaling_stress.npz')
        
        self.Lstm_layer1_Lt  = np.load('./weights_scaler_3D_DenseMaterial/weights_scaler_Lt/LSTM_layer_1.npz')
        self.Lstm_layer2_Lt  = np.load('./weights_scaler_3D_DenseMaterial/weights_scaler_Lt/LSTM_layer_2.npz')
        self.Dense_layer1_Lt  = np.load('./weights_scaler_3D_DenseMaterial/weights_scaler_Lt/Dense_layer_3.npz')
        self.Dense_layer2_Lt  = np.load('./weights_scaler_3D_DenseMaterial/weights_scaler_Lt/Dense_layer_4.npz')
        self.scaling_Lt = np.load('./weights_scaler_3D_DenseMaterial/weights_scaler_Lt/scaling_Lt.npz')
        
        
        
        
        
        #units = 100
        units = self.Lstm_layer1['W_i'].shape[1] #NN model parameter -> size of memory AND size of lstm_layer output 
        self.units = units
        
        self.memory = [] ; self.output = []
        for i in range(2): #lstm layer            
            self.memory.append(np.zeros((nb_points, units)))
            self.output.append(np.zeros((nb_points, units))) 
            
        for i in range(2,4): #lstm layer for Lt          
            self.memory.append(np.zeros((nb_points, units)))
            self.output.append(np.zeros((nb_points, units))) 
        
        self.memory_start = copy.deepcopy(self.memory)
        self.__currentStrain = np.zeros((nb_points, 6))
        self.__currentStress = np.zeros((nb_points, 6))
        self.Lt = np.zeros((nb_points, 6,6))
        
        # if np.isscalar(self.__InitialStatev): 
        #     statev = np.zeros((nb_points, int(self.__InitialStatev)))
        # else: 
        #     statev = np.atleast_2d(self.__InitialStatev).astype(float)
        #     if len(statev) == 1: statev = np.tile(statev.copy(),[nb_points,1])
        #     else: statev = assembly.ConvertData(statev)    

        # sim.Umat_fedoo.Initialize(self, initialTime, statev, nlgeom)
        self.Run(0.) #Launch the UMAT to compute the elastic 
        
    def Run(self,Dtime):
        X_input =  self.__currentStrain[:,[0,1,2,3,4,5]]
        X = self.__transform (X_input, self.scaling_stress['scalingX_min'], self.scaling_stress['scalingX_max'])
        
        #launch lstm layer
        h1 = self.__lstm_layer(X, 0, self.Lstm_layer1) ; X = h1
        h2 = self.__lstm_layer(X, 1, self.Lstm_layer2) ; X = h2
        
        #extract model parameters (biais and weights) for the 2 dense layers
        W_1 = self.Dense_layer1['W']
        b_1 = self.Dense_layer1['b']
        W_2 = self.Dense_layer2['W']
        b_2 = self.Dense_layer2['b']
        
        X = np.maximum(np.dot(X, W_1) + b_1,0)
        Y = np.dot(X, W_2) + b_2                
        
        Y = self.__inv_transform (Y, self.scaling_stress['scalingY_min'], self.scaling_stress['scalingY_max'])
        
        self.__currentStress[:,0:6] = Y[:,0:6]
        
        
        # Get Lt
        X = self.__transform (X_input, self.scaling_Lt['scalingX_min'], self.scaling_Lt['scalingX_max'])
        
        #launch lstm layer
        h1 = self.__lstm_layer(X, 2, self.Lstm_layer1_Lt) ; X = h1
        h2 = self.__lstm_layer(X, 3, self.Lstm_layer2_Lt) ; X = h2
        
        #extract model parameters (biais and weights) for the 2 dense layers
        W_1 = self.Dense_layer1_Lt['W']
        b_1 = self.Dense_layer1_Lt['b']
        W_2 = self.Dense_layer2_Lt['W']
        b_2 = self.Dense_layer2_Lt['b']
        
        X = np.maximum(np.dot(X, W_1) + b_1,0)
        Y = np.dot(X, W_2) + b_2                
        
        Y = self.__inv_transform (Y, self.scaling_Lt['scalingY_min'], self.scaling_Lt['scalingY_max'])
        
        for i in range(6):
            self.Lt[:,i,i] = Y[:,i]
        
        
        self.Lt[:,0,1] = Y[:,6]
        self.Lt[:,1,0] = Y[:,6]
        self.Lt[:,0,2] = Y[:,7]
        self.Lt[:,2,0] = Y[:,7]
        self.Lt[:,0,3] = Y[:,8]
        self.Lt[:,3,0] = Y[:,8]        
        self.Lt[:,0,4] = Y[:,9]
        self.Lt[:,4,0] = Y[:,9]
        self.Lt[:,0,5] = Y[:,10]
        self.Lt[:,5,0] = Y[:,10]
        
        
        self.Lt[:,1,2] = Y[:,11]
        self.Lt[:,2,1] = Y[:,11]
        self.Lt[:,1,3] = Y[:,12]
        self.Lt[:,3,1] = Y[:,12]
        self.Lt[:,1,4] = Y[:,13]
        self.Lt[:,4,1] = Y[:,13]        
        self.Lt[:,1,5] = Y[:,14]
        self.Lt[:,5,1] = Y[:,14]
     

        self.Lt[:,2,3] = Y[:,15]
        self.Lt[:,3,2] = Y[:,15]
        self.Lt[:,2,4] = Y[:,16]
        self.Lt[:,4,2] = Y[:,16]
        self.Lt[:,2,5] = Y[:,17]
        self.Lt[:,5,2] = Y[:,17]         
        
        
        self.Lt[:,3,4] = Y[:,18]
        self.Lt[:,4,3] = Y[:,18]
        self.Lt[:,3,5] = Y[:,19]
        self.Lt[:,5,3] = Y[:,19]        
        
        self.Lt[:,4,5] = Y[:,20]
        self.Lt[:,5,4] = Y[:,20]         
        
        """"   
        self.Lt[:,0,1] = Y[:,6]
        self.Lt[:,1,0] = Y[:,6]
        self.Lt[:,0,2] = Y[:,7]
        self.Lt[:,2,0] = Y[:,7]
        self.Lt[:,1,2] = Y[:,8]
        self.Lt[:,2,1] = Y[:,8]        
        self.Lt[:,0,3] = Y[:,9]
        self.Lt[:,3,0] = Y[:,9]
        self.Lt[:,1,3] = Y[:,10]
        self.Lt[:,3,1] = Y[:,10]
        self.Lt[:,2,3] = Y[:,11]
        self.Lt[:,3,2] = Y[:,11] 
        """
            
#        self.Lt[:,0:4,0:4] = Y[:,:16].reshape(self.nb_points,4,4)
#        self.Lt[:,4,4] = Y[:,16]
#        self.Lt[:,5,5] = Y[:,17]
        
    def __lstm_layer(self,X, i, lstm_layer):
        # if i == 1: lstm_layer = self.Lstm_layer1
        # elif i == 2: lstm_layer = self.Lstm_layer2
        
        #extract layer parameters (biais and weights) for each gates of the 2 lstm layers
        W_i = lstm_layer['W_i'] ; W_f = lstm_layer['W_f'] #W_i.shape = (dim_strain, units)
        W_c = lstm_layer['W_c'] ; W_o = lstm_layer['W_o']
        U_i = lstm_layer['U_i'] ; U_f = lstm_layer['U_f']
        U_c = lstm_layer['U_c'] ; U_o = lstm_layer['U_o']
        b_i = lstm_layer['b_i'] ; b_f = lstm_layer['b_f']
        b_c = lstm_layer['b_c'] ; b_o = lstm_layer['b_o']
        
        xi = np.dot(X,W_i) + b_i #[1,m] for all
        xf = np.dot(X,W_f) + b_f
        xc = np.dot(X,W_c) + b_c
        xo = np.dot(X,W_o) + b_o
        
        hprev = self.output[i] #output of the lstm layer at last iteration
                
        i_t = self.__sigmoid(xi + np.dot(hprev, U_i)) #[1,m] + [m]*[m,m] -> [1,m]
        f_t = self.__sigmoid(xf + np.dot(hprev, U_f)) #[1,m] + [m]*[m,m] -> [1,m]
        o_t = self.__sigmoid(xo + np.dot(hprev, U_o)) #[1,m] + [m]*[m,m] -> [1,m]
        c_t = f_t*self.memory[i] + i_t * np.tanh(xc + np.dot(hprev, U_c)) #[1,m]*[m] + [1,m] * [1,m] -> [1,m]
        h_t = o_t * np.tanh(c_t) #[1,m]*[1,m] (elementwise)
        self.memory[i] = c_t
        self.output[i] = h_t
        
        return h_t
       


    def Update(self,assembly, pb, dtime, nlgeom=True):            
        displacement = pb.GetDoFSolution()
            
        if displacement is 0: 
            self.__currentStrain = 0
            self.__currentStress = None
        else:                    
            self.__currentStrain = np.array(assembly.GetStrainTensor(displacement, "GaussPoint")).T         
                
        #tranpose for comatibility with simcoon
        # if displacement is 0: 
        #     self.__currentGradDisp = 0
        #     self.__currentStress = None
        #     F1 = np.multiply(np.eye(3).reshape(3,3,1) , np.ones((1,1,self.nb_points)), order='F').transpose(2,0,1)
        #     # self.__F1 = np.eye(3).T.reshape(1,3,3)
        # else:                    
        #     self.__currentGradDisp = np.array(assembly.GetGradTensor(displacement, "GaussPoint"))            
        #     #F0.strides and F1.strides should be [n_cols*n_rows*8, 8, n_rows*8] for compatibiliy with the sim.RunUmat_fedoo function
        #     F1 = np.add( np.eye(3).reshape(3,3,1), self.__currentGradDisp, order='F').transpose(2,0,1)                        

        # self.compute_Detot(dtime, F1)   
        self.Run(dtime)
        # (DRloc , listDR, Detot, statev) = self.Run(dtime)
        
        
    # def GetStress(self, GradDispTensor, time = None):
    #     self.__GradDispTenorOld = self.__GradDispTensor
    #     self.__GradDispTensor = GradDispTensor
        
        
    #     # time not used here because this law require no time effect
    #     # initilialize values plasticity variables if required

    #     self.__currentStress = listStressTensor(sigmaFull.T) # list of 6 objets

        








####################################################################################################################################




















"""
#derive de ConstitutiveLaw
#This law should be used with an InternalForce WeakForm

from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw

from fedoo.libUtil.StrainOperator import *
from fedoo.libUtil.ModelingSpace  import Variable, GetDimension
from fedoo.libUtil.PostTreatement import listStressTensor, listStrainTensor

import numpy as np
import copy 
from simcoon import simmit as sim

class LSTM(ConstitutiveLaw):
    def __init__(self, ID=""):
        #props is a nparray containing all the material variables
        #nstatev is a nparray containing all the material variables
        ConstitutiveLaw.__init__(self, ID) # heritage
    
        
        # self.__InitialStatev = statev #statev may be an int or an array        
        
        self.__currentGradDisp = self.__initialGradDisp = 0        
        self.__currentStress = None

        # Variable("DispX")
        # Variable("DispY")
    
        if GetDimension() == "3D":
            # Variable("DispZ")
            ndi = 3 ; nshr = 3
        elif GetDimension() in ['2Dstress']:
            # ndi = 2 ; nshr = 1
            ndi = 3 ; nshr = 3###shoud be modified?
        elif GetDimension() in ['2Dplane']:
             ndi = 3 ; nshr = 3 # the constitutive law is treated in a classical way

        #statev = ??? require to get te number of gauss point
        
        ### initialization of the simcoon UMAT
        # sim.Umat_fedoo.__init__(self, umat_name, np.atleast_2d(props), corate, ndi, nshr)
        # sim.Umat_fedoo.__init__(self, umat_name, np.atleast_2d(props), statev, corate, ndi, nshr, 0.)
            
    def GetPKII(self):
        return listStressTensor(self.__currentStress.T)
    
    def GetKirchhoff(self):
        return listStressTensor(self.Kirchhoff.T)        
    
    def GetCauchy(self):
        return listStressTensor(self.Cauchy.T)        
    
    def GetStrain(self, **kargs):
        #print(len(listStrainTensor(self.__currentStrain.T)[0]))
        return listStrainTensor(self.__currentStrain.T)
           
    def GetStatev(self):
        return self.statev.T
    
    
    
    def GetCurrentStress(self):
        #print(len(listStressTensor(self.__currentStress.T)))
        return listStressTensor(self.__currentStress.T)

    def GetStress(self, **kargs): #same as GetPKII (used for small def)
        #print(len(listStressTensor(self.__currentStress.T)[0]))
        return listStressTensor(self.__currentStress.T)

    # def GetHelas (self):
    #     # if self.__L is None:                
    #     #     self.RunUmat(np.eye(3).T.reshape(1,3,3), np.eye(3).T.reshape(1,3,3), time=0., dtime=1.)

    #     return np.squeeze(self.L.transpose(1,2,0)) 
    
    # def GetCurrentGradDisp(self):
    #     if self.__currentGradDisp is 0: return 0
    #     else: return self.__currentGradDisp
        
    def GetH(self):
        # if self.__Lt is None:
        #     if self.__L is None:                
        #         self.RunUmat(np.eye(3).T.reshape(1,3,3), np.eye(3).T.reshape(1,3,3), time=0., dtime=1.)
        #         self.__Statev = None
        #         self.__currentStress = None

        #     return np.squeeze(self.__L.transpose(1,2,0))            
        # else: 
        #     return np.squeeze(self.__Lt.transpose(1,2,0))
        #print(np.squeeze(self.Lt.transpose(1,2,0))[:,:,0])             
        return np.squeeze(self.Lt.transpose(1,2,0))
                    
    def __ChangeBasisH(self, H):       
        #Change of basis capability for laws on the form : StressTensor = H * StrainTensor
        #StressTensor and StrainTensor are column vectors based on the voigt notation 
        if self._ConstitutiveLaw__localFrame is not None:
            localFrame = self._ConstitutiveLaw__localFrame
            #building the matrix to change the basis of the stress and the strain
#            theta = np.pi/8
#            np.array([[np.cos(theta),np.sin(theta),0], [-np.sin(theta),np.cos(theta),0], [0,0,1]]) 
            R_epsilon = np.empty((len(localFrame), 6,6))
            R_epsilon[:,  :3,  :3] = localFrame**2 
            R_epsilon[:,  :3, 3:6] = localFrame[:,:,[0,2,1]]*localFrame[:,:,[1,0,2]]
            R_epsilon[:, 3:6,  :3] = 2*localFrame[:,[0,2,1]]*localFrame[:,[1,0,2]] 
            R_epsilon[:, 3:6, 3:6] = localFrame[:,[[0],[2],[1]], [0,2,1]]*localFrame[:,[[1],[0],[2]],[1,0,2]] + localFrame[:,[[1],[0],[2]],[0,2,1]]*localFrame[:,[[0],[2],[1]],[1,0,2]] 
            R_sigma_inv = R_epsilon.transpose(0,2,1)    # np.transpose(R_epsilon,[0,2,1])        
            
            if len(H.shape) == 3: H = np.rollaxis(H,2,0)
            H = np.matmul(R_sigma_inv, np.matmul(H,R_epsilon))
            if len(H.shape) == 3: H = np.rollaxis(H,0,3)  
        
        print(H[:,:,0])
        return H
    
    def GetStressOperator(self, localFrame=None):
        H = self.__ChangeBasisH(self.GetH())
                      
        eps, eps_vir = GetStrainOperator()
        sigma = [sum([0 if eps[j] is 0 else eps[j]*H[i][j] for j in range(6)]) for i in range(6)]

        return sigma # list de 6 objets de type OpDiff
    
    def NewTimeIncrement(self):
        self.memory_start = copy.deepcopy(self.memory)
        # self.set_start() #in set_start -> set tangeant matrix to elastic
        
        # #save variable at the begining of the Time increment
        # self.__initialGradDisp = self.__currentGradDisp

        
    def ResetTimeIncrement(self):
        self.memory = copy.deepcopy(self.memory_start)
    
    def Reset(self):

        self.memory = []
        for i in range(2): #lstm layer            
            self.memory.append(np.zeros(nb_points, units))
        
        self.memory_start = copy.deepcopy(self.memory)

    def __sigmoid(self,x):  
            return 1 / (1 + np.exp(-x))
            
    def __transform(self, X_input, mean, scale):
        return (X_input-mean)/scale

    def __inv_transform(self, Y, mean, scale):
        return Y*scale+mean

    def Initialize(self, assembly, pb, initialTime = 0., nlgeom=True):

        #if the number of material points is not defined (=0) we need to initialize statev
        nb_points = assembly.GetNumberOfGaussPoints() * assembly.GetMesh().GetNumberOfElements()
        self.nb_points = nb_points
        self.PKII = np.zeros((self.nb_points, 6))
        
        self.Lstm_layer1  = np.load('./NN_data/EPICP/Lstm_layer1.npz')
        self.Lstm_layer2  = np.load('./NN_data/EPICP/Lstm_layer2.npz')
        self.Dense_layer1  = np.load('./NN_data/EPICP/Dense_layer3.npz')
        self.Dense_layer2  = np.load('./NN_data/EPICP/Dense_layer4.npz')
        self.scaling_stress = np.load('./NN_data/EPICP/scaling_stress.npz')
        
        self.Lstm_layer1_Lt  = np.load('./NN_data/EPICP/Lstm_layer1_Lt.npz')
        self.Lstm_layer2_Lt  = np.load('./NN_data/EPICP/Lstm_layer2_Lt.npz')
        self.Dense_layer1_Lt  = np.load('./NN_data/EPICP/Dense_layer3_Lt.npz')
        self.Dense_layer2_Lt  = np.load('./NN_data/EPICP/Dense_layer4_Lt.npz')
        self.scaling_Lt = np.load('./NN_data/EPICP/scaling_Lt.npz')
        
        #units = 100
        units = self.Lstm_layer1['W_i'].shape[1] #NN model parameter -> size of memory AND size of lstm_layer output 
        self.units = units
        
        self.memory = [] ; self.output = []
        for i in range(2): #lstm layer            
            self.memory.append(np.zeros((nb_points, units)))
            self.output.append(np.zeros((nb_points, units))) 
            
        for i in range(2,4): #lstm layer for Lt          
            self.memory.append(np.zeros((nb_points, units)))
            self.output.append(np.zeros((nb_points, units))) 
        
        self.memory_start = copy.deepcopy(self.memory)
        self.__currentStrain = np.zeros((nb_points, 6))
        self.__currentStress = np.zeros((nb_points, 6))
        self.Lt = np.zeros((nb_points, 6,6))
        
        # if np.isscalar(self.__InitialStatev): 
        #     statev = np.zeros((nb_points, int(self.__InitialStatev)))
        # else: 
        #     statev = np.atleast_2d(self.__InitialStatev).astype(float)
        #     if len(statev) == 1: statev = np.tile(statev.copy(),[nb_points,1])
        #     else: statev = assembly.ConvertData(statev)    

        # sim.Umat_fedoo.Initialize(self, initialTime, statev, nlgeom)
        self.Run(0.) #Launch the UMAT to compute the elastic 
        
    def Run(self,Dtime):
        X_input =  self.__currentStrain[:,[0,1,3]]
        X = self.__transform (X_input, self.scaling_stress['scalingX_mean'], self.scaling_stress['scalingX_std'])
        
        #launch lstm layer
        h1 = self.__lstm_layer(X, 0, self.Lstm_layer1) ; X = h1
        h2 = self.__lstm_layer(X, 1, self.Lstm_layer2) ; X = h2
        
        #extract model parameters (biais and weights) for the 2 dense layers
        W_1 = self.Dense_layer1['W']
        b_1 = self.Dense_layer1['b']
        W_2 = self.Dense_layer2['W']
        b_2 = self.Dense_layer2['b']
        
        X = np.tanh(np.dot(X, W_1) + b_1)
        Y = np.dot(X, W_2) + b_2                
        
        Y = self.__inv_transform (Y, self.scaling_stress['scalingY_mean'], self.scaling_stress['scalingY_std'])
        
        self.__currentStress[:,0:4] = Y[:,0:4]
        
        
        # Get Lt
        X = self.__transform (X_input, self.scaling_Lt['scalingX_mean'], self.scaling_Lt['scalingX_std'])
        
        #launch lstm layer
        h1 = self.__lstm_layer(X, 2, self.Lstm_layer1_Lt) ; X = h1
        h2 = self.__lstm_layer(X, 3, self.Lstm_layer2_Lt) ; X = h2
        
        #extract model parameters (biais and weights) for the 2 dense layers
        W_1 = self.Dense_layer1_Lt['W']
        b_1 = self.Dense_layer1_Lt['b']
        W_2 = self.Dense_layer2_Lt['W']
        b_2 = self.Dense_layer2_Lt['b']
        
        X = np.tanh(np.dot(X, W_1) + b_1)
        Y = np.dot(X, W_2) + b_2                
        
        Y = self.__inv_transform (Y, self.scaling_Lt['scalingY_mean'], self.scaling_Lt['scalingY_std'])
        
        self.Lt[:,0:4,0:4] = Y[:,:16].reshape(self.nb_points,4,4)
        
        
    def __lstm_layer(self,X, i, lstm_layer):
        # if i == 1: lstm_layer = self.Lstm_layer1
        # elif i == 2: lstm_layer = self.Lstm_layer2
        
        #extract layer parameters (biais and weights) for each gates of the 2 lstm layers
        W_i = lstm_layer['W_i'] ; W_f = lstm_layer['W_f'] #W_i.shape = (dim_strain, units)
        W_c = lstm_layer['W_c'] ; W_o = lstm_layer['W_o']
        U_i = lstm_layer['U_i'] ; U_f = lstm_layer['U_f']
        U_c = lstm_layer['U_c'] ; U_o = lstm_layer['U_o']
        b_i = lstm_layer['b_i'] ; b_f = lstm_layer['b_f']
        b_c = lstm_layer['b_c'] ; b_o = lstm_layer['b_o']
        
        xi = np.dot(X,W_i) + b_i #[1,m] for all
        xf = np.dot(X,W_f) + b_f
        xc = np.dot(X,W_c) + b_c
        xo = np.dot(X,W_o) + b_o
        
        hprev = self.output[i] #output of the lstm layer at last iteration
                
        i_t = self.__sigmoid(xi + np.dot(hprev, U_i)) #[1,m] + [m]*[m,m] -> [1,m]
        f_t = self.__sigmoid(xf + np.dot(hprev, U_f)) #[1,m] + [m]*[m,m] -> [1,m]
        o_t = self.__sigmoid(xo + np.dot(hprev, U_o)) #[1,m] + [m]*[m,m] -> [1,m]
        c_t = f_t*self.memory[i] + i_t * np.tanh(xc + np.dot(hprev, U_c)) #[1,m]*[m] + [1,m] * [1,m] -> [1,m]
        h_t = o_t * np.tanh(c_t) #[1,m]*[1,m] (elementwise)
        self.memory[i] = c_t
        self.output[i] = h_t
        
        return h_t
       


    def Update(self,assembly, pb, dtime, nlgeom=True):            
        displacement = pb.GetDoFSolution()
            
        if displacement is 0: 
            self.__currentStrain = 0
            self.__currentStress = None
        else:                    
            self.__currentStrain = np.array(assembly.GetStrainTensor(displacement, "GaussPoint")).T         
                
        #tranpose for comatibility with simcoon
        # if displacement is 0: 
        #     self.__currentGradDisp = 0
        #     self.__currentStress = None
        #     F1 = np.multiply(np.eye(3).reshape(3,3,1) , np.ones((1,1,self.nb_points)), order='F').transpose(2,0,1)
        #     # self.__F1 = np.eye(3).T.reshape(1,3,3)
        # else:                    
        #     self.__currentGradDisp = np.array(assembly.GetGradTensor(displacement, "GaussPoint"))            
        #     #F0.strides and F1.strides should be [n_cols*n_rows*8, 8, n_rows*8] for compatibiliy with the sim.RunUmat_fedoo function
        #     F1 = np.add( np.eye(3).reshape(3,3,1), self.__currentGradDisp, order='F').transpose(2,0,1)                        

        # self.compute_Detot(dtime, F1)   
        self.Run(dtime)
        # (DRloc , listDR, Detot, statev) = self.Run(dtime)
        
        
    # def GetStress(self, GradDispTensor, time = None):
    #     self.__GradDispTenorOld = self.__GradDispTensor
    #     self.__GradDispTensor = GradDispTensor
        
        
    #     # time not used here because this law require no time effect
    #     # initilialize values plasticity variables if required

    #     self.__currentStress = listStressTensor(sigmaFull.T) # list of 6 objets

"""        

