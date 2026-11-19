import numpy as np
import pandas as pd
from math import sqrt,pi
from matplotlib.ticker import FormatStrFormatter
import os, sys
from pandas import DataFrame
from simcoon import simmit as sim
from random import *
from fedoo import *
import shutil
import multiprocessing as mp
from joblib import Parallel, delayed
import time







    
def create_path(Nsteps,Dninc,E,name_file):
    
    dir = os.path.dirname(os.path.realpath('__file__'))
    pathh = dir + '/' + folder_data

    variable = '/path_{}.txt'.format(name_file)
    fichier = open(pathh+variable,'w')
    fichier.write('#initial_temperature\n'+str(290)+'\n')
    fichier.write('#Number_of_blocks\n'+str(1)+'\n')
    fichier.write('#Block\n'+str(1)+'\n')
    fichier.write('#Loading_type\n'+str(1)+'\n')
    fichier.write('#Control_type(NLGEOM)\n'+str(1)+'\n')
    fichier.write('#repeat\n'+str(1)+'\n')
    #N-steps
    fichier.write('#steps\n'+str(Nsteps)+'\n')
    fichier.write('\n')
    #Mode 1######################################################################################

    for i in range(Nsteps):
        fichier.write('#Mode\n'+str(1)+'\n')        
        fichier.write('#Dn_init\t'+str(1.)+'\n')        
        fichier.write('#Dn_mini\t'+str(1.)+'\n')        
        #Dn_inc
        fichier.write('#Dn_inc\t' + str(Dninc) + '\n')
        fichier.write('#time\n'+str(1)+'\n')
        fichier.write('#Consigne\n')
    
        fichier.write('E '+ str(E[i,0]) +'\n')
        fichier.write('E '+ str(E[i,1]) +' ')
        fichier.write('E '+ str(E[i,2]) +'\n')       
        fichier.write('E '+ str(E[i,3]) +' ')       
        fichier.write('E '+ str(E[i,4]) +' ')       
        fichier.write('E '+ str(E[i,5]) +'\n')
        fichier.write('#Consigne_T\nT\t'+str(290)+'\n')
        fichier.write('\n')
    fichier.close()

def generate_loadings(N_data,N_Steps):
    E = np.zeros((N_data,N_Steps,6))
    for i in range(N_data):
        for j in range(N_Steps):
            e11 = np.random.uniform(0,0.05)
            e12 = np.random.uniform(0,0.025)
            e22 = np.random.uniform(0,0.05)
            e13 = 0
            e23 = 0
            e33 = 0
            E[i][j] = np.array([e11,e12,e22,e13,e23,e33],dtype=object)
    return E  
  

def create_paths(N,Nsteps,Dninc,E):
    for i in range(N):
        create_path(int(Nsteps[i]), Dninc[i], E[i],i)  


def Run_simulation(meshID, umat_name, props, nstatev, solver_type, corate_type, path_data, path_results, path_file, outputfile, outputdat_file):
    Homogen.GetResultsUnitCell(meshID, umat_name, props, nstatev, solver_type, corate_type, path_data, path_results, path_file, outputfile, outputdat_file) 
    



#############################################################################################################

N_Data = 10000
N_Steps = 4
Increment = 0.04



folder_data = 'Path'

folder_results = 'Results_6'
os.mkdir(folder_results)


   
Util.ProblemDimension("3D")
Mesh.ImportFromFile('octet_surf.msh', meshID = "Domain")
meshID = "Domain2"
umat_name = 'EPICP'

E = 113800
nu = 0.342
alpha = 0.86E-5
sigma_Y = 1000
H = 1600
beta = 0.25

props = np.array([E, nu, alpha, sigma_Y, H, beta])
nstatev = 8


solver_type = 0
corate_type = 2


path_data = folder_data
path_results = folder_results
outputdat_file='output.dat'



num_cores = mp.cpu_count() 
pool = mp.Pool(num_cores)




for i in range(5000,6000):
    
    try:
    
        path_file = 'path_{}.txt'.format(i)
        outputfile = 'Data_{}.csv'.format(i)
        #Run_simulation(meshID, umat_name, props, nstatev, solver_type, corate_type, path_data, path_results, path_file, outputfile, outputdat_file, ProblemID = 'pb'+str(i))
        r = pool.apply_async(Run_simulation, args=(meshID, umat_name, props, nstatev, solver_type, corate_type, path_data, path_results, path_file, outputfile, outputdat_file))

        
    except:
        
        print('Convergence Failed_'+str(i))


for i in range(5000,6000):
    r.wait()         



print("Toto")       
pool.terminate()


 