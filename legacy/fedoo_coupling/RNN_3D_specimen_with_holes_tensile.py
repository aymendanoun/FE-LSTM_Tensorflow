#--------------- Pre-Treatment --------------------------------------------------------

from fedoo import *
import numpy as np
from time import time
import os
import pylab as plt
from numpy import linalg

start = time()
#--------------- P

Util.ProblemDimension("3D")

Mesh.ImportFromFile('specimen_with_holes_refined.msh', meshID = "Domain")

meshID = "Domain2"


mesh = Mesh.GetAll()[meshID]

crd = mesh.GetNodeCoordinates()
type_el = mesh.GetElementShape()





NLGEOM = False


xmax = np.max(crd[:,0]) ; xmin = np.min(crd[:,0])
mesh.AddSetOfNodes(list(np.where(crd[:,0] == xmin)[0]), "left")
mesh.AddSetOfNodes(list(np.where(crd[:,0] == xmax)[0]), "right")





E = 113800
nu=0.342
alpha = 0.86e-5 #???






mat = 3
if mat == 0:
    props = np.array([[E, nu, alpha]])
    Material = ConstitutiveLaw.Simcoon("ELISO", props, 1, ID='ConstitutiveLaw')
    Material.corate = 0
elif mat == 1:
    Re = 1000
    k=1600
    m=0.5
    props = np.array([[E, nu, alpha, Re,k,m]])
    Material = ConstitutiveLaw.Simcoon("EPICP", props, 8, ID='ConstitutiveLaw')
    Material.corate = 0
elif mat == 3:
    #lstm
    Material = ConstitutiveLaw.LSTM(ID='ConstitutiveLaw')
else:
    Material = ConstitutiveLaw.ElasticIsotrop(E, nu, ID='ConstitutiveLaw')

WeakForm.InternalForce("ConstitutiveLaw", nlgeom = NLGEOM)




Assembly.Create("ConstitutiveLaw", meshID, 'tet4', ID="Assembling")     #uses MeshChange=True when the mesh change during the time

Problem.NonLinearStatic("Assembling")

# Problem.SetSolver('cg', precond = True)

Problem.SetNewtonRaphsonErrorCriterion("Displacement")
# Problem.SetNewtonRaphsonErrorCriterion("Work")
# Problem.SetNewtonRaphsonErrorCriterion("Force")

#create a 'result' folder and set the desired ouputs
if not(os.path.isdir('RNN_results_specimen_with_holes_tensile')): os.mkdir('RNN_results_specimen_with_holes_tensile')
# Problem.AddOutput('results/bendingPlastic', 'Assembling', ['disp', 'kirchhoff', 'cauchy', 'PKII', 'strain', 'cauchy_vm', 'statev'], output_type='Node', file_format ='vtk')    
# Problem.AddOutput('results/bendingPlastic', 'Assembling', ['kirchhoff', 'cauchy', 'PKII', 'strain', 'cauchy_vm', 'statev'], output_type='Element', file_format ='vtk')    

if mat == 3: filename = 'RNN_specimen_with_holes_tensile'


Problem.AddOutput('RNN_results_specimen_with_holes_tensile/'+filename, 'Assembling', ['disp', 'stress', 'strain', 'stress_vm'], output_type='Node', file_format ='vtk')   
Problem.AddOutput('RNN_results_specimen_with_holes_tensile/'+filename, 'Assembling', ['disp', 'stress', 'strain', 'stress_vm'], output_type='Node', file_format ='txt')   
# Problem.AddOutput('results/bendingPlastic', 'Assembling', ['kirchhoff', 'cauchy', 'PKII', 'strain', 'cauchy_vm', 'statev'], output_type='Element', file_format ='vtk')    


################### step 1 ################################
tmax = 1
uimp = 1
F = 50



Problem.BoundaryCondition('Dirichlet','DispX',0,mesh.GetSetOfNodes("left"))
Problem.BoundaryCondition('Dirichlet','DispY', 0,mesh.GetSetOfNodes("left"))
Problem.BoundaryCondition('Dirichlet','DispZ', 0,mesh.GetSetOfNodes("left"))
Problem.BoundaryCondition('Dirichlet','DispY',0,mesh.GetSetOfNodes("right"))
Problem.BoundaryCondition('Dirichlet','DispX',0.04,mesh.GetSetOfNodes("right"))


#Problem.BoundaryCondition('Dirichlet','DispY',0,nodes_bottomRight)



Problem.NLSolve(dt = 0.01, tmax = tmax, update_dt = False, ToleranceNR = 0.05)

################### step 2 ################################
# bc.Remove()
# #We set initial condition to the applied force to relax the load
# F_app = Problem.GetExternalForce('DispY')[nodes_topCenter]
# bc = Problem.BoundaryCondition('Neumann','DispY', 0, nodes_topCenter, initialValue=F_app)#face_center)

# Problem.NLSolve(dt = 1., update_dt = True, ToleranceNR = 0.01)

print(time()-start)








