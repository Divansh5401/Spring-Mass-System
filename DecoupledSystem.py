import numpy as np
import scipy as sp
from scipy.sparse import diags,csr_array
import scipy.integrate as sp_int
import matplotlib.pyplot as plt

mass_Number = int(input("number of mass blocks: "))
kValues = np.random.randint(5,1000,mass_Number)
mMatrix = np.diag(np.random.randint(5,1000,mass_Number))
x0 = 30*np.ones(mass_Number)
v0 = np.zeros(mass_Number)


def Building_K_Matrix(Number_of_Masses,Values):
        matrix=[]
        for i in range(Number_of_Masses):
            row = np.zeros(Number_of_Masses)
            if i !=0:
                row[i-1]= - Values [i]
            if i != (Number_of_Masses - 1):
                row[i+1] = - Values[i+1]
                row[i] = Values[i] + Values[i+1]
            if i == Number_of_Masses-1:
                row[i] = Values[i]
            matrix.append(row)
        return np.array(matrix)

kMatrix = Building_K_Matrix(mass_Number,kValues)

Eignvalues, Eignvectors = sp.linalg.eigh(kMatrix,mMatrix)
zeroMatrix = np.zeros((mass_Number,mass_Number))
identityMatrix = np.eye(mass_Number)
transformMatrix = np.block([[zeroMatrix,identityMatrix],[-np.diag(Eignvalues),zeroMatrix]])
def odeeq(t,y):
    dotY = transformMatrix @ y
    return dotY

tSpan = [0, 100]
ts = np.linspace(tSpan[0],tSpan[1],5000)

print(Eignvectors.T@mMatrix,np.linalg.det(Eignvectors)*np.sqrt(np.linalg.det(mMatrix)))
X0 = Eignvectors.T@mMatrix @ x0
V0 = Eignvectors.T@mMatrix @ v0
y0 = np.concatenate([X0,V0])
V = sp_int.solve_ivp(odeeq,tSpan,y0,t_eval=ts)
print(V.y.shape)


displacementMatrix = Eignvectors @ V.y[0:mass_Number,:]


plt.plot(displacementMatrix.T)
plt.show()