import jax
import jax.numpy as jnp
from jax.scipy.linalg import eigh
import matplotlib.pyplot as plt

key = jax.random.PRNGKey(0)
key, subkey,k2 = jax.random.split(key,3)
mass_Number = int(input("number of mass blocks: "))
kValues = jax.random.randint(key, shape=(mass_Number,), minval=5, maxval=1000)
mValues = jax.random.randint(subkey, shape=(mass_Number,), minval=5, maxval=1000)
print(kValues,mValues)
mMatrix = jnp.diag(mValues)
v0 = jnp.zeros(mass_Number)
x0 = 30*jnp.ones(mass_Number)
    
def Building_K_Matrix(Number_of_Masses,Values):
    matrix= jnp.zeros((Number_of_Masses,Number_of_Masses))
    for i in range(Number_of_Masses):
        if i !=0:
            matrix = matrix.at[i,i-1].set(- Values.at[i].get())
        if i != (Number_of_Masses - 1):
            matrix = matrix.at[i,i+1].set(- Values.at[i+1].get())
            matrix = matrix.at[i,i].set(Values.at[i].get() + Values.at[i+1].get())
        if i == Number_of_Masses-1:
            matrix = matrix.at[i,i].set(Values.at[i].get())
    return matrix

kMatrix = Building_K_Matrix(mass_Number,kValues)
MassSqrt = jnp.diag(1.0 / jnp.sqrt(jnp.diag(mMatrix))) 
A = MassSqrt @ kMatrix @ MassSqrt
B = jnp.linalg.inv(mMatrix) @ kMatrix
Eigenvalues, Eigenvectors = eigh(A)
Eigenvectors = MassSqrt @ Eigenvectors

omega = jnp.sqrt(jnp.diag(Eigenvalues))
invEigenvectors = jnp.linalg.inv(Eigenvectors)

A0 = invEigenvectors @ x0
V0 = invEigenvectors @ v0

B0 = jnp.linalg.solve(omega,V0)
t = jnp.tile(jnp.linspace(0.0, 100.0, 2000), (mass_Number,1))
X_t = jnp.diag(A0) @ jnp.cos(omega@t) + jnp.diag(B0) @ jnp.sin(omega@t)

X_t = Eigenvectors @ X_t


plt.plot(X_t.T)
plt.show()