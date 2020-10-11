# For 2-D array, it is matrix multiplication 
import numpy.matlib 
import numpy as np 

a = [[3,-np.sqrt(3)],[-np.sqrt(3),1]] # V
b = [[1/2,np.sqrt(3)/2],[-np.sqrt(3)/2,1/2]] #PT
print(b)
c= [[1/2,-np.sqrt(3)/2],[np.sqrt(3)/2,1/2]]  #P
t=np.linalg.inv(c)
print(t)
r=np.matmul(a,c)
s=np.matmul(b,r)
print (s)
