import numpy as np
f=open("airfoil_self_noise.dat",mode="r")
line=np.array(f.read().split())
print(line.shape)