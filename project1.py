# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np

#settings for the model
isPoly=False
isLeanear=True
PolyDegree=2
usingRegularizer=True
Lambda=0.01
W=None # current weight
def my_regression(trainX,testX,noutputs):
    ## coding here
    print(trainX.shape)
    print(testX.shape)
    print(noutputs)
    phi=None
    T=trainX[:,trainX.shape[1]-noutputs:]
    if isLeanear:
        print("Using leanear model")
        phi=trainX[:,0:trainX.shape[1]-noutputs]
    
        ones=np.ones((trainX.shape[0],1))
        phi=np.append(ones,phi,axis=1)
        print("phi=")
        print(phi)
    elif isPoly:
        print("using poly")
        phi=trainX[:,0:trainX.shape[1]-noutputs]
        new_phi=phi
        for i in range(2,PolyDegree+1): 
            new_phi=np.append(new_phi,phi**i,axis=1)
        phi=new_phi
        print("phi:\n",phi)
        ones=np.ones((trainX.shape[0],1))
        phi=np.append(ones,phi,axis=1)

    if usingRegularizer==True:
        print("using regularizer")
        W=np.dot(np.dot(np.linalg.pinv( Lambda * np.eye(phi.shape[1]) + np.dot(phi.T,phi)),phi.T),T)
    else:
        print("not using regularizer")
        W=np.dot(np.dot(np.linalg.pinv(np.dot(phi.T,phi)),phi.T),T)
    print("T=")
    print(T)
    print("W=")
    print(W)
    outputs=np.dot(phi,W)
    print("outputs=")
    print(outputs)

    return np.dot(np.append(np.ones((testX.shape[0],1)),testX,axis=1),W)

    

    

    pass




##main:


# %%


trainX=np.array([[1,2,4],[2,3,7],[3,4,10],[9,100,118]])
testX=np.array([[1,1],[2,2],[3,3]])
noutputs=1
isLeanear=False
isPoly=True
PolyDegree=3
testOut=my_regression(trainX,testX,noutputs)
print(testOut)

   


# %%
data="./airfoil_self_noise.dat"
inputs=[]
with open(data) as f:
    for line in iter(f.readline,''):
        inputs.append(line.split())
inputs=np.array(inputs)
trainX=inputs[0:-1]
testX=inputs[-1:]
print(trainX.shape)
print(testX.shape)
print(testX)
testOut=my_regression(trainX,testX[:,0:-1],1)
print(testOut)
pass


# %%


