# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def norm(input):
    input=input-input.mean(axis=0)
    input=input/input.std(axis=0)
    return input


# %%

def my_regression(trainX,testX,noutputs):
    phi=None
    phi=trainX[:,0:trainX.shape[1]-noutputs]

    T=trainX[:,trainX.shape[1]-noutputs:]
    T=norm(T)

    # try different model 
    poly_degree=(1,2,3,4,5,6) # 1 for linear model,2~6 for poly model
    m_lambda=(0,0.0001,0.001,0.01,0.1,1,10) # 0 for no regular, 

    # using different method to build the input for different modle
    models_map=dict() # record all the models and cv loss
    for t_poly in poly_degree: # test every poly degree
        #print("degree = %d"%(t_poly))
        phi=trainX[:,0:trainX.shape[1]-noutputs]
        # build poly input
        poly=PolynomialFeatures(t_poly)
        phi=poly.fit_transform(phi) # the fist column is all one

        phi[:,1:]=norm(phi[:,1:]) # exclude the first column


        for t_lambda in m_lambda: #test every lambda for
            #print("lmbda=",t_lambda)
            cv=np.array([])
            for cross_valid in range(5):# cross validation
                total=len(phi)
                start=int(total*cross_valid/5)
                end=int(total*(cross_valid+1)/5)
                if start==end:
                    end=end+1
                test_case=phi[start:end]
                train_case=np.append(phi[0:start],phi[end:],axis=0)
                test_T=T[start:end]
                train_T=np.append(T[0:start],T[end:],axis=0)
                #print("test_case",test_case)
                #print("train_case",train_case)
                #print("test_T=",test_T)
                #print("train_T=",train_T)
                
                W=np.dot(np.dot(np.linalg.inv( t_lambda * np.eye(train_case.shape[1]) + np.dot(train_case.T,train_case)),train_case.T),train_T)
                #print("W=",W)
                #print("train_out=",np.dot(train_case,W))
                test_out=np.dot(test_case,W)
                #print(test_out)
                t_loss=np.average((test_out-test_T)**2)
                #print("t_loss=",t_loss)
                cv=np.append(cv,t_loss)
            #print(cv)# the cv 
            models_map[(t_poly,t_lambda)]=np.average(cv)
    #print(models_map)
    min=float("inf")
    min_model=0
    #print(models_map)
    for model in models_map:
        if models_map[model]<min:
            min=models_map[model]
            min_model=model
    
    #print(min_model)
    #print(min)
    t_poly,t_lambda=min_model

    # train the whole trainning set
    phi=trainX[:,0:trainX.shape[1]-noutputs]
        # build poly input
    poly=PolynomialFeatures(t_poly)
    phi=poly.fit_transform(phi) # the fist column is all one   
    phi[:,1:]=norm(phi[:,1:]) # exclude the first column

    W=np.dot(np.dot(np.linalg.inv( t_lambda * np.eye(phi.shape[1]) + np.dot(phi.T,phi)),phi.T),T)
    #print(testX)
    poly=PolynomialFeatures(t_poly)
    testX=poly.fit_transform(testX) # the fist column is all one   
    testX[:,1:]=norm(testX[:,1:]) # exclude the first column
    #print(testX)
    return np.dot(testX,W)




# %%

##main:
import matplotlib.pyplot as plt
#trainX=np.array([[1,2,4],[4,21,29],[5,10,20],[2,3,7],[8,100,116],[3,4,10],[9,100,118]])
def fx(x,delta):
    return 0.1*(x**3)+(np.cos(x**2)) + np.random.normal(0,delta)
x=np.linspace(-5.0,5.0,101)
y=np.array([fx(t,0) for t in x])
print(x)
print(y)
plt.plot(x,y)

y1=np.array([fx(t,0.1) for t in x])
y2=np.array([fx(t,0.2) for t in x])
y3=np.array([fx(t,0.5) for t in x])
y4=np.array([fx(t,1) for t in x])

plt.plot(x,y1,label="0.1")
plt.plot(x,y2,label="0.2")
plt.plot(x,y3,label="0.5")
plt.plot(x,y4,label="1")
plt.legend()
plt.show()

#%%
#trainX=np.array([[1,2,4],[4,21,29],[5,10,20],[2,3,7],[8,100,116],[3,4,10],[9,100,118]])
def fx(x,y):
    return x**3+y**2 + np.random.rand()
x=[2,4,9,11,23,44,98,91,49,29,87]
y=[5,9,18,13,22,19,100,2,19,11,23]
tx=[22,33,44,55]
ty=[11,33,12,42]

trainX=np.array([])
for i in range(len(x)):
    trainX=np.append(trainX,[[x[i],y[i],fx(x[i],y[i])]])
trainX=trainX.reshape(-1,3)

testX=np.array([])

for i in range(len(tx)):
    testX=np.append(testX,[[tx[i],ty[i]]])
testX=testX.reshape(-1,2)
#print(trainX)
#print(testX)
#trainX=np.array([[1,2,4],[4,21,29],[5,10,20],[2,3,7],[8,100,116],[3,4,10],[9,100,118]])
#testX=np.array([[1,1],[2,2],[3,3]])

testOut=my_regression(trainX,testX,1)
print(testOut)
testT=np.array([fx(x,y) for x,y in zip(tx,ty)])
testT.reshape(-1,1)
testT=norm(testT)
print(testT)
#print(testOut)
#print([fx(x,y) for x,y in zip(tx,ty) ])


   


# %%
data="./airfoil_self_noise.dat"
inputs=[]
with open(data) as f:
    for line in iter(f.readline,''):
        inputs.append(line.split())
inputs=np.array(inputs)
inputs=inputs.astype(np.float)
trainX=inputs[0:-10]
testX=inputs[-10:]
print(trainX.shape)
print(testX.shape)
print(testX)

testOut=my_regression(trainX,testX[:,0:-1],1)
print(testOut)
testT=testX[:,-1:]
testT=norm(testT)
print(testT)
pass


# %%


