import numpy as np

a = [[1,2,3,5],[7,8,9,2],[1,5,9,4],[7,5,3,4]]
a = np.array(a)
print(a)
print(a.shape)

b = a.reshape(2,8)
print(b)

c = b.reshape(4,4)
print(c)

d = np.concatenate((a[:2,:],a[2:,:]), axis=1)
print(d)

e = np.concatenate((d[:,:4],d[:,4:8]), axis=0)
print(e)