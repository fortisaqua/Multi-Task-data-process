import numpy as np

array1 = np.zeros([64,64,64],np.uint8)
array2 = np.zeros([64,64,64],np.uint8)

array3 = np.stack([array1,array2],axis=0)
print np.shape(array3)

