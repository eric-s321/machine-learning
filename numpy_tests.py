import time
import numpy as np



def my_dot(a, b): 
    x=0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x


def time_function(a,b,func):
    start = time.time()
    func(a,b)
    print(f'function took: {time.time()-start} seconds')



array_size = int(1E7)
a = np.random.rand(array_size)
b = np.random.rand(array_size)


print(f'Computing dot product using np.dot with array size {array_size}')
time_function(a,b,np.dot)
print(f'Computing dot product using a custom loop implementation with array '
      f'size {array_size}')
time_function(a,b,my_dot)

