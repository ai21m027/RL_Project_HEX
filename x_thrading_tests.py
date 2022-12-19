from multiprocessing import Pool, freeze_support
import time
import numpy as np

def foo(x):
    return [1*x,2*x,3*x]

if __name__ == '__main__':
    freeze_support()
    pool = Pool(8)

    print(np.reshape(pool.map(foo, [1,2,3,4,5,6,7,8,9,10]),-1))

    pool.close()
    pool.join()