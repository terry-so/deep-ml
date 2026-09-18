import numpy as np

def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:



                a = np.array(a)
                b = np.array(b)
    
                try:
                    c = np.matmul(a,b)
                    return c
                except:
                    return -1

