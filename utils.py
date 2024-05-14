import os
import numpy as np


# taken from Physics-as-Inverse-Graphics
def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape

    bordered = 0.5*np.ones([nindex, height+2, width+2, intensity])
    for i in range(nindex):
        bordered[i,1:-1,1:-1,:] = array[i]

    array = bordered
    nindex, height, width, intensity = array.shape

    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result