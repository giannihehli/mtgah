import numpy as np


def bottom(image_thr):
    x_bottom = 0
    y_bottom = 0

    nrows, ncols = np.shape(image_thr)

    ind, _ = np.mgrid[0:nrows, 0:ncols]

    print('ind fun: ', ind)

    b = image_thr == 1

    print('b.shape: ', b.shape)
    print('ind.shape: ', ind.shape)

    x_bottom = np.amax(ind[b])
    print('x_bottom: ', x_bottom)

    ind_1d = np.arange(ncols)

    print('ind_1d: ', ind_1d)

    b_1d = image_thr[x_bottom, :] == 1
    print('b_1d: ', b_1d)
    
    y_bottom = int(np.median(ind_1d[b_1d]))
    print('y_bottom: ', y_bottom)

    return x_bottom, y_bottom
arr = np.zeros((100,100))
arr[30:70,30:70]= 1

col_arr  = arr[50,:]



ind = np.arange(len(col_arr))

b = col_arr == 1

p_max = np.amax(ind[b])



arr_test = arr[:, 45:55]

col_max, row_max = bottom(arr_test)

ind = np.mgrid[45:55,0:100]

