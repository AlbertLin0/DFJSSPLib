import numpy as np 


if __name__ == "__main__":
    data = np.load('J20_M15.npy')
    print(data[0,:,:,:])
    data_10 = np.load('dynamic/10/J20_M15.npy')
    print(data_10[0,:,:,:])
    a = data[1,:,:,:]
    b = data_10[1,:,:,:]
    print(a - b)