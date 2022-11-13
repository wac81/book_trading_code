import numpy as np
import matplotlib.pyplot as plt


def delextremum(array, n=2):
    '''
    @param array:<type> numpy.array  2D or 1D
    '''
    sigma = array.std(axis=0)
    mu = array.mean(axis=0)
    up_limit = mu + n*sigma
    down_limit = mu - n*sigma


    if up_limit.shape == ():
        array[array > up_limit] = up_limit
        array[array < down_limit] = down_limit
    else:
        array_len = array.shape
        array[array > up_limit] = np.repeat(up_limit.reshape(1, up_limit.shape[0]), array_len[0], axis=0)[array > up_limit]
        array[array < down_limit] = np.repeat(down_limit.reshape(1, down_limit.shape[0]), array_len[0], axis=0)[array < down_limit]
    return array


if __name__ == '__main__':

    # a = np.random.normal(1000,50,(100, 200))

    # a[33][56] = 2000
    # a[34][58] = 2000

    # a[88][38] = 100
    # a[84][18] = 100

    a = np.random.normal(1000, 100, 100)
    
    a[33] = 100
    a[58] = 2000

    n = 2

    mu = a.std(axis=0)
    mean = a.mean(axis=0)
    up = [mean + n * mu] * len(a)
    down = [mean - n * mu] * len(a)
    plt.plot(a, 'green', alpha=0.5)

    plt.plot(up)
    plt.plot(down)

    b = delextremum(a,n)
    plt.plot(b, 'red', alpha=0.5)
    plt.show()
    plt.savefig('extremum.png')