#import os
#import numpy as np
#import matplotlib.pyplot as plt
#from numpy.fft import rfftn, irfftn
#import h5py 
#import math
#import scipy

def sqrt(x):
    return(np.sqrt(x))
def log(x):
    return(np.log(x))

def get_param(x,path):
    a1=(1-2.65/3)
    f=open(path + "example.c", 'r')
    file=f.readlines()
    f.close()
    for l in file:
        if '#define' in l and x in l.split('(')[0]:
            break
    print(l)
    print(eval(l.split()[2][1:-1]))
    return  (eval(l.split()[2][1:-1]))
def convert_binary(path, NX, NY, NZ):
    dat = np.fromfile(path)
    dat_2 = dat.reshape(NX,NY,NZ)
    return dat_2
#loads Q tensor
def load_Q(root, frame, NX, NY, NZ):
    """
    Loads array from binary files for order param \n
    \n
    Parameters\n
    ----------\n
    root : String of the path to the directory containing the raw files to be loaded \n 
    frame : Integer giving which time point to take Q from. Will be substituded in with string replacement \n  
    NX,NY,NZ: Gives the grid size of the simulation\n
    \n
    Returns\n
    -------\n
    Q: returns a 3x3xNXxNYxNZ numpy array representing the Q tensor at each point in space


    """
    Q = np.zeros((3,3,NX,NY,NZ))
    Q[0,0] = convert_binary(path+r'/Qxx_%d.dat'%frame, NX, NY, NZ)
    Q[1,0] = convert_binary(path+r'/Qxy_%d.dat'%frame, NX, NY, NZ)
    Q[2,0] = convert_binary(path+r'/Qxz_%d.dat'%frame, NX, NY, NZ)
    Q[1,1] = convert_binary(path+r'/Qyy_%d.dat'%frame, NX, NY, NZ)
    Q[1,2] = convert_binary(path+r'/Qyz_%d.dat'%frame, NX, NY, NZ)
    Q[1,0] = Q[0,1]
    Q[0,2] = Q[2,0]
    Q[2,1] = Q[1,2]
    Q[2,2] = -Q[0,0] - Q[1,1]
    return Q
#different functions bc u is smaller than Q
def load_u(root,frame,NX, NY, NZ):
    """
    Loads velocity vectors\n
    
    Parameters\n
    ----------\n
    root : String of the path to the directory containing the raw files to be loaded\n  
    frame : Integer giving which time point to take Q from. Will be substituded in with string replacement\n  
    NX,NY,NZ: Gives the grid size of the simulation\n
    \n
    Returns\n
    -------\n
    \n
    Q: returns a 3xNXxNYxNZ numpy array representing the velocity in x, y, and z directions at each point in space 
    

    """
    u=np.zeros((3,NX,NY,NZ))
    u[0]=convert_binary(path+r'/ux_%d.dat'%frame, NX, NY, NZ)
    u[1]=convert_binary(path+r'/uy_%d.dat'%frame, NX, NY, NZ)
    u[2]=convert_binary(path+r'/uz_%d.dat'%frame, NX, NY, NZ)
    return u
def diagonalizeQ(qtensor):
    """
    Diagonalization of Q tensor in 3D nematics.
    Currently it onply provides the uniaxial information.
    Will be updated to derive biaxial analysis in the future.
    Algorythm provided by Matthew Peterson:
    THIS CODE IS FROM YINGYOU MA, NOT ME
    https://github.com/YingyouMa/3D-active-nematics/blob/405c8d54d797cc39c1f14c82112cb43d304ef16c/reference/order_parameter_calculation.pdf

    Parameters
    ----------
    qtensor : numpy array, N x M x L x 5  or  N x M x L x 3x 3
              tensor order parameter Q of each grid
              N, M and L are the number of grids in each dimension.
              The Q tensor for each grid could be represented by 5 numbers or 3 x 3 = 9 numbers
              If 5, then qtensor[..., 0] = Q_xx, qtensor[..., 1] = Q_xy, and so on. 
              If 3 x 3, then qtensor[..., 0,0] = Q_xx, qtensor[..., 0,1] = Q_xy, and so on.
              

    Returns
    -------
    S : numpy array, N x M x L
        the biggest eigenvalue as the scalar order parameter of each grid

    n : numpy array, N x M x L x 3
        the eigenvector corresponding to the biggest eigenvalue, as the director, of each grid.


    Dependencies
    ------------
    - NumPy: 1.22.0

    """

    N, M, L = np.shape(qtensor)[:3]

    if np.shape(qtensor) == (N, M, L, 3, 3):
        Q = qtensor
    elif np.shape(qtensor) == (N, M, L, 5):
        Q = np.zeros( (N, M, L, 3, 3)  )
        Q[..., 0,0] = qtensor[..., 0]
        Q[..., 0,1] = qtensor[..., 1]
        Q[..., 0,2] = qtensor[..., 2]
        Q[..., 1,0] = qtensor[..., 1]
        Q[..., 1,1] = qtensor[..., 3]
        Q[..., 1,2] = qtensor[..., 4]
        Q[..., 2,0] = qtensor[..., 2]
        Q[..., 2,1] = qtensor[..., 4]
        Q[..., 2,2] = - Q[..., 0,0] - Q[..., 1,1]
    else:
        raise NameError(
            "The dimension of qtensor would be (N, M, L, 3, 3) or (N, M, L, 5)"
            )

    p = 0.5 * np.einsum('ijkab, ijkba -> ijk', Q, Q)
    q = np.linalg.det(Q)
    r = 2 * np.sqrt( p / 3 )

    # derive S and n
    temp = 4 * q / r**3
    temp[temp>1]  =  1
    temp[temp<-1] = -1
    S = r * np.cos( 1/3 * np.arccos( temp ) )
    temp = np.array( [
        Q[..., 0,2] * ( Q[..., 1,1] - S ) - Q[..., 0,1] * Q[..., 1,2] ,
        Q[..., 1,2] * ( Q[..., 0,0] - S ) - Q[..., 0,1] * Q[..., 0,2] ,
        Q[..., 0,1]**2 - ( Q[..., 0,0] - S ) * ( Q[..., 1,1] - S  )
        ] )
    n = temp / np.linalg.norm(temp, axis = 0)
    n = n.transpose((1,2,3,0))
    S = S * 1.5

    return S, n
def autocorrelation(x,axes=None):
    """
    returns the autocorellation function\n
    \n
    Parameters\n
    ----------\n
  \n
    x=array that you give to operate on \n 
    axes: axes which will be corellated over. If none given, will assume all axes\n
  \n
    Returns\n
    -------\n
  
    autocorellation function of given array\n
   

    """
    f=rfftn(x,axes=axes)
    cor=irfftn(f*np.conjugate(f), axes=axes)
    return np.fft.fftshift(cor,axes=axes)

def Q_show(Q,t):
    """
    Plots average values in x-y plane of Qxx, Qyy, and Qzz over the z axis\n
    \n
    Parameters\n
    ----------\n
    \n
    Q = Order Parameter Tensor\n
    t = frame of order parameter tensor\n 
    
    

    """
    plt.title("Qxx over Z at time " + str(t))
    plt.xlabel("Z")
    plt.ylabel("Qxx")
    plt.plot(np.mean(Q[0,0,:,:,:],(0,1)))
    plt.show()
    plt.title("Qyy over Z at time " + str(t))
    plt.plot(np.mean(Q[1,1,:,:,:],(0,1)))
    plt.xlabel("Z")
    plt.ylabel("Qyy")
    plt.show()
    plt.title("Qzz over Z at time " + str(t))
    plt.plot(np.mean(Q[2,2,:,:,:],(0,1)))
    plt.xlabel("Z")
    plt.ylabel("Qzz")
    plt.show()
def Display_Cor():
    """
    Calculates the average corellation function of a quantity in the midplane\n
    \n
  
    Returns\n
    -------\n
    
    average corellation of quantity with itself at the center of the midplane sorted by distance from center of the plane

    """
    X, Y = np.meshgrid(np.arange(NX), np.arange(NY))
    R = np.sqrt((X-128)**2 + (Y-128)**2)
    C=np.sum(autocorrelation(u[:,:,:,int(NZ/2)], axes=[1,2]), axis=0)
    C=C/np.max(C)
    #Unique_R=np.unique(R.flatten())
    #Unique_R=np.where(Unique_R%1==0)[0]
    avg_cor=np.empty(0)
    R=np.round(R)
    for r in range(int(NX/2)):
        Cmask=np.ma.masked_where(R!=r,C)
        avg_Cmask=np.average(Cmask)
        avg_cor=np.append(avg_cor,avg_Cmask)
        
   # plt.plot( np.arange(128),avg_cor)
    #plt.ylabel("Velocity Corellation")
    #plt.xlabel("Horizontal Distance")
    #plt.close()
    return avg_cor
