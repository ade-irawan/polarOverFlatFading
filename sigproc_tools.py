from keras.models import Sequential
from keras import backend as K
import numpy as np

def modulateBPSK(x):
    return -2*x +1;

def addNoise(x, sigma):
    w = K.random_normal(K.shape(x), mean=0.0, stddev=sigma)
    return x + w

def addFading(x, sigma):
    # noise
    n_re = K.random_normal(K.shape(x), mean=0.0, stddev=sigma)
    n_im = K.random_normal(K.shape(x), mean=0.0, stddev=sigma)
    # rayleigh fading
    h_re = K.random_normal((K.shape(x)[0],1), mean=0.0, stddev=np.sqrt(1/2))
    h_im = K.random_normal((K.shape(x)[0],1), mean=0.0, stddev=np.sqrt(1/2))
    # out
    x_re = x * h_re + n_re
    x_im = x * h_im + n_im 
    return h_re * x_re + h_im * x_im

def ber(y_true, y_pred):
    return K.mean(K.not_equal(y_true, K.round(y_pred)))

def bler(y_true, y_pred):
    return K.mean(K.not_equal(y_true, K.round(y_pred)), axis=-1)

def returnOutputShape(input_shape):  
    return input_shape

def logLikelihoodRatio(x, sigma):
    return 2*x/np.float32(sigma**2)

def errors(y_true, y_pred):
    #return K.sum(K.not_equal(y_true, K.round(y_pred)))
    return K.sum(K.cast(K.not_equal(y_true, K.round(y_pred)), 'int32')) 

def errorblocks(y_true, y_pred):
    #return K.sum(K.not_equal(y_true, K.round(y_pred)))
    return K.sum(K.cast(K.any(K.not_equal(y_true, K.round(y_pred)), axis=1), 'int32'))

def fullAdder(a,b,c):
    s = (a ^ b) ^ c
    c = (a & b) | (c & (a ^ b))
    return s,c

def addBool(a,b):
    if len(a) != len(b):
        raise ValueError('arrays with different length')
    k = len(a)
    s = np.zeros(k,dtype=bool)
    c = False
    for i in reversed(range(0,k)):
        s[i], c = fullAdder(a[i],b[i],c)    
    if c:
        warnings.warn("Addition overflow!")
    return s

def incBool(a):
    k = len(a)
    increment = np.hstack((np.zeros(k-1,dtype=bool), np.ones(1,dtype=bool)))
    a = addBool(a,increment)
    return a

def bitrevorder(x):
    m = np.amax(x)
    n = np.ceil(np.log2(m)).astype(int)
    for i in range(0,len(x)):
        x[i] = int('{:0{n}b}'.format(x[i],n=n)[::-1],2)  
    return x

def int2bin(x,N):
    if isinstance(x, list) or isinstance(x, np.ndarray):
        binary = np.zeros((len(x),N),dtype='bool')
        for i in range(0,len(x)):
            binary[i] = np.array([int(j) for j in bin(x[i])[2:].zfill(N)])
    else:
        binary = np.array([int(j) for j in bin(x)[2:].zfill(N)],dtype=bool)
    
    return binary

def bin2int(b):
    if isinstance(b[0], list):
        integer = np.zeros((len(b),),dtype=int)
        for i in range(0,len(b)):
            out = 0
            for bit in b[i]:
                out = (out << 1) | bit
            integer[i] = out
    elif isinstance(b, np.ndarray):
        if len(b.shape) == 1:
            out = 0
            for bit in b:
                out = (out << 1) | bit
            integer = out     
        else:
            integer = np.zeros((b.shape[0],),dtype=int)
            for i in range(0,b.shape[0]):
                out = 0
                for bit in b[i]:
                    out = (out << 1) | bit
                integer[i] = out
        
    return integer

def polarDesign(N, k, design_snr_dB):  
        
    S = 10**(design_snr_dB/10)
    z0 = np.zeros(N)

    z0[0] = np.exp(-S)
    for j in range(1,int(np.log2(N))+1):
        u = 2**j
        for t in range(0,int(u/2)):
            T = z0[t]
            z0[t] = 2*T - T**2     # upper channel
            z0[int(u/2)+t] = T**2  # lower channel
        
    # sort into increasing order
    idx = np.argsort(z0)
        
    # select k best channels
    idx = np.sort(bitrevorder(idx[0:k]))
    
    A = np.zeros(N, dtype=bool)
    A[idx] = True
        
    return A

def polarTransform(u):

    N = len(u)
    n = 1
    x = np.copy(u)
    stages = np.log2(N).astype(int)
    for s in range(0,stages):
        i = 0
        while i < N:
            for j in range(0,n):
                idx = i+j
                x[idx] = x[idx] ^ x[idx+n]
            i=i+2*n
        n=2*n
    return x