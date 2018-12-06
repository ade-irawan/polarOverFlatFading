import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.core import Dense, Lambda
from keras.layers import Input
import matplotlib.pyplot as plt
from sigproc_tools import *

## Parameters
k = 8                                 # number of information bits
N = 16                                # code length

nb_epoch = 2**16                      # number of learning epochs
design = [512, 256, 128, 64, 32]      # the number of nodes per layer
code = 'polar'
batch_size = 256                      # size of batches for calculating the gradient
optimizer = 'adam'
loss = 'mse'                

train_SNR_Es = -2
train_sigma = np.sqrt(1/(2*10**(train_SNR_Es/10)))

# Parameters for Testing
test_batch = 10000
num_words = 1000000      # multiple of test_batch
SNR_dB = np.array(range(-10,41))

# Create all possible information words
d = np.zeros((2**k,k),dtype=bool)
for i in range(1,2**k):
    d[i]= incBool(d[i-1])

# logical vector indicating the nonfrozen bit locations 
A = polarDesign(N, k, design_snr_dB=0)  

x = np.zeros((2**k, N),dtype=bool)
u = np.zeros((2**k, N),dtype=bool)
u[:,A] = d

# Create sets of all possible codewords (codebook)
for i in range(0,2**k):
    x[i] = polarTransform(u[i])

## Creating NN Model
# Input Layer    
inputs    = Input(shape=(N,), name="Tx")
dec_input = Input(shape=(N,), name="Dec_input")
# Lambda layers
mod_out = Lambda(modulateBPSK, input_shape=(N,), output_shape=returnOutputShape, name="modulator")(inputs)
fading = Lambda(addFading, arguments={'sigma':train_sigma}, input_shape=(N,), output_shape=returnOutputShape, name="fading")(mod_out)
rx = Lambda(logLikelihoodRatio, arguments={'sigma':train_sigma}, input_shape=(N,), output_shape=returnOutputShape, name="LLR")(fading)
# Hidden (Dense) Layers
dec = Dense(design[0], activation='relu', name="dec_layer1")(dec_input)
dec = Dense(design[1], activation='relu', name="dec_layer2")(dec)
dec = Dense(design[2], activation='relu', name="dec_layer3")(dec)
# Output Layer
dec_out = Dense(k, activation="sigmoid", name="Tx_Decoded")(dec)
# Create Decoder Model
decoder = Model(inputs=dec_input, outputs=dec_out)
o = decoder(inputs=rx)
# Create The System Model
system_model = Model(inputs=inputs, outputs=o)
# Summarize layers
system_model.summary()

# Compile the decoder and the model
decoder.compile(optimizer=optimizer, loss=loss, metrics=[errorblocks])
system_model.compile(optimizer=optimizer, loss=loss, metrics=[ber])

## Train NN
system_model.fit(x, d, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=True)    
 
## Test NN    
nb_errors = np.zeros(len(SNR_dB),dtype=int)
nb_blocks = np.zeros(len(SNR_dB),dtype=int)

for i in range(0,len(SNR_dB)):

    for ii in range(0,np.round(num_words/test_batch).astype(int)):
        
        # Source
        d_test = np.random.randint(0,2,size=(test_batch,k)) 
    
        # Encoder
        x_test = np.zeros((test_batch, N),dtype=bool)
        u_test = np.zeros((test_batch, N),dtype=bool)
        u_test[:,A] = d_test
    
        for iii in range(0,test_batch):
            x_test[iii] = polarTransform(u_test[iii])
        
        # Modulator (BPSK)
        s_test = -2*x_test + 1
    
        # Fading
        h_re = np.sqrt(1/2)*np.random.standard_normal((s_test.shape[0],1))
        h_im = np.sqrt(1/2)*np.random.standard_normal((s_test.shape[0],1))
        n_re = np.sqrt(1/(2*10**(SNR_dB[i]/10)))*np.random.standard_normal(s_test.shape)
        n_im = np.sqrt(1/(2*10**(SNR_dB[i]/10)))*np.random.standard_normal(s_test.shape)
        y_test = h_re * (s_test * h_re + n_re) + h_im * (s_test * h_im + n_im)

        # LLR
        y_test = 2*y_test/(np.sqrt(1/(2*10**(SNR_dB[i]/10)))**2)

        # Decoder
        nb_errors[i] += decoder.evaluate(y_test, d_test, batch_size=test_batch, verbose=1)[1]
        nb_blocks[i] += d_test.shape[0]
        

# Plot BLER
plt.plot(SNR_dB, nb_errors/nb_blocks, marker="o")
plt.yscale('log')
plt.xlabel('SNR')
plt.ylabel('BLER')
plt.title('Fading Channel')    
plt.grid(True)
plt.axis([-10,35,10**-3,1])
plt.show()    
