# Spring 2017 student project at ENSAE

This repository includes the code, report and slides for the Python project I completed in Spring of 2017. 
The topic was dimensionality reduction through PCA (Principal Component Analysi)  and NMF (Non-negative Matrix Factorization).

## PCA

I chose to implement PCA from the ground up without resorting to any existing scientific libraries 
(that is to say no Numpy, Scipy, etc...). The resulting code boils down to Householder reduction and QR iteration 
(which is obtained through Givens rotations). The implementation is 100% Vanilla Python, hence much slower than the actual
Fortran routine upon which Scipy is based. It is still robust and will yield good results in large dimensions (provided you're 
willing to wait long enough...)

## NMF

I didn't have time to implement NMF by myself and just used what is available in `sklearn.decomposition`. I merely 
reproduced the results of Lee & Seung regarding Grolier encyclopedia articles. 

## Disclaimer

The code and the report is written French, but an English translation should follow shortly.
