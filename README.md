# Spring 2017 student project at ENSAE

This repository includes the code, report and slides for the Python project I completed in Spring of 2017. 
The topic was dimensionality reduction through PCA (Principal Component Analysis)  and NMF (Non-negative Matrix Factorization).

## PCA

I chose to implement PCA from the ground up without resorting to any existing scientific libraries 
(that is to say no Numpy, Scipy, etc...). The actual goal is to find the eigenvectors associated with the greatest eigenvalues of a positive semi-definite matrix. The resulting code boils down to Householder reduction and QR iteration 
(which is obtained through Givens rotations). The implementation is 100% Vanilla Python, hence much slower than the actual
Fortran routine upon which Scipy is based. It is still robust and will yield good results in large dimensions (provided you're 
willing to wait long enough...)

## NMF

I didn't have time to implement NMF by myself and just used what is available in `sklearn.decomposition`. I
reproduced the results of Lee & Seung in their foundational 1999 paper regarding Grolier encyclopedia articles. 

## Files description
* `report.pdf` contains theoretical explanations behind the algorithms as well as practical results
* `slides.pdf` contains the oral presentation (final grade: 15/20)
* `calcmat.py` contains matrix calculus-related functions
* `power.py` contains a naive approach to PCA with power iteration
* `QR.py` contains my actual PCA implementation 
* `image.py` applies PCA to compression of Lena's face
* `NMF.py` applies NMF to words in encyclopedia articles

## Disclaimer

The code and the report are written in French, but an English translation should follow shortly.
