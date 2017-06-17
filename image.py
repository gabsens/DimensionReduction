from calcmat import *
from QR import eigenQR,dotprodlist,listbyreal
import numpy as np

#commenter l'appel à testQR() dans QR.py avant de lancer ce script
#sinon la fonction testQR() est appellée

from scipy import misc
#importation de lena.jpg dans l'array image
image=misc.imread('images/lena.jpg')
#copie de l'array image dans une matrice A
A=[]
for i in range(image.shape[0]):
        l=[]
        for j in range(image.shape[0]):
                l.append(image[i][j])
        A.append(l)
#centrage du nuage (le centre est stocké dans grav pour la reconstruction)
(A,grav)=center(A)
#nuage est ensuite normé (les ecart-types de chaque ligne sont stockés dans norm pour la reconstruction)
(A,norm)=normalize(A)
#calcul de la matrice de variance-covariance
B=mult(transpose(A),A)
#execution de QR
(eigenval,eigenvec)=eigenQR(B,30)

#Reconstruction du nuage à partir des composantes principales
def rebuild(A,eigenvec,k):
	eigenvec=[ line[:k] for line in eigenvec]
	eigenvec=transpose(eigenvec)
	scalmat=[ [dotprodlist(line,line2) for line2 in eigenvec] for line in A]
	return mult(scalmat, eigenvec)

#Recentre le nuage 
def uncenter(A,grav):
	dimA=dim(A)
	return [add(A[i],grav) for i in range(dimA[0])]

#Dénormalise le nuage
def unnormalize(A,norm):
	dimA=dim(A)
	return [listbyreal(A[i],norm[i]) for i in range(dimA[0])]

#production des images reconstruites	
for i in (2,5,10,20,30,40,50,60,70):
	AA=uncenter( unnormalize( rebuild(A,eigenvec,i),norm ),grav )
	misc.imsave('images/lena'+str(i)+'eigen.jpg',AA)
