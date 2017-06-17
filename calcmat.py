from math import sqrt
from random import random
import numpy as np
import scipy as sc
from scipy import linalg
import warnings
import dis 

#on code une matrice par la liste de ses lignes
#une matrice est donc une liste de listes

#renvoie les dimensions d'une matrice
def dim(A):
	return (len(A),len(A[0]))

#convertit une liste en une matrice n*1
def vect2mat(l):
	return [[a] for a in l ]

#multiplication de deux matrices avec compréhension
def mult2(A,B):
	dimA, dimB = dim(A), dim(B)
	if dimA[1] == dimB[0]:
		return [ [ sum([A[i][k]*B[k][j] for k in range(dimA[1])]) for j in range(dimB[1]) ] for i in range(dimA[0]) ]
	else:
		raise ValueError('Les dimensions ne sont pas valides')

#multiplication de deux matrices via Numpy
def mult(A,B):
	return np.dot(A,B).tolist()

#multiplication de deux matrices sans compréhension
def mult3(A,B):
	dimA, dimB = dim(A), dim(B)
	if dimA[1] == dimB[0]:
		C=[]
		for i in range(dimA[0]):
			l=[]
			for j in range(dimB[1]):
				Cij= sum([A[i][k]*B[k][j] for k in range(dimA[1])])
				l.append(Cij)
			C.append(l)
		return C
	else:
		raise ValueError('Les dimensions ne sont pas valides')

#multiplication d'une matrice et d'une liste
def eval(A,x):
	return mult(A,vect2mat(x))

#calcule la norme euclidienne d'une matrice n*1
def norm(x):
	return sqrt(sum( [(x[i][0])**2 for i in range(dim(x)[0])]) )

#calcule la distance entre deux matrices n*1
def dist(x,y):
	return sqrt(sum( [(x[i][0]-y[i][0])**2 for i in range(dim(x)[0])]) )

#divise une matrice n*1 par sa norme euclidienne
def scale(A,a):
	dimA = dim(A)
	return [ [ A[i][j]/a for j in range(dimA[1])] for i in range(dimA[0])]

#divise deux matrices n*1 coordonnées par coordonnées
def comp(x,y):
	return [ x[i][0]/y[i][0] for i in range(len(x))]

#transposée d'une matrice
def transpose(A):
	dimA = dim(A)
	return [ [ A[j][i] for j in range(dimA[0])] for i in range(dimA[1])]

#renvoie une matrice aléatoire de dimensions p,n
#réalisations de la loi uniforme continue sur -1,1
def randomM(p,n):
	A=[]
	for _ in range(p):
		A.append([(2*random() -1) for _ in range(n)])
	return A

#renvoie un matrice aléatoire de dimension n*1
#réalisations de la loi uniforme continue sur -1,1
def randomV(n):
	x=[]
	for _ in range(n):
		x.append([2*random()-1])
	return x

#soustrait deux listes terme à terme
def sub(x,y):
        if len(x)==len(y):
                return [x[i]-y[i] for i in range(len(x))]
        else:
                raise ValueError('Les dimensions ne sont pas valides')

#ajoute deux listes terme à terme              
def add(x,y):
        if len(x)==len(y):
                return [x[i]+y[i] for i in range(len(x))]
        else:
                raise ValueError('Les dimensions ne sont pas valides')

#centre la matrice: retranche a chaque ligne le centre de gravité du nuage
#renvoie la matrice centrée et le centre de gravité
def center(A):
	dimA = dim(A)
	grav = [0] * dimA[1]
	for i in range(dimA[0]):
		grav = [grav[j] + A[i][j] for j in range(dimA[1])]
	grav=[grav[j]/dimA[0] for j in range(dimA[1])]
	return ([sub(A[i],grav) for i in range(dimA[0])], grav)

#divise chaque ligne de la matrice par son ecart-type
#renvoie la matrice normée et un vecteur qui contient la norme de chaque ligne de la matrice de départ
def normalize(A):
	dimA = dim(A)
	norm=[]
	for i  in range(dimA[0]):
		moyi = sum(A[i])/(dimA[1])
		vari = sum([(A[i][j]-moyi)**2 for j in range(dimA[1])])
		sdi = sqrt(vari)
		A[i] = [A[i][j]/sdi for j in range(dimA[1])]
		norm.append(sdi)
	return (A,norm)

#calcule le produit scalaire entre deux matrices n*1
def dotprod(x,y):
	dimX= dim(x)
	dimY= dim(y)
	if dimX[0]==dimY[0]:
		return sum([x[i][0]*y[i][0] for i in range(dimX[0])])
	else:
		raise ValueError('Les dimensions ne sont pas valides')

#multiplie une matrice par un réel
def multreal(A,a):
       	dimA = dim(A)
        return [ [ A[i][j]*a  for j in range(dimA[1])] for i in range(dimA[0])]

#calcule la soustraction de deux matrices
def submat(A,B):
        dimA = dim(A)
        dimB = dim(B)
        if dimA==dimB:
                return [sub(A[i],B[i]) for i in range(dimA[0])]
        else:
                raise ValueError('Les dimensions ne sont pas valides')

#renvoie les k plus grandes valeurs propres avec Scipy
def largestEig(A,k):
        dimA = dim(A)
        B=sc.matrix(A)
        return linalg.eigh(B, eigvals_only = True, eigvals=(dimA[0]-k,dimA[0]-1))


