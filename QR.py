from calcmat import *
import copy
import numpy as np

#Une fonction de test est située à la fin du fichier

#Afin de réduire le nombre de calculs et de multiplications de matrices dans Householder, 
#on ne considère plus les vecteurs comme des matrices n*1 mais comme des listes
#il faut donc recoder certaines fonctions pour tenir de ce changement
def normlist(l):
	return sqrt(sum([i**2 for i in l]))

#calcule le produit scalaire de deux listes
def dotprodlist(l1,l2):
	return sum([ l1[i]*l2[i] for i in range(len(l1))])

#applique une matrice à une liste
def matbylist(A,l):
	dimA = dim(A)
	return [ sum([ A[i][k]*l[k] for k in range(dimA[1])]) for i in range(dimA[0])]

#multiplie une liste par un scalaire
def listbyreal(l,a):
	return [a*e for e in l]

#soustrait une liste à une autre
def sublist(l1,l2):
	return [l1[i]-l2[i] for i in range(len(l1))]

#calcul le produit dyadique entre deux listes
def outerprod(l1,l2):
	return [ [l1[i]*l2[j] for j in range(len(l2))] for i in range(len(l1))]

#effectue la réduction de symétrique à tridiagonale
#un exemple d'application ainsi que le pseudo code de l'algorithme 
#sont donnés dans le rapport
def householder(A):
	dimA = dim(A)
	n=dimA[0]
	diagres = [A[0][0]]
	updiagres = []
	accmat = [[0 for _ in range(n)] for _ in range(n)]
	for i in range(n):
		accmat[i][i]=1
	x = A[0][1:]
	for i in range(1,n):
		print("Householder: ", i )
		del A[0]
		for j in range(len(A)):
			del A[j][0]
		normx = normlist(x)
		if x[0]>0:
			k=normx
		else:
			k=-normx
		x[0]=k+x[0] # x devient u
		H=dotprodlist(x,x)/2
		p=listbyreal(matbylist(A,x),1/H)
		K=dotprodlist(p,x)/(2*H)
		q=sublist(p,listbyreal(x,K))
		A=submat( submat(A, outerprod(q,x) ) , outerprod(x,q) )
		updiagres.append(-k)
		diagres.append(A[0][0])
		B=[ accmat[j][i:] for j in range(n)]
		y=listbyreal(matbylist(B,x),1/H)
		B=submat(B,outerprod(y,x))
		for j in range(n):
			accmat[j][i:]=B[j]
		x = A[0][1:]
	return (diagres, updiagres,accmat)

#Calcule les coefficients de la matrice de Givens associée au couple (a,b)
def givens(a,b):
	if b==0:
		return (1,0)
	else:
		if abs(b)>abs(a):
			temp=(-a)/b
			s=1/sqrt(1+temp**2)
			c=s*temp
		else:
			temp=(-b)/a
			c=1/sqrt(1+temp**2)
			s=c*temp
		return (c,s)

#Calcule la décomposition QR d'une tridiagonale 
#Tire profit du caractère sparse des matrices de Givens
def qrdecomp(A):
	dimA = dim(A)
	n=dimA[0]
	accmat = [[0 for _ in range(n)] for _ in range(n)]
	for i in range(n):
		accmat[i][i]=1
	for k in range(0,n-1):
		(c,s)=givens(A[k][k],A[k+1][k])
		for i in range(0,n):
			a1=A[k][i]
			a2=A[k+1][i]
			A[k][i]=c*a1-s*a2
			A[k+1][i]=s*a1+c*a2
			acc1=accmat[k][i]
			acc2=accmat[k+1][i]
			accmat[k][i]=c*acc1-s*acc2
			accmat[k+1][i]=s*acc1+c*acc2
		A[k+1][k]=0
	return (transpose(accmat),A)

#Algorithme QR appliqué à A tridiagonale
def qralg(A,iter):
	dimA=dim(A)
	n=dimA[0]
	matres = A
	accmat = [[0 for _ in range(n)] for _ in range(n)]
	for i in range(n):
		accmat[i][i]=1
	for k in range(iter):
		print("Lancement decompo QR", k+1)
		(Q,R)=qrdecomp(matres)
		matres = mult(R,Q)
		accmat = mult(accmat, Q)
		print("Fin iteration QR", k+1)
	return (matres,accmat) #les vecteurs propres de A sont sur les colonnes de accmat

#Construit une matrice complète à partir de 2 vecteurs codant une tridiagonale symétrique
def totridiag(diagres, updiagres):
	n=len(diagres)
	A=[[0 for _ in range(n)] for _ in range(n)]  
	for i in range(n):
		A[i][i]=diagres[i]
	for i in range(n-1):
		A[i][i+1]=updiagres[i]
		A[i+1][i]=updiagres[i] 
	return A

#Donne les valeurs propres et les vecteurs propres d'une matrice symétrique semi-définie positive
#les valeurs propres sont naturellement classées par ordre décroissant (Théorème QR)
def eigenQR(A,iter):
	print("Debut Householder")
	C=copy.deepcopy(A)
	(diag,up,mat) = householder(C)
	print("Fin Householder")
	print("Debut QR")
	(eigenvalT,eigenvecT)=qralg(totridiag(diag,up),iter)
	print("Fin QR")
	eigenval = [eigenvalT[i][i] for i in range(dim(C)[0])]
	eigenvec= mult(mat,eigenvecT)
	return (eigenval,eigenvec)



#TESTS NUMERIQUES
#la matrice représentant le nuage des données est p*n, avec p grand devant n: grand nombre d'individus
#on travaille ensuite sur une matrices carrée symétrique semi-définie positive de taille n
n=200
p=1000
iter=100 #nombre d'itérations de l'algorithme QR

#Renvoie les 5 plus grandes valeurs propres avec QR et avec Scipy
def testQR():
	#On génère le nuage de données
	A=normalize(center(randomM(p,n))[0])[0]
	#On calcule la matrice de variance-covariance associée
	B=mult(transpose(A),A)
	(eigenval,eigenvec)=eigenQR(B,iter)
	scipyEigs = largestEig(B,5)
	print('QR: ',eigenval[0:5])
	print('Scipy:', scipyEigs[::-1])

#testQR()