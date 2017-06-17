from calcmat import *
import numpy as np

#Deux fonctions de test sont situées à la fin du fichier

#première implémentation de la méthode des puissances
#condition d'arrêt: la distance euclidienne entre les deux derniers vecteurs
#       calculés est inférieure à une précision donnée par acc
def power(A,x,acc):
	eprev=scale(x,norm(x))
	temp = mult(A,eprev)
	ecurr=scale(temp,norm(temp))
	count=0
	while dist(eprev,ecurr)>acc:
		eprev=ecurr
		temp=mult(A,temp)
		ecurr=scale(temp,norm(temp))
		count = count +1
	print("nombre d'itérations:",count)
	return ecurr

#deuxième implémentation de la méthode des puissances
#on calcule A^(2^pow) puis on l'applique à x et on renvoie un vecteur de norme 1
def power2(A,x,pow):
	x=scale(x,norm(x))
	for _ in range(pow):
		A=mult(A,A)
	e=mult(A,x)
	return scale(e,norm(e))

#implémentation de la méthode de déflation: passage aux valeurs propres inférieures
def update(A,v):
	dimA= dim(A)
	for i in range(dimA[0]):
		prodi= dotprod(v,vect2mat(A[i]))
		vtemp= [prodi*v[i][0] for i in range(dimA[1])]
		A[i]= sub(A[i],vtemp)
	return A




#TESTS NUMERIQUES
#la matrice représentant le nuage des données est p*n, avec p grand devant n: grand nombre d'individus
#on travaille ensuite sur une matrices carrée symétrique semi-définie positive de taille n
#pour les tests, on prend une précision de 10^-4
prec=0.0001
n=10
p=100
pow = 4


#TESTS DE LA PREMIERE IMPLEMENTATION
def testpower1():
	print("Première implémentation:")
	#On génère le nuage de données
	A=normalize(center(randomM(p,n))[0])[0]
	#On calcule la matrice de variance-covariance associée
	B=mult(transpose(A),A)
	#on génère un vecteur aléatoire pour initialiser la méthode des puissances
	x=randomV(n)
	#calcul des vraies valeurs des valeurs propres avec scipy
	scipyEigs = largestEig(B,2)
	#premier vecteur propre renvoyé par la méthode de la puissance
	e=power(B,x,prec)
	#si e est une bonne approximation du vecteur propre cherché, alors le vecteur Be doit être proche de (lambda)e
	#on fait apparaitre les valeurs propres empiriques calculées en divisant les coordoonées de Be par celles de e
	print("premières valeurs propres empiriques", comp(mult(B,e),e))
	#on fait les changements sur B pour pouvoir passer à la valeur propre suivante
	B=update(B,e)
	#on execute une nouvelle fois la méthode de la puissances sur la matrice modifiée
	e2=power(B,x,prec)
	#on fait apparaitre les valeurs propres empiriques obtenues
	print("deuxièmes valeurs propres empiriques", comp(mult(B,e2),e2))
	#on affiche les vraies valeurs des deux plus grandes valeurs propres de B
	print('2 premières valeurs propres Scipy:', scipyEigs[::-1])
	print('\n \n')



#TESTS DE LA DEUXIEME IMPLEMENTATION 
def testpower2():
	print("Deuxième implémentation:")
	print("calcul de A^(2^"+str(pow)+')')
	#On génère le nuage de données
	A=normalize(center(randomM(p,n))[0])[0]
	#On calcule la matrice de variance-covariance associée
	B=mult(transpose(A),A)
	#on génère un vecteur aléatoire pour initialiser la méthode des puissances
	x=randomV(n)
	#calcul des vraies valeurs des valeurs propres avec scipy
	scipyEigs = largestEig(B,2)
	#premier vecteur propre renvoyé par la méthode de la puissance
	e=power2(B,x,pow)
	#si e est une bonne approximation du vecteur propre cherché, alors le vecteur Be doit être proche de (lambda)e
	#on fait apparaitre les valeurs propres empiriques calculées en divisant les coordoonées de Be par celles de e
	print("premières valeurs propres empiriques", comp(mult(B,e),e))
	#on fait les changements sur B pour pouvoir passer à la valeur propre suivante
	B=update(B,e)
	#on execute une nouvelle fois la méthode de la puissances sur la matrice modifiée
	e2=power2(B,x,pow)
	#on fait apparaitre les valeurs propres empiriques obtenues
	print("deuxièmes valeurs propres empiriques", comp(mult(B,e2),e2))
	#on affiche les vraies valeurs des deux plus grandes valeurs propres de B
	print('2 premières valeurs propres Scipy:', scipyEigs[::-1])

testpower1()
testpower2()