from scipy import sparse
from sklearn.decomposition import NMF
import scipy
import numpy as np

mat=[ [0 for _ in range(30991)] for _ in range(15276)]
print("fin creation matrice")
with open("nmf/grolier15276.csv") as infile:
    for line in infile:
        liste=line.split(',')
        i=int(liste[0])-1
        if len(liste)>=4:
            liste=liste[1:]
            for j in range(len(liste)//2):
                word=int(liste[2*j])-1
                occ=int(liste[2*j+1])
                mat[word][i]=occ
print("fin remplissage matrice")
A=np.array(mat)
del mat[:]
print("fin conversion en array")
nmf_model = NMF(n_components = 400, init='random',solver='cd', random_state=0,max_iter=20)
W = nmf_model.fit_transform(A).T
print("fin NMF")
for i in range(10):
    s=W[i]
    occsort = sorted(range(len(s)), key=lambda k: s[k], reverse=True)
    print(occsort[:8])

fo = open("nmf/grolier15276_words.txt", "r")
words = fo.readlines()
def f(i):
    return words[i][:-1]

from functools import map
file = open("nmf/columns.txt","w")
for i in range(400):
    s=W[i]
    occsort = sorted(range(len(s)), key=lambda k: s[k], reverse=True)
    wordslist = list(map(f,occsort))[:8]
    word = ', '.join(wordslist)
    file.write(word+'\n')
file.close() 
                
            
            
