# -*- coding: utf-8 -*-
"""
python 3Grap_generator.py K N


@author: ivan
"""

import sys
import numpy as np



if len(sys.argv) != 3:
    print "2 args needed,", len(sys.argv), "given"
else :    
    
    N = int(sys.argv[1])
    K = int(sys.argv[2])
    
    if K > 0 :
        N_Ver = 2 * N * K
        N_Blocks = N_Ver/N
        
        A = np.eye(N, k = 1) + np.eye(N, k= -1)
        A[0, N-1] = 1
        A[N-1, 0] = 1
        
        B = np.fliplr(np.eye(N))
        
        C = np.eye(N) + np.eye(N, k = -1)
        C[0,N-1] = 1
        
        
        S1 = np.zeros((2*N, 2*N))
        S1[0:N, N:2*N] = np.eye(N)
        
        S2 = np.zeros((2*N, 2*N), dtype=int)
        S2[N:2*N, 0:N] = C
        
        G = np.zeros((2*N*K, 2*N*K))
        
        # Generating half matrix
        for i in range(K):
            for j in range(i,K):
                # Diago
                if i == j:
                    G[2*N*i : 2*N*(i+1), 2*N*i : 2*N*(i+1)] = S1
                elif (j == (i + 1)):
                    G[2*N*i : 2*N*(i+1), 2*N*j : 2*N*(j+1)] = S2
        
        G[0:N,0:N] = A
        G[(2 * K * N) - N :(2 * K * N), (2 * K * N) - N :(2 * K * N)] = A
        
        # Fill the other half
        for i in range(2*N*K):
            for j in range(i,2*N*K):
                G[j,i] = G[i,j]
        
        correct = True
        n_nods = 2*N*K
    
    elif (K == 0) and (N % 2 == 0):
        
        G = np.zeros((N, N)) + np.eye(N, k = 1) + np.eye(N, k = -1)
        G[N-1,0] = 1
        G[0, N-1]= 1
        print G
        
        # Fixing links
        half = N/2
        for i in range(N):
            G[i, (half + i) % N] = 1
            
        correct = True
        n_nods = N
    else:
        print "Invalid input"
        
    if correct :
    
        print
        print "The folowing graph has been generated :"
        
        #printing 
        for i in range(n_nods):
            aux = ""
            for j in range(n_nods):
                aux += str(int(G[i,j])) + " "
                if ((j+1) % N) == 0:
                    aux += " "
            if ((i) % N) == 0:
                print
            print aux
        correct = True
        
        
        
        out = "3Graph_" + str(N) + "_" + str(K) + ".txt"
        flat = G.flatten()
        
        text = ""
        for number in flat: text += (str(int(number)) + '\n')
        
        myFile = open(out, 'w')
        myFile.write(text)
        myFile.close() 
        
        print
        print "Graph named", out, "with", 2 * N * K, "nodes, successfully saved!"
