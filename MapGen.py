# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:47:42 2016

@author: ivan
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys


print
print "¡Welcome to the incredible TSP Map Generator!"
print "I'll create some maps for your TSP benchmarks based on some popular curves"
print

N_CIT = int(raw_input("First of all... give me the number of cities : "))
while (10 > N_CIT) or (N_CIT > 10000):
    print "This number is not allowed"
    N_CIT = int(raw_input("Give me another number : "))
print

print "Ok... let'see.."
print "Now you have to choose one of this curves :"
print
print "CURVE KEY       DESCRIPTION"
print "----------------------------------------"
print "CIRC            A circle"
print "ELLIP           An ellipse"
print "LISS12          Lissajous with parametesr 1 and 2"
print "LISS32          Lissajous with parametesr 3 and 2"
print "LISS54          Lissajous with parametesr 5 and 4"
print "EPICYC3         Epicycloid with k = 3"
print "EPICYC4         Epicycloid with k = 4"
print "EPICYC32        Epicycloid with k = 3/2"
print "EPICYC16        Epicycloid with k = -1.6"
print "EPICYC7         Epicycloid with k = -7"
print

valid_inputs = ["CIRC", "ELLIP","LISS12", "LISS32", "LISS54", "EPICYC3", "EPICYC4", "EPICYC32", "EPICYC16", "EPICYC7"]

CURVE = raw_input("Now, give me the curve's key : ")
while not CURVE in valid_inputs:
    print "I'm afraid this is not a correc answer"
    CURVE = raw_input("Try again : ")
print
print "All right!"
print

t = np.linspace(0,1,N_CIT)

if CURVE == "CIRC": # Circle
    X = np.cos(2 * np.pi * t)
    Y = np.sin(2 * np.pi * t)
elif CURVE == "LISS12": # Lissajous
    K_x = 1
    K_y = 2
    X = np.cos(K_x * 2 * np.pi * t)
    Y = np.sin(K_y * 2 * np.pi * t)
elif CURVE == "LISS32":
    K_x = 3
    K_y = 2
    X = np.cos(K_x * 2 * np.pi * t)
    Y = np.sin(K_y * 2 * np.pi * t)
elif CURVE == "LISS54":
    K_x = 5
    K_y = 4
    X = np.cos(K_x * 2 * np.pi * t)
    Y = np.sin(K_y * 2 * np.pi * t)
elif CURVE == "EPICYC3": # Hypotrochoid
    R = 3
    r = 1
    X = (R + r) * np.cos(2 * np.pi * t) - r * np.cos((R + r)/r * 2 * np.pi * t)
    Y = (R + r) * np.sin(2 * np.pi * t) - r * np.sin((R + r)/r * 2 * np.pi * t)
elif CURVE == "EPICYC4": # Hypotrochoid
    k = 4
    r = 1
    X = r * (k + 1) * np.cos(2 * np.pi * t) - r * np.cos((k + 1) * 2 * np.pi * t)
    Y = r * (k + 1) * np.sin(2 * np.pi * t) - r * np.sin((k + 1) * 2 * np.pi * t)
elif CURVE == "EPICYC32": # Hypotrochoid
    k = 3/2
    r = 1
    X = r * (k + 1) * np.cos(4 * np.pi * t) - r * np.cos((k + 1) * 4 * np.pi * t)
    Y = r * (k + 1) * np.sin(4 * np.pi * t) - r * np.sin((k + 1) * 4 * np.pi * t)
elif CURVE == "EPICYC16": # Hypotrochoid
    k = -1.6
    r = 1
    X = r * (k + 1) * np.cos(12 * np.pi * t) - r * np.cos((k + 1) * 12 * np.pi * t)
    Y = r * (k + 1) * np.sin(12 * np.pi * t) - r * np.sin((k + 1) * 12 * np.pi * t)
elif CURVE == "EPICYC7": # Hypotrochoid
    k = -7
    r = 1
    X = r * (k + 1) * np.cos(2 * np.pi * t) - r * np.cos((k + 1) * 2 * np.pi * t)
    Y = r * (k + 1) * np.sin(2 * np.pi * t) - r * np.sin((k + 1) * 2 * np.pi * t)
elif CURVE == "ELLIP":
    X = 2 * np.cos(2 * np.pi * t)
    Y = np.sin(2 * np.pi * t)

    
out = "myTPScurve_" + str(N_CIT) + "_" + CURVE + ".tsp"
text = "NODE_COORD_SECTION" + '\n'

for i in range(N_CIT): text += str(i + 1) + ' ' + (str(X[i]) + ' ' + str(Y[i]) + '\n')
text+='EOF'

myFile = open(out, 'w')
myFile.write(text)
myFile.close() 

print "Map successfully saved!"
answ = raw_input("Do you want to plot it? [Y/N] : ")

if answ in ['Y', 'y', 'yes', 'Yes']:
    #plt.plot(X,Y)
    plt.plot(X,Y,'ro')
    plt.show()
