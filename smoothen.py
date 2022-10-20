from fenics import *
import numpy as np
import pandas as pd
import csv
import random as rnd
def smoothen(Th0,Tc0,Tr0,Dn0,D0,M0,C0,N0,A0,S1,mesh):

    #Functions on the fine mesh
    P1 = FiniteElement('P', triangle,1)
    PB = FiniteElement('B', triangle,3)
    # NEE = NodalEnrichedElement(P1, PB)
    NEE = FiniteElement('P', triangle,1)
    element1 = MixedElement([NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE])
    Mixed_Space1 = FunctionSpace(mesh, element1)

    U = Function(Mixed_Space1)
    Th,Tc,Tr,Dn,D,M,C,N,A = split(U)
    v1, v2, v3, v4, v5, v6, v7, v8, v9 = TestFunctions(Mixed_Space1)

    ############################################################################
    D_1 = 1e-3
    D_2 = 5e-6

    F1 = D_1*dot(grad(Th),grad(v1))*dx+Th*v1*dx-Th0*v1*dx\
    + D_1*dot(grad(Tc),grad(v2))*dx+Tc*v2*dx-Tc0*v2*dx\
    + D_1*dot(grad(Tr),grad(v3))*dx+Tr*v3*dx-Tr0*v3*dx\
    + D_1*dot(grad(Dn),grad(v4))*dx+Dn*v4*dx-Dn0*v4*dx\
    + D_1*dot(grad(D),grad(v5))*dx+D*v5*dx-D0*v5*dx\
    + D_1*dot(grad(M),grad(v6))*dx+M*v6*dx-M0*v6*dx\
    + D_2*dot(grad(C),grad(v7))*dx+C*v7*dx-C0*v7*dx\
    + D_2*dot(grad(N),grad(v8))*dx+N*v8*dx-N0*v8*dx\
    + D_2*dot(grad(A),grad(v9))*dx+A*v9*dx-A0*v9*dx\

    solve(F1==0,U,[])
    Th_,Tc_,Tr_,Dn_,D_,M_,C_,N_,A_= U.split()
    Th = project(Th_,S1)
    Tc = project(Tc_,S1)
    Tr = project(Tr_,S1)
    Dn = project(Dn_,S1)
    D = project(D_,S1)
    M = project(M_,S1)
    C = project(C_,S1)
    N = project(N_,S1)
    A = project(A_,S1)
    return Th,Tc,Tr,Dn,D,M,C,N,A
