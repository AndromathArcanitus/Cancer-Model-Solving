from fenics import *
import numpy as np
import pandas as pd
import csv
import random as rnd
def IC_Loc_DG(k,mesh,S1):

    #Read the data
    ICLoc = pd.read_csv('input/input/MMTV-PyMT mIHC early tumor fold out R101.csv').to_numpy()
    ############################################################################

    #Early cell numbers (from collaborators) corresponding to feat_name
    #Arg1->?   aSMA->?   CC3->N   CD11b->Dn   CD11c->D   CD31->Endothelial Cells
    #CD3->Pan T-cells   CD45->leukocytes(Mn?)    CD4->Th    CD8->Tc    CSF1R->M
    #Epcam->C    F480->M    Foxp3->Tr    Ki67->proliferation factor?
    #Ly6G->Neutrophils    MHCII->M     PDL1->A
    feat_name = ['Arg1','aSMA','CC3','CD11b','CD11c','CD31','CD3','CD45','CD4','CD8','CSF1R','Epcam','F480','Foxp3','Ki67','Ly6G','MHCII','PDL1']
    early_cell_num = np.array([85,1194,49,454,108,593,302,358,100,175,379,9351,356,13,4046,10,939,524]);    #Given by collaborators
    ############################################################################

    #Finding a thereshold for each cell.
    #These values have been transferred from matlab quantile(f_late(:,i), 1-early_cell_num(i)/size(f_late,1))
    #The reason we used Matlab is because it gives a better quantile.
    #np.quantile and Matlab quantile use two different sampling methods so their results are very different
    #No use for lines 139-150 yet
    Thd_early=[0.0795880995000000,0.0796087520000000,0.0703477220000000,0.105173750000000,\
    	      0.0845434515000000,0.0213784675000000,0.0173849660000000,0.0103555425000000,\
              0.0528803005000000,0.0208479765000000,0.0674677905000000,0.0880372985000000,\
              0.0221500940000000,0.0536739445000000,0.0699621865000000,0.103125037000000,\
              0.00868828550000000,0.0502249710000000]
    ############################################################################

    X = mesh.coordinates()
    tree = BoundingBoxTree()
    tree.build(mesh)
    ############################################################################

    #Functions on the fine mesh
    Th = Function(S1)
    Tc = Function(S1)
    Tr = Function(S1)
    Dn = Function(S1)
    D  = Function(S1)
    Mn = Function(S1)
    M  = Function(S1)
    C  = Function(S1)
    N  = Function(S1)
    A  = Function(S1)
    ############################################################################


    #Getting the fine mesh IC functions
    for i in range(np.shape(ICLoc)[0]):
        p = Point(ICLoc[i,20]/10000,ICLoc[i,21]/10000,0)
        id = tree.compute_first_entity_collision(p)
        if id>np.shape(ICLoc)[0]:
            continue
        if ICLoc[i,13]<Thd_early[11] and ICLoc[i,9]>Thd_early[7] and ICLoc[i,8]>Thd_early[6] and ICLoc[i,10]>Thd_early[8] and ICLoc[i,11]<Thd_early[9]:
            Th.vector()[id] = Th.vector().get_local()[id]+1
        if ICLoc[i,13]<Thd_early[11] and ICLoc[i,9]>Thd_early[7] and ICLoc[i,8]>Thd_early[6] and ICLoc[i,10]<Thd_early[8] and ICLoc[i,11]>Thd_early[9]:
            Tc.vector()[id] = Tc.vector().get_local()[id]+1
        # if ICLoc[i,13]<Thd_early[11] and ICLoc[i,9]>Thd_early[7] and ICLoc[i,8]>Thd_early[6] and ICLoc[i,10]>Thd_early[8] and ICLoc[i,11]<Thd_early[9] and ICLoc[i,15]>Thd_early[13]:
        if ICLoc[i,13]<Thd_early[11] and ICLoc[i,9]>Thd_early[7] and ICLoc[i,8]>Thd_early[6] and ICLoc[i,10]<Thd_early[8] and ICLoc[i,11]<Thd_early[9]:
            Tr.vector()[id] = Tr.vector().get_local()[id]+1
        if ICLoc[i,13]<Thd_early[11] and ICLoc[i,9]>Thd_early[7] and ICLoc[i,14]<Thd_early[12] and ICLoc[i,6]>Thd_early[4]:
            Dn.vector()[id] = Dn.vector().get_local()[id]+1
        if ICLoc[i,13]<Thd_early[11] and ICLoc[i,9]>Thd_early[7] and ICLoc[i,14]<Thd_early[12] and ICLoc[i,6]>Thd_early[4] and ICLoc[i,18]>Thd_early[16]:
            D.vector()[id] = D.vector().get_local()[id]+1
        ##################################################
        #We take M1 and M2 to be M
        if ICLoc[i,13]<Thd_early[11] and ICLoc[i,9]>Thd_early[7] and ICLoc[i,14]>Thd_early[12] and ICLoc[i,6]<Thd_early[4] and ICLoc[i,12]>Thd_early[10]:
            M.vector()[id] = M.vector().get_local()[id]+1
        if ICLoc[i,13]<Thd_early[11] and ICLoc[i,9]>Thd_early[7] and ICLoc[i,14]>Thd_early[12] and ICLoc[i,6]<Thd_early[4] and ICLoc[i,12]<Thd_early[10] and ICLoc[i,18]>Thd_early[16]:
            M.vector()[id] = M.vector().get_local()[id]+1
        ##################################################
        if ICLoc[i,13]>Thd_early[11] and ICLoc[i,9]<Thd_early[7]:
            C.vector()[id] = C.vector().get_local()[id]+1
        if ICLoc[i,4]>Thd_early[2]:
            N.vector()[id] = N.vector().get_local()[id]+1
        if ICLoc[i,13]<Thd_early[11] and ICLoc[i,9]<Thd_early[7]:     #Adipocytes differentiate from mesenchymal cells
            A.vector()[id] = A.vector().get_local()[id]+1
        ############################################################################

    #Max values
    maxTh = max(Th.vector()[:])
    maxTc = max(Tc.vector()[:])
    maxTr = max(Tr.vector()[:])
    maxDn = max(Dn.vector()[:])
    maxD = max(D.vector()[:])
    maxM = max(M.vector()[:])
    maxC = max(C.vector()[:])
    maxN = max(N.vector()[:])
    maxA = max(A.vector()[:])
    ############################################################################


    #Save the IC plots
    File('DG_IC/Th.pvd')<<Th
    File('DG_IC/Tc.pvd')<<Tc
    File('DG_IC/Tr.pvd')<<Tr
    File('DG_IC/Dn.pvd')<<Dn
    File('DG_IC/D.pvd')<<D
    File('DG_IC/M.pvd')<<M
    File('DG_IC/C.pvd')<<C
    File('DG_IC/N.pvd')<<N
    File('DG_IC/A.pvd')<<A
    ############################################################################

    #Svaing the max values for dimensionalizing and post-processing
    maxes = [maxTh,maxTc,maxTr,maxDn,maxD,maxM,maxC,maxN,maxA]
    c=csv.writer(open('DG_IC/max_ICs.csv',"w"))
    c.writerow(maxes)
    del c
    ############################################################################

    return Th,Tc,Tr,Dn,D,M,C,N,A,maxTh,maxTc,maxTr,maxDn,maxD,maxM,maxC,maxN,maxA
