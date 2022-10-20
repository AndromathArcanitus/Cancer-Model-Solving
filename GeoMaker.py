from fenics import *
import math
import os
import platform
from subprocess import call
def GeoMaker(MeshSize,mesh,file_name,Refine,j):
    if platform.system() == "Darwin":
        Gmsh_str = "/Applications/Gmsh.app/Contents/MacOS/gmsh"
    else:
        Gmsh_str = "gmsh"
        dolfin_convert_str = "dolfin-convert"


    lc = str(MeshSize)
    center = [0.0,0.0]
    bmesh = BoundaryMesh(mesh, "exterior", True)
    Bxytmp = bmesh.coordinates()
    Theta = [math.atan2(Bxytmp[i,1]-center[1],Bxytmp[i,0]-center[0]) for i in range(len(Bxytmp))]
    X = [x for _,x in sorted(zip(Theta,Bxytmp[:,0]))]
    Y = [y for _,y in sorted(zip(Theta,Bxytmp[:,1]))]
    NumberOfPoint = str(len(Bxytmp))
    #Sections = int(len(Bxytmp)/4)
    lst='{'
    with open(file_name+'.geo', 'w') as the_file:
        #the_file.write('SetFactory("OpenCASCADE");\n')
        the_file.write('\nlc ='+lc+';\n')
        the_file.write('\nr ='+Refine+';\n')
        the_file.write('\nn ='+NumberOfPoint+';\n')
        for c in range(len(Bxytmp)):
                 idx = str(c+1)
                 lst +=idx+','
                 x = str(X[c])
                 y = str(Y[c])
                 z = str(0)

                 the_file.write('//+\nPoint('+idx+')={'+x+','+y+','+z+'};\n')
        the_file.write('\ns1 = newreg;\n')
        the_file.write('\nSpline(s1) = '+lst+'1};\n')
        the_file.write('\nLine Loop(1) = {1};\n')
        the_file.write('\nPlane Surface(1) = {1};\n')
        the_file.write('\nField[1] = Distance;\nField[1].NNodesByEdge = 500;\nField[1].EdgesList = {s1};\n')
        the_file.write('\nField[2] = Threshold;\nField[2].IField = 1;\nField[2].LcMin = lc/r;\nField[2].LcMax = lc;\nField[2].DistMin = 0.002;\nField[2].DistMax = 0.005;\n')



        the_file.write('\nPhysical Line(1) = {1};\n')
        the_file.write('\nPhysical Surface(2) = {1};\n')
        the_file.write('\nBackground Field = 2;\n')
        the_file.write('\nMesh.CharacteristicLengthExtendFromBoundary = 0;')


        the_file.close()
        #if MPI.rank(MPI.comm_world) == 0: #add this if parallel
        call([Gmsh_str, "-v", "0", "-2", file_name+'.geo'])
        call([dolfin_convert_str, file_name+'.msh', file_name+'.xml'])
