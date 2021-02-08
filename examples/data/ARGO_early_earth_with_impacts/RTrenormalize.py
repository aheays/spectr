import sys
import numpy 

def Renormalize():
    fin = open('./out/depth.dat','r')
    
    header = ''
    data = []
    species = []
    
    count = 0
    smoothing = 2
    
    for line in fin:
        if count < 3: header += line
        if count == 2: species = line.split()
        if count >= 3: data.append(line.split())
        count += 1
    
    fin.close()

    
    data = numpy.array(data,dtype=float)
    
    #Renormalize

    nocount = []

    for i in range(len(species)):
        if species[i] == 'h' or species[i] == 'f+': nocount.append(i)

    for i in range(len(data)):
        NConstant = numpy.sum(data[i,6:])
        for j in range(len(nocount)):
            NConstant -= data[i,nocount[j]]
        data[i,6:] = numpy.multiply(NConstant**-1,data[i,6:])

    fout = open('./out/depth.dat','w')
    
    fout.write(header)
    for i in range(len(data)):
        fout.write('%.3e' % data[i,0])
        fout.write('  ')
        fout.write('%.1f' % data[i,1])
        fout.write('  ')
        for j in range(2,len(data[i])):
            fout.write('%.3e' % data[i,j])
            fout.write('  ')
        fout.write('\n')
    
    fout.close()

