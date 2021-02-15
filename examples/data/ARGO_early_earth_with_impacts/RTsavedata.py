import os
import sys

name=raw_input('Name of folder: ')
Network=raw_input('Name of network file [default=\'Stand-2020October\']: ')
Network = Network or 'Stand-2020October'
Profile=raw_input('Name of profile file [leave blank and profile will not be saved]: ')
Profile = Profile or 'NoProfile'
CondInit=raw_input('Name of cond_initial file [defualt=\'cond_initial.dat\']: ')
CondInit = CondInit or 'cond_initial.dat'

print 'Moving files to: ./out/'+name

os.chdir('./out')
os.system('mkdir '+name)
os.chdir('./'+name)
os.system('mkdir Figures Plot Reactions')
os.system('mv ../../log.out ./')
os.system('cp ../../'+Network+' ./Reactions/')
#os.system('mv ../Figures/* Figures/')
os.system('mv ../Plot/* Plot/')
os.system('mv ../Reactions/* Reactions/')
os.system('cp ../depth* ./')
os.system('cp ../old-depth-* ./')
os.system('cp ../lifetime.dat ./')
os.system('cp ../surfacelife.dat ./')
os.system('cp ../../reservoir.dat ./')
os.system('cp ../../outgas.dat ./')
os.system('cp ../../cond_initial/'+CondInit+' ./')

if Profile != 'NoProfile':
    os.system('cp ../../profiles/'+Profile+' ./')
