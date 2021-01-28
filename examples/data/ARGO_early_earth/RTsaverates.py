import numpy as np
import sys

def save(direction,itime,p,h):
	if direction == 'Up' and itime == 0: fout = open('./out/Reactions/Kup.dat','w')
	if direction == 'Down' and itime == 0: fout = open('./out/Reactions/Kdown.dat','w')
	if direction == 'Up' and itime > 0: fout = open('./out/Reactions/Kup.dat','a')
	if direction == 'Down' and itime > 0: fout = open('./out/Reactions/Kdown.dat','a')
	if direction != 'Up' and direction != 'Down':
		print 'Error! Bad direction for Kout files!'
		sys.exit()	

	N, k = np.loadtxt('Kout.dat',skiprows=2,usecols=(0,1),unpack=True)

	N = N.astype(int)

	if itime == 0:
		fout.write('Rate file, ')
		fout.write('Direction = ' + direction)
		fout.write('\n\n   p[bar]   h[cm]')
		for i in range(len(N)):
			fout.write('  R' + str(N[i]))
	
	fout.write('\n  ')
	fout.write('%.3e' % p)
	fout.write('  ')
	fout.write('%.3e' % h)
	for i in range(len(N)):
		fout.write('  ')
		fout.write('%.3e' % k[i])

	fout.close()

#save('Up',0,1e-3,5e6)
