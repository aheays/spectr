import os
import re

# fin = open('./Stand-2020October','r')
# fout = open('../Stand-2020October','w')

fin = open('./Stand-2020October_impacts','r')
fout = open('../Stand-2020October_impacts','w')

i = 1
j = 0
k = 1

oldtype = 'A'

for line in fin:
	j += 1
	element = line.split()
	newtype = element[-3]
	if newtype != oldtype: i = 1
	if i < 10: line2 = line.replace(' XX ','   '+str(i)+' ')
	if i >= 10 and i < 100: line2 = line.replace(' XX ', '  '+str(i)+' ')
	if i >=100 and i < 1000: line2 = line.replace(' XX ', ' '+str(i)+' ')
	if i >=1000: line2 = line.replace(' XX ', str(i)+' ')
	if newtype == 'A' or newtype == 'D' or newtype == 'F' or newtype == 'V' or newtype == 'Z' or newtype == 'L' or newtype == 'U' or newtype == 'GL' or newtype == 'EV' or newtype == 'DP' or newtype == 'BA' or newtype == 'GO' or newtype == 'TR' or newtype == 'RA' or newtype == 'AI' or newtype == 'AS' or newtype == 'ID':
		i += 1
		if newtype == 'A': line3 = line2.replace(' A ', ' 7 ')
		if newtype == 'D': line3 = line2.replace(' D ', ' 2 ')
		if newtype == 'F': line3 = line2.replace(' F ', '14 ')
		if newtype == 'V': line3 = line2.replace(' V ', '13 ')
		if newtype == 'Z': line3 = line2.replace(' Z ', ' 1 ')
		if newtype == 'L': line3 = line2.replace(' L ', ' 3 ')
		if newtype == 'U': line3 = line2.replace(' U ', '10 ')
		if newtype == 'GL': line3 = line2.replace(' GL ', '99  ')
		if newtype == 'EV': line3 = line2.replace(' EV ', '17  ')
		if newtype == 'DP': line3 = line2.replace(' DP ', '18  ')
		if newtype == 'BA': line3 = line2.replace(' BA ', '66  ')
		if newtype == 'GO': line3 = line2.replace(' GO ', '67  ')
		if newtype == 'TR': line3 = line2.replace(' TR ', '88  ')
		if newtype == 'RA': line3 = line2.replace(' RA ', '8   ')
		if newtype == 'AI': line3 = line2.replace(' AI ', '96  ')
		if newtype == 'AS': line3 = line2.replace(' AS ', '98  ')
		if newtype == 'ID': line3 = line2.replace(' ID ', '44  ')
	if newtype == 'B' or newtype == 'D' or newtype == 'G' or newtype == 'I' or newtype == 'Q' or newtype == 'R' or newtype == 'T' or newtype == 'AA' or newtype == 'BB' or newtype == 'BI' or newtype == 'BS':
		if k == 2:
			k = 0
			i += 1
		k += 1
		if newtype == 'B': line3 = line2.replace(' B ', ' 0 ')
		if newtype == 'I': line3 = line2.replace(' I ', ' 6 ')
		if newtype == 'G': line3 = line2.replace(' G ', '15 ')
		if newtype == 'Q': line3 = line2.replace(' Q ', '11 ')
		if newtype == 'R': line3 = line2.replace(' R ', '16 ')
		if newtype == 'T': line3 = line2.replace(' T ', ' 5 ')
		if newtype == 'AA': line3 = line2.replace(' AA ', '19  ')
		if newtype == 'BB': line3 = line2.replace(' BB ', '20  ')
		if newtype == 'BI': line3 = line2.replace(' BI ', '95  ')
		if newtype == 'BS': line3 = line2.replace(' BS ', '97  ')
		
	oldtype = newtype

	if j < 10: line4 = line3.replace('*XXXX*','*   '+str(j)+'*')
	if j >= 10 and j < 100: line4 = line3.replace('*XXXX*','*  '+str(j)+'*')
	if j >= 100 and j < 1000: line4 = line3.replace('*XXXX*','* '+str(j)+'*')
	if j >= 1000: line4 = line3.replace('*XXXX*','*'+str(j)+'*')

	fout.write(line4)
	


	
fin.close()
fout.close()
