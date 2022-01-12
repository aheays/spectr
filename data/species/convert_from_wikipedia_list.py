## downloaded data from  https://en.wikipedia.org/wiki/List_of_CAS_numbers_by_chemical_compound and modify slightly here

from spectr.env import *
d = dataset.load('wikipedia_list_2022-01-11.psv')
d['CAS'][d['CAS']=='nan'] = '0'
d.description = 'Downloaded from wikipedia downloaded data from  https://en.wikipedia.org/wiki/List_of_CAS_numbers_by_chemical_compound and modify slightly here on 2022-01-11'
d['unicode'] = convert.species(d['ascii'],'ascii','unicode')
d['CASint'] = [int(t.replace('-','')) for t in d['CAS']]
print( d[:10].format())
d.save('translations.psv')


# print('DEBUG: Timing initiated') ; import time; timer=time.perf_counter()
# d = dataset.load('cas_numbers.psv')
# x  = d.copy()
# x.sort('CASint')
# y  = d.copy()
# y.sort('species')
# z  = d.copy()
# z.sort('species_ascii')
# w  = d.copy()
# w.sort('CAS')
# print('DEBUG: Timing elapsed',format(time.perf_counter()-timer,'12.6f'))

