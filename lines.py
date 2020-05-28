from spectra.levels import Level
from spectr import infer

class Transition(Level):

    _class_prototypes = {
        'class':{'description':"What kind of data this is.",'kind':'str', 'infer_functions':(('self',lambda self:self.__class__.__name__),),},
        'description':{'kind':str,'description':"",},
        'notes':{'description':"Notes regarding this line.", 'kind':str, },
        'author':{'description':"Author of data or printed file", 'kind':str, },
        'reference':{'description':"", 'kind':str, },
        'date':{'description':"Date data collected or printed", 'kind':str, },
        'species':{'description':"Chemical species",
                   'infer':((('encoded',),infer.species_from_encoded_level,None),)},

        'E':{},
        
        }

    quantum_numbers = ('species','label','v','Σ','ef','J',
                        'S','Λ','LSsign','s','gu','Ihomo', 'group','term_symbol'
                        'sublevel','N','F','Ω','SR','sa','pm','g','σv','encoded')
    
    defining_quantum_numbers = ('species','label','v','Σ','ef','J')



