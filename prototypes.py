
prototypes = {

    ## misc
    'class':{'description':"What kind of data this is.",'kind':'str', 'infer_functions':(('self',lambda self:self.__class__.__name__),),},
    'description':{'kind':str,'description':"",},
    'notes':{'description':"Notes regarding this line.", 'kind':str, },
    'author':{'description':"Author of data or printed file", 'kind':str, },
    'reference':{'description':"", 'kind':str, },
    'date':{'description':"Date data collected or printed", 'kind':str, },

    ## level quantum numbers
    'species':{'description':"Chemical species",},


    ## level data
    'E':dict(description="Level energy (cm-1)",kind=float,fmt='<14.7f'),


}
