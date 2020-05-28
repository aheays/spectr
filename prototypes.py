
prototypes = {

    'class':{'description':"What kind of data this is.",'kind':'str',},
    'description':{'kind':str,'description':"",},
    'notes':{'description':"Notes regarding this line.", 'kind':str, },
    'author':{'description':"Author of data or printed file", 'kind':str, },
    'reference':{'description':"", 'kind':str, },
    'date':{'description':"Date data collected or printed", 'kind':str, },
    'species':{'description':"Chemical species",},
    'E':dict(description="Level energy (cm-1)",kind=float,fmt='<14.7f'),
    'branch':dict(description="Rotational branch ΔJ.Fp.Fpp.efp.efpp", dtype='8U', cast=str, fmt='<10s'),
    'ν':dict(description="Transition wavenumber (cm-1)", kind=float, fmt='>13.6f'),

}



















