prototypes = {

    ## levels
    'class' :dict(description="Dataset subclass" ,kind='str' ,infer={}) ,
    'description':dict( description="",kind=str ,infer={}) ,
    'notes' :dict(description="Notes regarding this line" , kind=str ,infer={}) ,
    'author' :dict(description="Author of data or printed file" ,kind=str ,infer={}) ,
    'reference' :dict(description="Published reference" ,kind=str ,infer={}) ,
    'date' :dict(description="Date data collected or printed" ,kind=str ,infer={}) ,
    'species' :dict(description="Chemical species" ,kind=str ,infer={}) ,
    'E' :dict(description="Level energy (cm-1)" ,kind=float ,fmt='<14.7f' ,infer={}) ,
    'J' :dict(description="Total angular momentum quantum number excluding nuclear spin" , kind=float,infer={}) ,
    'g' :dict(description="Level degeneracy including nuclear spin statistics" , kind=int ,infer={}) ,
    'pm' :dict(description="Total inversion symmetry" ,kind=int ,infer={}) ,
    'Γ' :dict(description="Total natural linewidth of level or transition (cm-1 FWHM)" , kind=float, fmt='<10.5g', infer={('τ',):lambda τ: 5.309e-12/τ,}),
    'label' :dict(description="Label of electronic state", kind=str,infer={}),
    'v' :dict(description="Vibrational quantum number", kind=int,infer={}),
    'J' :dict(description="Total angular momentum quantum number excluding nuclear spin", kind=float,infer={}),
    'N' :dict(description="Angular momentum excluding nuclear and electronic spin", kind=float, infer={('J','SR') : lambda J,SR: J-SR,}),
    'S' :dict(description="Total electronic spin quantum number", kind=float,infer={}),
    'Λ' :dict(description="Total orbital angular momentum aligned with internuclear axis", kind=int,infer={}),
    'LSsign' :dict(description="For Λ>0 states this is the sign of the spin-orbit interacting energy. For Λ=0 states this is the sign of λ-B. In either case it controls whether the lowest Σ level is at the highest or lower energy.", kind=int,infer={}),
    's' :dict(description="s=1 for Σ- states and 0 for all other states", kind=int,infer={}),
    'σv' :dict(description="Symmetry with respect to σv reflection.", kind=int,infer={}),
    'gu' :dict(description="Gerade / ungerade symmetry.", kint=int,infer={}),
    'sa' :dict(description="Symmetry with respect to nuclear exchange, s=symmetric, a=antisymmetric.", kind=int,infer={}),
    'ef' :dict(description="e/f symmetry", kind=int,infer={}),
    'Fi' :dict(description="Spin multiplet index", kind=int,infer={}),
    'Ω' :dict(description="Absolute value of total angular momentum aligned with internuclear axis", kind=float,infer={}),
    'Σ' :dict(description="Signed spin projection in the direction of Λ, or strictly positive if Λ=0", kind=float,infer={}),
    'SR' :dict(description="Signed projection of spin angular momentum onto the molecule-fixed z (rotation) axis.", kind=float,infer={}),
    'Eref' :dict(description="Reference point of energy scale relative to potential-energy minimum (cm-1).", kind=float,infer={() :lambda : 0.,}),
    'partition_source':dict(description="Data source for computing partition function, 'self' or 'database' (default).", kind=str, infer={('database',): lambda : 'database',}),

    ## transitions
    'levels_class':dict(description="What Dataset subclass of Levels this is a transition between",kind='object',infer={}),
    'branch':dict(description="Rotational branch ΔJ.Fu.Fl.efu.efl", dtype='8U', cast=str, fmt='<10s'),
    'ν':dict(description="Transition wavenumber (cm-1)", kind=float, fmt='>13.6f', infer={}),
    'ΓD':dict(description="Gaussian Doppler width (cm-1 FWHM)",kind=float,fmt='<10.5g', infer={}),
    'f':dict(description="Line f-value (dimensionless)",kind=float,fmt='<10.5e',infer={}),


}
