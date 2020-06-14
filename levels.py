import itertools
from copy import deepcopy

from spectr.dataset import Dataset

class Levels(Dataset):
    """A generic level."""

    _prototypes = {
        'class':dict(description="What kind of data this is.",kind='str',infer={}),
        'description':dict(kind=str,description="",infer={}),
        'notes':dict(description="Notes regarding this line.", kind=str,infer={}),
        'author':dict(description="Author of data or printed file",kind=str,infer={}),
        'reference':dict(description="",kind=str,infer={}),
        'date':dict(description="Date data collected or printed",kind=str,infer={}),
        'species':dict(description="Chemical species",kind=str,infer={}),
        'E':dict(description="Level energy (cm-1)",kind=float,fmt='<14.7f',infer={}),
        'J':dict(description="Total angular momentum quantum number excluding nuclear spin", kind=float,infer={}),
        'g':dict(description="Level degeneracy including nuclear spin statistics.", kind=int,infer={}),
        'pm':dict(description="Total inversion symmetry.",kind=int,infer={}),
        'Γ':dict(description="Linewidth (cm-1 FWHM)", kind=float, fmt='<10.5g', infer={('τ',):lambda τ: 5.309e-12/τ,}),
    }

    def __init__(self,name=None,**keys_vals,):
        """Default_name is decoded to give default values. Kwargs can be
        scalar data, further default values of vector data, or if vectors
        themselves will populate data arrays."""
        Dataset.__init__(self)
        self.permit_nonprototyped_data = False
        self['class'] = type(self).__name__
        self.name = (name if name is not None else self['class'])
        for key,val in keys_vals.items():
            self[key] = val



class Cinfv(Levels):
    """Rotational levels of a C∞v diatomic molecule"""

    _prototypes = deepcopy(Levels._prototypes)
    _prototypes['label']  = dict(description="Label of electronic state", kind=str,infer={})
    _prototypes['v']  = dict(description="Vibrational quantum number", kind=int,infer={})
    _prototypes['J'] = dict(description="Total angular momentum quantum number excluding nuclear spin", kind=float,infer={})
    _prototypes['N']  = dict(description="Angular momentum excluding nuclear and electronic spin", kind=float, infer={('J','SR'): lambda J,SR: J-SR,})
    _prototypes['S']  = dict(description="Total electronic spin quantum number", kind=float,infer={})
    _prototypes['Λ']  = dict(description="Total orbital angular momentum aligned with internuclear axis", kind=int,infer={})
    _prototypes['LSsign']  = dict(description="For Λ>0 states this is the sign of the spin-orbit interacting energy. For Λ=0 states this is the sign of λ-B.  In either case it controls whether the lowest Σ level is at the highest or lower energy.", kind=int,infer={})
    _prototypes['s']  = dict(description="s=1 for Σ- states and 0 for all other states", kind=int,infer={})
    _prototypes['σv']  = dict(description="Symmetry with respect to σv reflection.", kind=int,infer={})
    _prototypes['gu']  = dict(description="Gerade / ungerade symmetry.", kint=int,infer={})
    _prototypes['sa']  = dict(description="Symmetry with respect to nuclear exchange, s=symmetric, a=antisymmetric.", kind=int,infer={})
    _prototypes['ef']  = dict(description="e/f symmetry", kind=int,infer={})
    _prototypes['F']  = dict(description="Spin multiplet index", kind=int,infer={})
    _prototypes['Ω']  = dict(description="Absolute value of total angular momentum aligned with internuclear axis", kind=float,infer={})
    _prototypes['Σ']  = dict(description="Signed spin projection in the direction of Λ, or strictly positive if Λ=0", kind=float,infer={})
    _prototypes['SR']  = dict(description="Signed projection of spin angular momentum onto the molecule-fixed z (rotation) axis.", kind=float,infer={})
    _prototypes['Eref'] = dict(description="Reference point of energy scale relative to potential-energy minimum (cm-1).", kind=float,infer={():lambda: 0.,})
    _prototypes['partition_source'] = dict(description="Data source for computing partition function, 'self' or 'database' (default).", kind=str, infer={('database',): lambda: 'database',})
    _prototypes['Γ'] = dict(description="Total linewidth (cm-1 FWHM)", kind=float,fmt='<10.5g',infer={('τ',):lambda τ: 5.309e-12/τ,}) # tau=1/2/pi/gamma/c

    ## additional infer functions
    _prototypes['g']['infer']['J'] = lambda J:2*J+1


