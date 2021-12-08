"""
classname: What kind of Dataset subclass to load into

load_tweak: function which modified loaded hitran Dataset in place
"""

import warnings

species_data={}

species_data['¹H₂¹⁶O'] = {'classname':'lines.Generic',}
species_data['¹H₂¹⁸O'] = {'classname':'lines.Generic',}
species_data['¹H₂¹⁷O'] = {'classname':'lines.Generic',}
species_data['¹H²H¹⁶O'] = {'classname':'lines.Generic',}
species_data['¹H²H¹⁸O'] = {'classname':'lines.Generic',}
species_data['¹H²H¹⁷O'] = {'classname':'lines.Generic',}
species_data['²H₂¹⁶O'] = {'classname':'lines.Generic',}

species_data['¹²C¹⁶O₂'] = {'classname':'lines.Generic',}
species_data['¹³C¹⁶O₂'] = {'classname':'lines.Generic',}
species_data['¹⁶O¹²C¹⁸O'] = {'classname':'lines.Generic',}
species_data['¹⁶O¹²C¹⁷O'] = {'classname':'lines.Generic',}
species_data['¹⁶O¹³C¹⁸O'] = {'classname':'lines.Generic',}
species_data['¹⁶O¹³C¹⁷O'] = {'classname':'lines.Generic',}
species_data['¹²C¹⁸O₂'] = {'classname':'lines.Generic',}
species_data['¹⁷O¹²C¹⁸O'] = {'classname':'lines.Generic',}
species_data['¹²C¹⁷O₂'] = {'classname':'lines.Generic',}
species_data['¹³C¹⁸O₂'] = {'classname':'lines.Generic',}
species_data['¹⁸O¹³C¹⁷O'] = {'classname':'lines.Generic',}
species_data['¹³C¹⁷O₂'] = {'classname':'lines.Generic',}
species_data['¹⁶O₃'] = {'classname':'lines.Generic',}
species_data['¹⁶O¹⁶O¹⁸O'] = {'classname':'lines.Generic',}
species_data['¹⁶O¹⁸O¹⁶O'] = {'classname':'lines.Generic',}
species_data['¹⁶O¹⁶O¹⁷O'] = {'classname':'lines.Generic',}
species_data['¹⁶O¹⁷O¹⁶O'] = {'classname':'lines.Generic',}
species_data['¹⁴N₂¹⁶O'] = {'classname':'lines.Generic',}
species_data['¹⁴N¹⁵N¹⁶O'] = {'classname':'lines.Generic',}
species_data['¹⁵N¹⁴N¹⁶O'] = {'classname':'lines.Generic',}
species_data['¹⁴N₂¹⁸O'] = {'classname':'lines.Generic',}
species_data['¹⁴N₂¹⁷O'] = {'classname':'lines.Generic',}
species_data['¹²C¹⁶O'] = {'classname':'lines.Diatomic',}
species_data['¹³C¹⁶O'] = {'classname':'lines.Diatomic',}
species_data['¹²C¹⁸O'] = {'classname':'lines.Diatomic',}
species_data['¹²C¹⁷O'] = {'classname':'lines.Diatomic',}
species_data['¹³C¹⁸O'] = {'classname':'lines.Diatomic',}
species_data['¹³C¹⁷O'] = {'classname':'lines.Diatomic',}
species_data['¹²C¹H₄'] = {'classname':'lines.Generic',}
species_data['¹³C¹H₄'] = {'classname':'lines.Generic',}
species_data['¹²C¹H₃²H'] = {'classname':'lines.Generic',}
species_data['¹³C¹H₃²H'] = {'classname':'lines.Generic',}
species_data['¹⁶O₂'] = {'classname':'lines.Diatomic',}
species_data['¹⁶O¹⁸O'] = {'classname':'lines.Diatomic',}
species_data['¹⁶O¹⁷O'] = {'classname':'lines.Diatomic',}
species_data['¹⁴N¹⁶O'] = {'classname':'lines.Diatomic',}
species_data['¹⁵N¹⁶O'] = {'classname':'lines.Diatomic',}
species_data['¹⁴N¹⁸O'] = {'classname':'lines.Diatomic',}
species_data['³²S¹⁶O₂'] = {'classname':'lines.Generic',}
species_data['³⁴S¹⁶O₂'] = {'classname':'lines.Generic',}
species_data['¹⁴N¹⁶O₂'] = {'classname':'lines.Generic',}
species_data['¹⁴N¹H₃'] = {'classname':'lines.Generic',}
species_data['¹⁵N¹H₃'] = {'classname':'lines.Generic',}
species_data['¹H¹⁴N¹⁶O₃'] = {'classname':'lines.Generic',}
species_data['¹H¹⁵N¹⁶O₃'] = {'classname':'lines.Generic',}
species_data['¹⁶O¹H'] = {'classname':'lines.Diatomic',}
species_data['¹⁸O¹H'] = {'classname':'lines.Diatomic',}
species_data['¹⁶O²H'] = {'classname':'lines.Diatomic',}
species_data['¹H¹⁹F'] = {'classname':'lines.Diatomic',}
species_data['²H¹⁹F'] = {'classname':'lines.Diatomic',}
species_data['¹H³⁵Cl'] = {'classname':'lines.Diatomic',}
species_data['¹H³⁷Cl'] = {'classname':'lines.Diatomic',}
species_data['²H³⁵Cl'] = {'classname':'lines.Diatomic',}
species_data['²H³⁷Cl'] = {'classname':'lines.Diatomic',}
species_data['¹H⁷⁹Br'] = {'classname':'lines.Diatomic',}
species_data['¹H⁸¹Br'] = {'classname':'lines.Diatomic',}
species_data['²H⁷⁹Br'] = {'classname':'lines.Diatomic',}
species_data['²H⁸¹Br'] = {'classname':'lines.Diatomic',}
species_data['¹H¹²⁷I'] = {'classname':'lines.Diatomic',}
species_data['²H¹²⁷I'] = {'classname':'lines.Diatomic',}
species_data['³⁵Cl¹⁶O'] = {'classname':'lines.Diatomic',}
species_data['³⁷Cl¹⁶O'] = {'classname':'lines.Diatomic',}
species_data['¹⁶O¹²C³²S'] = {'classname':'lines.Generic',}
species_data['¹⁶O¹²C³⁴S'] = {'classname':'lines.Generic',}
species_data['¹⁶O¹³C³²S'] = {'classname':'lines.Generic',}
species_data['¹⁶O¹²C³³S'] = {'classname':'lines.Generic',}
species_data['¹⁸O¹²C³²S'] = {'classname':'lines.Generic',}
species_data['¹H₂¹²C¹⁶O'] = {'classname':'lines.Generic',}
species_data['¹H₂¹³C¹⁶O'] = {'classname':'lines.Generic',}
species_data['¹H₂¹²C¹⁸O'] = {'classname':'lines.Generic',}
species_data['¹H¹⁶O³⁵Cl'] = {'classname':'lines.Generic',}
species_data['¹H¹⁶O³⁷Cl'] = {'classname':'lines.Generic',}
species_data['¹⁴N₂'] = {'classname':'lines.Diatomic',}
species_data['¹⁴N¹⁵N'] = {'classname':'lines.Diatomic',}
species_data['¹H¹²C¹⁴N'] = {'classname':'lines.Generic',}
species_data['¹H¹³C¹⁴N'] = {'classname':'lines.Generic',}
species_data['¹H¹²C¹⁵N'] = {'classname':'lines.Generic',}
species_data['¹²C¹H₃³⁵Cl'] = {'classname':'lines.Generic',}
species_data['¹²C¹H₃³⁷Cl'] = {'classname':'lines.Generic',}
species_data['¹H₂¹⁶O₂'] = {'classname':'lines.Generic',}
species_data['¹²C₂¹H₂'] = {'classname':'lines.Generic',}
species_data['¹H¹²C¹³C¹H'] = {'classname':'lines.Generic',}
species_data['¹H¹²C¹²C²H'] = {'classname':'lines.Generic',}
species_data['¹²C₂¹H₆'] = {'classname':'lines.Generic',}
species_data['¹²C¹H₃¹³C¹H₃'] = {'classname':'lines.Generic',}
species_data['³¹P¹H₃'] = {'classname':'lines.Generic',}
species_data['¹²C¹⁶O¹⁹F₂'] = {'classname':'lines.Generic',}
species_data['¹³C¹⁶O¹⁹F₂'] = {'classname':'lines.Generic',}
species_data['³²S¹⁹F₆'] = {'classname':'lines.Generic',}
species_data['¹H₂³²S'] = {'classname':'lines.Generic',}
species_data['¹H₂³⁴S'] = {'classname':'lines.Generic',}
species_data['¹H₂³³S'] = {'classname':'lines.Generic',}
species_data['¹H¹²C¹⁶O¹⁶O¹H'] = {'classname':'lines.Generic',}
species_data['¹H¹⁶O₂'] = {'classname':'lines.Generic',}
species_data['¹⁶O'] = {'classname':'lines.Generic',}
species_data['³⁵Cl¹⁶O¹⁴N¹⁶O₂'] = {'classname':'lines.Generic',}
species_data['³⁷Cl¹⁶O¹⁴N¹⁶O₂'] = {'classname':'lines.Generic',}
species_data['¹⁴N¹⁶O+'] = {'classname':'lines.Diatomic',}
species_data['¹H¹⁶O⁷⁹Br'] = {'classname':'lines.Generic',}
species_data['¹H¹⁶O⁸¹Br'] = {'classname':'lines.Generic',}
species_data['¹²C₂¹H₄'] = {'classname':'lines.Generic',}
species_data['¹²C¹H₂¹³C¹H₂'] = {'classname':'lines.Generic',}
species_data['¹²C¹H₃¹⁶O¹H'] = {'classname':'lines.Generic',}
species_data['¹²C¹H₃⁷⁹Br'] = {'classname':'lines.Generic',}
species_data['¹²C¹H₃⁸¹Br'] = {'classname':'lines.Generic',}
species_data['¹²C¹H₃¹²C¹⁴N'] = {'classname':'lines.Generic',}
species_data['¹²C¹⁹F₄'] = {'classname':'lines.Generic',}
species_data['¹²C₄¹H₂'] = {'classname':'lines.Generic',}
species_data['¹H¹²C₃¹⁴N'] = {'classname':'lines.Generic',}
species_data['¹H₂'] = {'classname':'lines.Diatomic',}
species_data['¹H²H'] = {'classname':'lines.Diatomic',}
species_data['¹²C³²S'] = {'classname':'lines.Diatomic',}
species_data['¹²C³⁴S'] = {'classname':'lines.Diatomic',}
species_data['¹³C³²S'] = {'classname':'lines.Diatomic',}
species_data['¹²C³³S'] = {'classname':'lines.Diatomic',}
species_data['³²S¹⁶O₃'] = {'classname':'lines.Generic',}
species_data['¹²C₂¹⁴N₂'] = {'classname':'lines.Generic',}
species_data['¹²C¹⁶O³⁵Cl₂'] = {'classname':'lines.Generic',}
species_data['¹²C¹⁶O³⁵Cl³⁷Cl'] = {'classname':'lines.LinearTriatomic',}

## CS2
def _f(line):
    """Quantum number tweak for CS₂, ignore isotope symmetry breaking. Probably wrong."""
    warnings.warn("CS₂ HITRAN quantum-number load tweaks are probably wrong, shouldn't matter for standard purposes.")
    line['label_u'] = line['label_l'] = 'X'
    line['S_u'] = line['S_l'] = 0
    line['s_u'] = line['s_l'] = 0
    line['Λ_u'] = line['l2_u']
    line['Λ_l'] = line['l2_l']
    ## set ef=+1 undefinined for most levels except odd-J (0,1,1) level
    line['ef_u'] = line['ef_l'] = 0
    line.set('ef_u','value',-1,ν2_u=1,even_J_u=False)
    line.set('ef_u','value',+1,ν2_u=1,even_J_u= True)
    line.set('ef_l','value',-1,ν2_l=1,even_J_l=False)
    line.set('ef_l','value',+1,ν2_l=1,even_J_l= True)

species_data['¹²C³²S₂'] = {'classname':'lines.LinearTriatomic','load_tweak':_f,}
species_data['³²S¹²C³⁴S'] = species_data['¹²C³²S₂']
species_data['³²S¹²C³³S'] = species_data['¹²C³²S₂']
species_data['¹³C³²S₂'] = species_data['¹²C³²S₂']

