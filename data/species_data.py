
metadata = {
    'E0'                            : { 'description' : 'Energy of the lowest physical energy level relative the equilibrium energy.','units':'cm-1',},
    'Inuclear'                      : { 'description' : 'Nuclear spin angulat momentum for homonuclear diatomic species'            ,         }        ,
    'characteristic_infrared_bands' : { 'description' : 'Wavenumber bands of the main infrared bands of this species'               , 'units' : 'cm-1' ,  }  ,
    'characteristic_infrared_lines' : { 'description' : 'Wavenumber bands of a few of the strongest infrared lines of this species' , 'units' : 'cm-1' ,  }  ,
    'charge'                        : {'description'  : 'Total charge of species.'},
    'chemical_formula'              : { 'description' : 'common to all isotopologues'                                               ,         }        ,
    'elements'                      : {'description'  : 'Sorted tuple of element'},
    'formula'                       : {'description'  : 'Chemical formula including any isotope.'},
    'is_isotopologue'               : {'description'  : 'True if species contains any mass information, else False.'},
    'isotopes'                      : {'description'  : 'Sorted tuple of isotopes (mass_number,element)'},
    'isotopologue_formula'          : {'description'  : 'Chemical species with isotope information, assuming most abundand isotope where this information is missing.'},
    'isotopologue_ratio'            : { 'description' : 'Natural abundance of this isotopologue'                                    ,         }        ,
    'mass'                          : { 'description' : 'Total mass'                                                                , 'units' : 'amu'  ,  }  ,
    'nelectrons'                    : {'description'  : 'Number of electrons'},
    'nnuclei'                       : {'description'  : 'Number of nuclei'},
    'nuclei'                        : {'description'  : 'List of (mass_number,element,multiplicity) matching the species formula.'},
    'point_group'                   : { 'description' : ''                                                                          ,         }        ,
    'prefix'                        : {'description'  : 'Structure information prefixing species formula.'},
    'reduced_mass'                  : { 'description' : 'Reduced mass'                                                              , 'units' : 'amu'  ,  }  ,
}


data = {

    'CS₂'     :     {
        'chemical_formula'  : 'CS₂',
        'mass'              : 76,
        'characteristic_infrared_bands': [[1500,1560],],
        'characteristic_infrared_lines': [[1530,1550],],
        'classname': 'LinearTriatom',
    },

    '¹²C³²S₂' :     {
        'chemical_formula'  : 'CS₂',
        'point_group'       : 'D∞h',
    },

    '¹²C³³S₂' :     {
        'chemical_formula'  : 'CS₂',
        'point_group'       : 'D∞h',
    },

    '¹²C³⁴S₂' :     {
        'chemical_formula'  : 'CS₂',
        'point_group'       : 'D∞h',
    },

    '¹³C³²S₂' :     {
        'chemical_formula'  : 'CS₂',
        'point_group'       : 'D∞h',
    },

    '¹³C³³S₂' :     {
        'chemical_formula'  : 'CS₂',
        'point_group'       : 'D∞h',
    },

    '¹³C³⁴S₂' :     {
        'chemical_formula'  : 'CS₂',
        'point_group'       : 'D∞h',
    },

    '³²S¹²C³⁴S':     {
        'chemical_formula'  : 'CS₂',
        'point_group'       : 'C∞v',
    },

    '³²S¹²C³³S':     {
        'chemical_formula'  : 'CS₂',
        'point_group'       : 'C∞v',
    },

    '³³S¹²C³⁴S':     {
        'chemical_formula'  : 'CS₂',
        'point_group'       : 'C∞v',
    },

    '³²S¹³C³⁴S':     {
        'chemical_formula'  : 'CS₂',
        'point_group'       : 'C∞v',
    },

    '³²S¹³C³³S':     {
        'chemical_formula'  : 'CS₂',
        'point_group'       : 'C∞v',
    },

    '³³S¹³C³⁴S':     {
        'chemical_formula'  : 'CS₂',
        'point_group'       : 'C∞v',
    },

    '¹²C'     :     {
        'chemical_formula'  : 'C',
        'mass'              : 12.0107,
        'point_group'       : 'K',
    },

    '¹⁶O'     :     {
        'chemical_formula'  : 'O',
        'mass'              : 15.994915,
        'point_group'       : 'K',
        'classname': 'Atom',
    },

    '¹⁷O'     :     {
        'chemical_formula'  : 'O',
        'mass'              : 16.999131,
        'point_group'       : 'K',
    },

    '¹⁸O'     :     {
        'chemical_formula'  : 'O',
        'mass'              : 17.999159,
        'point_group'       : 'K',
    },

    '³²S'     :     {
        'chemical_formula'  : 'S',
        'mass'              : 31.9720707,
        'point_group'       : 'K',
    },

    '³³S'     :     {
        'chemical_formula'  : 'S',
        'mass'              : 32.97145843,
        'point_group'       : 'K',
    },

    '³⁴S'     :     {
        'chemical_formula'  : 'S',
        'mass'              : 33.96786665,
        'point_group'       : 'K',
    },

    '³⁶S'     :     {
        'chemical_formula'  : 'S',
        'mass'              : 35.96708062,
        'point_group'       : 'K',
    },

    '¹⁴N'     :     {
        'chemical_formula'  : 'N',
        'mass'              : 14.003074,
        'point_group'       : 'K',
    },

    '¹⁵N'     :     {
        'chemical_formula'  : 'N',
        'mass'              : 15.0001088982,
        'point_group'       : 'K',
    },

    'Na'      :     {
        'chemical_formula'  : 'Na',
        'mass'              : 22.9898,
        'point_group'       : 'K',
    },

    'Ar'      :     {
        'chemical_formula'  : 'Ar',
        'mass'              : 39.948,
        'point_group'       : 'K',
    },

    'CO'  :     {
        'chemical_formula'  : 'CO',
        'characteristic_infrared_bands': [[1980,2280], [4150,4350],],
        'characteristic_infrared_lines': [[2160,2180],],
        'classname': 'Diatom',
    },

    '¹²C¹⁶O'  :     {
        'chemical_formula'  : 'CO',
        'mass'              : 28.0,
        'isotopologue_ratio': 0.986544,
        'isotopologue_ratio:ref': 'NIST',
        'reduced_mass'      : 6.8562,
        'point_group'       : 'C∞v',
        'E0'              : 1081.7756,
        'E0:ref'          : 'coxon2004',
        'E0'                : 1081.775631,
    },

    '¹²C¹⁷O'  :     {
        'chemical_formula'  : 'CO',
        'mass'              : 29.0,
        'isotopologue_ratio': 0.000367867,
        'isotopologue_ratio:ref': 'NIST',
        'reduced_mass'      : 7.0343,
        'point_group'       : 'C∞v',
        'E0'              : 1068.031,
        'E0:ref'          : 'coxon2004',
        'E0'                : 1068.031015,
    },

    '¹²C¹⁸O'  :     {
        'chemical_formula'  : 'CO',
        'mass'              : 30.0,
        'isotopologue_ratio': 0.00197822,
        'isotopologue_ratio:ref': 'NIST',
        'reduced_mass'      : 7.1998,
        'point_group'       : 'C∞v',
        'E0'              : 1055.7172,
        'E0:ref'          : 'coxon2004',
        'E0'                : 1055.717274,
    },

    '¹³C¹⁶O'  :     {
        'chemical_formula'  : 'CO',
        'mass'              : 29.0,
        'isotopologue_ratio': 0.0110836,
        'isotopologue_ratio:ref': 'NIST',
        'reduced_mass'      : 7.1724,
        'point_group'       : 'C∞v',
        'E0'              : 1057.7268,
        'E0:ref'          : 'coxon2004',
        'E0'                : 1057.726807,
    },

    '¹³C¹⁷O'  :     {
        'chemical_formula'  : 'CO',
        'mass'              : 30.0,
        'isotopologue_ratio': 4.13292e-06,
        'isotopologue_ratio:ref': 'NIST',
        'reduced_mass'      : 7.3675,
        'point_group'       : 'C∞v',
        'E0'              : 1043.6628,
        'E0:ref'          : 'coxon2004',
        'E0'                : 1043.662809,
    },

    '¹³C¹⁸O'  :     {
        'chemical_formula'  : 'CO',
        'mass'              : 31.0,
        'isotopologue_ratio': 2.2225e-05,
        'isotopologue_ratio:ref': 'NIST',
        'reduced_mass'      : 7.5493,
        'point_group'       : 'C∞v',
        'E0'              : 1031.0556,
        'E0:ref'          : 'coxon2004',
        'E0'                : 1031.055619,
    },

    '¹⁴C¹⁶O'  :     {
        'chemical_formula'  : 'CO',
        'mass'              : 29.998,
        'isotopologue_ratio': 0,
        'reduced_mass'      : 7.46648,
        'point_group'       : 'C∞v',
        'E0'              : 1036.7443,
        'E0:ref'          : 'coxon2004',
        'E0'                : 1036.744345,
    },

    '¹⁴C¹⁷O'  :     {
        'chemical_formula'  : 'CO',
        'mass'              : 31.002,
        'isotopologue_ratio': 0,
        'reduced_mass'      : 7.6782,
        'point_group'       : 'C∞v',
        'E0'              : 1022.3893,
        'E0:ref'          : 'coxon2004',
        'E0'                : 1022.389332,
    },

    '¹⁴C¹⁸O'  :     {
        'chemical_formula'  : 'CO',
        'mass'              : 32.002,
        'isotopologue_ratio': 0,
        'reduced_mass'      : 7.8758,
        'point_group'       : 'C∞v',
        'E0'              : 1009.5143,
        'E0:ref'          : 'coxon2004',
        'E0'                : 1009.514309,
    },

    '¹H₂'     :     {
        'chemical_formula'  : 'H₂',
        'mass'              : 2.01588,
        'reduced_mass'      : 0.50397,
        'point_group'       : 'D∞h',
        'Inuclear'          : 0.5,
    },

    '²H₂'     :     {
        'chemical_formula'  : 'H₂',
        'mass'              : 4.028,
        'reduced_mass'      : 1.007,
        'point_group'       : 'D∞h',
        'Inuclear'          : 1,
    },

    'D₂'      :     {
        'chemical_formula'  : 'H₂',
        'mass'              : 4.028,
        'reduced_mass'      : 1.007,
        'point_group'       : 'D∞h',
        'Inuclear'          : 1,
    },

    'HD'      :     {
        'chemical_formula'  : 'H₂',
        'mass'              : 3.02194,
        'reduced_mass'      : 0.671751,
        'point_group'       : 'C∞v',
    },

    '¹H²H'    :     {
        'chemical_formula'  : 'H₂',
        'mass'              : 3.02194,
        'reduced_mass'      : 0.671751,
        'point_group'       : 'C∞v',
    },

    '¹⁴N₂'    :     {
        'chemical_formula'  : 'N₂',
        'mass'              : 28.006147,
        'reduced_mass'      : 7.0015372,
        'point_group'       : 'D∞h',
        'Inuclear'          : 1,
        'E0'              : 1175.7,
    },

    '¹⁴N¹⁵N'  :     {
        'chemical_formula'  : 'N₂',
        'mass'              : 29.0032,
        'reduced_mass'      : 7.242227222,
        'point_group'       : 'C∞v',
        'E0'              : 1156.091,
    },

    '¹⁵N₂'    :     {
        'chemical_formula'  : 'N₂',
        'mass'              : 30.0002,
        'reduced_mass'      : 7.50005465,
        'point_group'       : 'D∞h',
        'Inuclear'          : 0.5,
        'E0'              : 1135.103,
    },

    'OH'      :     {
        'chemical_formula'  : 'OH',
        'mass'              : 17.0073,
        'reduced_mass'      : 0.948169,
        'point_group'       : 'C∞v',
        'classname': 'Diatom',
    },

    'OD'      :     {
        'chemical_formula'  : 'OH',
        'mass'              : 18.0135,
        'reduced_mass'      : 1.7889,
        'point_group'       : 'C∞v',
    },

    'OH+'     :     {
        'chemical_formula'  : 'OH+',
        'mass'              : 17.003288,
        'reduced_mass'      : 0.95,
        'point_group'       : 'C∞v',
    },

    'CS'      :     {
        'chemical_formula'  : 'CS',
        'mass'              : 44.0767,
        'reduced_mass'      : 8.73784,
        'point_group'       : 'C∞v',
        'characteristic_infrared_bands': [[1200,1350],],
    },

    'NO'  :     {
        'chemical_formula'  : 'NO',
        'characteristic_infrared_bands': [[1750,2000],],
        'characteristic_infrared_lines': [[1830,1850],],
        'classname': 'Diatom',
    },

    'H₂S'  :     {
        'chemical_formula'  : 'H₂S',
        'characteristic_infrared_bands': [[1000,1600],[3500,4100]],
        'characteristic_infrared_lines': [[1275,1300],],
    },

    '¹⁴N¹⁶O'  :     {
        'chemical_formula'  : 'NO',
        'mass'              : 29.99799,
        'reduced_mass'      : 7.4664331,
        'point_group'       : 'C∞v',
    },

    '¹²C¹⁴N'  :     {
        'chemical_formula'  : 'CN',
        'mass'              : 26.018,
        'reduced_mass'      : 6.4661,
        'point_group'       : 'C∞v',
    },

    '³²S¹⁶O'  :     {
        'chemical_formula'  : 'SO',
        'mass'              : 47.966984,
        'reduced_mass'      : 10.6613,
        'point_group'       : 'C∞v',
        'Inuclear'          : 0,
        'E0'              : 573.79105,
    },

    '³³S¹⁶O'  :     {
        'chemical_formula'  : 'SO',
        'mass'              : 48.9664,
        'reduced_mass'      : 10.7702,
        'point_group'       : 'C∞v',
        'Inuclear'          : 0,
        'E0'              : 570.89075,
    },

    '³⁴S¹⁶O'  :     {
        'chemical_formula'  : 'SO',
        'mass'              : 49.9628,
        'reduced_mass'      : 10.8744,
        'point_group'       : 'C∞v',
        'Inuclear'          : 0,
        'E0'              : 568.15644,
    },

    '³⁶S¹⁶O'  :     {
        'chemical_formula'  : 'SO',
        'mass'              : 51.962,
        'reduced_mass'      : 11.0714,
        'point_group'       : 'C∞v',
        'Inuclear'          : 0,
        'E0'              : 563.09265,
    },

    '³²S¹⁷O'  :     {
        'chemical_formula'  : 'SO',
        'mass'              : 48.9712,
        'reduced_mass'      : 11.0983,
        'point_group'       : 'C∞v',
        'Inuclear'          : 0,
    },

    '³³S¹⁷O'  :     {
        'chemical_formula'  : 'SO',
        'mass'              : 49.9706,
        'reduced_mass'      : 11.2163,
        'point_group'       : 'C∞v',
        'Inuclear'          : 0,
    },

    '³⁴S¹⁷O'  :     {
        'chemical_formula'  : 'SO',
        'mass'              : 50.967,
        'reduced_mass'      : 11.3294,
        'point_group'       : 'C∞v',
        'Inuclear'          : 0,
    },

    '³⁶S¹⁷O'  :     {
        'chemical_formula'  : 'SO',
        'mass'              : 52.9662,
        'reduced_mass'      : 11.5434,
        'point_group'       : 'C∞v',
        'Inuclear'          : 0,
    },

    '³²S¹⁸O'  :     {
        'chemical_formula'  : 'SO',
        'mass'              : 49.9712,
        'reduced_mass'      : 11.516,
        'point_group'       : 'C∞v',
        'Inuclear'          : 0,
    },

    '³³S¹⁸O'  :     {
        'chemical_formula'  : 'SO',
        'mass'              : 50.9706,
        'reduced_mass'      : 11.6431,
        'point_group'       : 'C∞v',
        'Inuclear'          : 0,
    },

    '³⁴S¹⁸O'  :     {
        'chemical_formula'  : 'SO',
        'mass'              : 51.967,
        'reduced_mass'      : 11.765,
        'point_group'       : 'C∞v',
        'Inuclear'          : 0,
    },

    '³⁶S¹⁸O'  :     {
        'chemical_formula'  : 'SO',
        'mass'              : 53.9662,
        'reduced_mass'      : 11.996,
        'point_group'       : 'C∞v',
        'Inuclear'          : 0,
    },

    '¹H³⁵Cl'  :     {
        'chemical_formula'  : 'HCl',
        'mass'              : 35.9768,
        'reduced_mass'      : 0.979701,
        'point_group'       : 'C∞v',
    },

    '¹H³⁷Cl'  :     {
        'chemical_formula'  : 'HCl',
        'mass'              : 38,
        'point_group'       : 'C∞v',
    },

    '²H³⁵Cl'  :     {
        'chemical_formula'  : 'HCl',
        'mass'              : 37,
        'point_group'       : 'C∞v',
    },

    '²H³⁷Cl'  :     {
        'chemical_formula'  : 'HCl',
        'mass'              : 39,
        'point_group'       : 'C∞v',
    },

    'HI'      :     {
        'chemical_formula'  : 'HI',
        'mass'              : 127.912,
        'reduced_mass'      : 0.999998,
        'point_group'       : 'C∞v',
    },

    'HBr'     :     {
        'chemical_formula'  : 'HBr',
        'mass'              : 80.9119,
        'reduced_mass'      : 0.995384,
        'point_group'       : 'C∞v',
    },

    '¹⁶O₂'    :     {
        'chemical_formula'  : 'O₂',
        'mass'              : 31.9988,
        'reduced_mass'      : 7.9997,
        'point_group'       : 'D∞h',
        'Inuclear'          : 0,
    },

    '³²S₂'    :     {
        'chemical_formula'  : 'S₂',
        'mass'              : 63.944141,
        'reduced_mass'      : 15.9860364,
        'point_group'       : 'D∞h',
        'Inuclear'          : 0,
    },

    'NO₂'     :     {
        'chemical_formula'  : 'NO₂',
        'mass'              : 46.0,
        'characteristic_infrared_bands': [[1540,1660],],
        'characteristic_infrared_lines': [[1590,1610],],
    },

    'CO₂'     :     {
        'chemical_formula'  : 'CO₂',
        'mass'              : 44.0,
        'characteristic_infrared_bands': [[2200,2400], [3550,3750], [4800,5150],] ,
        'characteristic_infrared_lines': [[2315,2320],],
    },

    '¹²C¹⁶O₂' :     {
        'chemical_formula'  : 'CO₂',
        'mass'              : 44.0,
    },

    '¹³C¹⁶O₂' :     {
        'chemical_formula'  : 'CO₂',
        'mass'              : 45.0,
    },

    '¹²C¹⁸O₂' :     {
        'chemical_formula'  : 'CO₂',
        'mass'              : 48.0,
    },

    '¹H₂¹⁶O'  :     {
        'chemical_formula'  : 'H₂O',
        'mass'              : 18.0,
        'characteristic_infrared_bands': [[1400,1750],[3800,4000],],
        'characteristic_infrared_lines': [[1552,1580],],
    },
    
    '¹H₂¹⁸O'  :     {
        'chemical_formula'  : 'H₂O',
        'mass'              : 18.0,
        'characteristic_infrared_bands': [[1400,1750],[3800,4000],],
        'characteristic_infrared_lines': [[1552,1580],],
    },
    
    '¹H₂¹⁷O'  :     {
        'chemical_formula'  : 'H₂O',
        'mass'              : 18.0,
        'characteristic_infrared_bands': [[1400,1750],[3800,4000],],
        'characteristic_infrared_lines': [[1552,1580],],
    },
    
    '¹H²H¹⁶O'  :     {
        'chemical_formula'  : 'H₂O',
        'mass'              : 18.0,
        'characteristic_infrared_bands': [[1400,1750],[3800,4000],],
        'characteristic_infrared_lines': [[1552,1580],],
    },
    
    '¹H²H¹⁸O'  :     {
        'chemical_formula'  : 'H₂O',
        'mass'              : 18.0,
        'characteristic_infrared_bands': [[1400,1750],[3800,4000],],
        'characteristic_infrared_lines': [[1552,1580],],
    },
    
    '¹H²H¹⁷O'  :     {
        'chemical_formula'  : 'H₂O',
        'mass'              : 18.0,
        'characteristic_infrared_bands': [[1400,1750],[3800,4000],],
        'characteristic_infrared_lines': [[1552,1580],],
    },
    
    '²H₂¹⁶O'  :     {
        'chemical_formula'  : 'H₂O',
        'mass'              : 18.0,
        'characteristic_infrared_bands': [[1400,1750],[3800,4000],],
        'characteristic_infrared_lines': [[1552,1580],],
    },

    'CH₄'     :     {
        'chemical_formula'  : 'CH₄',
        'mass'              : 16.0,
        'characteristic_infrared_bands': [[1200,1400],[2800,3200],],
        'characteristic_infrared_lines': [[3010,3020],],
    },

    '¹²CH₄'   :     {
        'chemical_formula'  : 'CH₄',
        'mass'              : 16.0,
    },

    '¹²C¹H₄'  :     {
        'chemical_formula'  : 'CH₄',
        'mass'              : 16.0,
    },

    '¹³C¹H₄'  :     {
        'chemical_formula'  : 'CH₄',
        'mass'              : 17.0,
    },

    '¹²C¹H₃²H'  :     {
        'chemical_formula'  : 'CH₄',
        'mass'              : 17.0,
    },
    
    '¹³C¹H₃²H'  :     {
        'chemical_formula'  : 'CH₄',
        'mass'              : 17.0,
    },

    'SO₂'     :     {
        'chemical_formula'  : 'SO₂',
        'mass'              : 63.9619,
        'characteristic_infrared_bands': [[1050,1400], [2450,2550],],
        'characteristic_infrared_lines': [[1350,1370],],
    },

    'S¹⁷O₂'   :     {
        'chemical_formula'  : 'SO₂',
        'mass'              : 65.9703,
    },

    'S¹⁸O₂'   :     {
        'chemical_formula'  : 'SO₂',
        'mass'              : 67.9704,
    },

    '³³SO₂'   :     {
        'chemical_formula'  : 'SO₂',
        'mass'              : 64.9613,
    },

    '³⁴SO₂'   :     {
        'chemical_formula'  : 'SO₂',
        'mass'              : 65.9577,
    },

    '³⁶SO₂'   :     {
        'chemical_formula'  : 'SO₂',
        'mass'              : 67.9569,
    },

    '³²S¹⁶O₂' :     {
        'chemical_formula'  : 'SO₂',
        'mass'              : 63.9619,
    },

    '³²S¹⁷O₂' :     {
        'chemical_formula'  : 'SO₂',
        'mass'              : 65.9703,
    },

    '³²S¹⁸O₂' :     {
        'chemical_formula'  : 'SO₂',
        'mass'              : 67.9704,
    },

    '³³S¹⁶O₂' :     {
        'chemical_formula'  : 'SO₂',
        'mass'              : 64.9613,
    },

    '³³S¹⁷O₂' :     {
        'chemical_formula'  : 'SO₂',
        'mass'              : 66.9697,
    },

    '³³S¹⁸O₂' :     {
        'chemical_formula'  : 'SO₂',
        'mass'              : 68.9698,
    },

    '³⁴S¹⁶O₂' :     {
        'chemical_formula'  : 'SO₂',
        'mass'              : 65.9577,
    },

    '³⁴S¹⁷O₂' :     {
        'chemical_formula'  : 'SO₂',
        'mass'              : 67.9661,
    },

    '³⁴S¹⁸O₂' :     {
        'chemical_formula'  : 'SO₂',
        'mass'              : 69.9662,
    },

    '³⁶S¹⁶O₂' :     {
        'chemical_formula'  : 'SO₂',
        'mass'              : 67.9569,
    },

    '³⁶S¹⁷O₂' :     {
        'chemical_formula'  : 'SO₂',
        'mass'              : 69.9653,
    },

    '³⁶S¹⁸O₂' :     {
        'chemical_formula'  : 'SO₂',
        'mass'              : 71.9654,
    },


    'CH₃Cl'   :     {
        'chemical_formula'  : 'CH₃Cl',
        'mass'              : 49.992329,
    },

    '¹²C¹H³⁵Cl':     {
        'chemical_formula'  : 'CH₃Cl',
        'mass'              : 49.992329,
    },

    '¹²C¹H³⁷Cl':     {
        'chemical_formula'  : 'CH₃Cl',
        'mass'              : 52,
    },

    'C₂H₂'    :     {
        'chemical_formula'  : 'C₂H₂',
        'characteristic_infrared_bands': [[600,850], [3200,3400],],
        'characteristic_infrared_lines': [[3255,3260],],
    },
    
    '¹²C₂¹H₂'    :     {
        'chemical_formula'  : 'C₂H₂',
        'characteristic_infrared_bands': [[600,850], [3200,3400],],
        'characteristic_infrared_lines': [[3255,3260],],
    },
    
    '¹H¹²C¹³C¹H'    :     {
        'chemical_formula'  : 'C₂H₂',
        'characteristic_infrared_bands': [[600,850], [3200,3400],],
        'characteristic_infrared_lines': [[3255,3260],],
    },
    
    '¹H¹²C¹²C²H'    :     {
        'chemical_formula'  : 'C₂H₂',
        'characteristic_infrared_bands': [[600,850], [3200,3400],],
        'characteristic_infrared_lines': [[3255,3260],],
    },

    'C₂H₄'    :     {
        'chemical_formula'  : 'C₂H₄',
        'mass'              : 28.0313,
    },

    'C₂H₆'    :     {
        'chemical_formula'  : 'C₂H₆',
        'mass'              : 30,
        'characteristic_infrared_bands': [[2850,3100]],
        'characteristic_infrared_lines': [[2975,2995]],
    },

    'HCOOH'    :     {
        'chemical_formula'  : 'HCOOH',
        'characteristic_infrared_bands': [[1000,1200],[1690,1850]],
        'characteristic_infrared_lines': [[1770,1780]],
    },

    '¹H¹²C¹⁴N'     :     {
        'chemical_formula'  : 'HCN',
        'mass'              : 27.010899,
        'characteristic_infrared_bands': [[3200,3400],],
        'characteristic_infrared_lines': [[3325,3350],],
    },
    
    '¹H¹³C¹⁴N'     :     {
        'chemical_formula'  : 'HCN',
        'mass'              : 27.010899,
        'characteristic_infrared_bands': [[3200,3400],],
        'characteristic_infrared_lines': [[3325,3350],],
    },
    
    '¹H¹²C¹⁵N'     :     {
        'chemical_formula'  : 'HCN',
        'mass'              : 27.010899,
        'characteristic_infrared_bands': [[3200,3400],],
        'characteristic_infrared_lines': [[3325,3350],],
    },

    'NH₃'     :     {
        'chemical_formula'  : 'NH₃',
        'characteristic_infrared_bands': [[800,1200],],
        'characteristic_infrared_lines': [[950,970],],
    },

    'OCS'     :     {
        'chemical_formula'  : 'OCS',
        'mass'              : 60,
        'characteristic_infrared_bands': [[2000,2100],],
        'characteristic_infrared_lines': [[2070,2080],],
    },

    '¹⁶O¹²C³²S':     {
        'chemical_formula'  : 'OCS',
        'mass'              : 60,
    },

    '¹⁶O¹²C³⁴S':     {
        'chemical_formula'  : 'OCS',
        'mass'              : 62,
    },

    'N₂O'     :     {
        'point_group': 'C∞v',
        'classname': 'Linear',
        'chemical_formula'  : 'N₂O',
        'mass'              : 48.0323,
        'characteristic_infrared_bands': [[2175,2270],],
        'characteristic_infrared_lines': [[2199,2210],],
    },

    '¹⁴N₂¹⁶O'     :     {
        'chemical_formula'  : 'N₂O',
    },

    '¹⁴N¹⁵N¹⁶O'     :     {
        'chemical_formula'  : 'N₂O',
    },
    '¹⁵N¹⁴N¹⁶O'     :     {
        'chemical_formula'  : 'N₂O',
    },
    '¹⁴N₂¹⁸O'     :     {
        'chemical_formula'  : 'N₂O',
    },
    '¹⁴N₂¹⁷O'     :     {
        'chemical_formula'  : 'N₂O',
    },

    'C₆H₆': {
        'characteristic_infrared_bands': [[1413, 1588], [2933, 3225]],
        'characteristic_infrared_lines': [[2933, 3225]],
        
        },

    'CH₃¹⁸OOH'     :     {
        'chemical_formula'  : 'CH₃OOH',
    },

    'CH₃¹⁷OOH'     :     {
        'chemical_formula'  : 'CH₃OOH',
    },

}

