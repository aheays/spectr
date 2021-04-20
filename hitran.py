import functools

import numpy as np


from . import lines
from . import tools
from .tools import *
from .dataset import Dataset


@tools.vectorise(cache=True)
def get_partition_function(
        species_or_isotopologue,
        temperature,
):
    """Use hapi to get a partition function.  Uses main isotopologue if
    not given."""
    from hapi import hapi
    Mol,Iso = translate_species_to_codes(species_or_isotopologue)
    return hapi.partitionSum(Mol,Iso,temperature)

def get_molparam(**match_keys_vals):
    retval = molparam.matches(**match_keys_vals)
    assert len(retval)>0,f'No molparams: {match_keys_vals=}'
    return retval

@tools.vectorise(cache=True)
def get_molparam_from_isotopologue(species_or_isotopologue):
    """Get HITEAN params data form a species in standard encoding. If
    species given returns main isotopologue."""
    if sum(i:=molparam.match(isotopologue=species_or_isotopologue)) > 0:
        assert sum(i) == 1
        return molparam.as_dict(i)
    if sum(i:=molparam.match(species=species)) > 0:
        j = np.argmax(molparam['natural_abundance'][i])
        return molparam.as_dict(tools.find(i)[j])

def translate_codes_to_species(
        species_ID,
        local_isotopologue_ID,
):
    """Turn HITRAN species and local isotoplogue codes into a standard
    species string. If no isotopologue_ID assumes the main one."""
    i = molparam.find(species_ID=species_ID,
                      local_isotopologue_ID=local_isotopologue_ID)
    return molparam['chemical_species'][i],molparam['isotopologue'][i]

def translate_species_to_codes(species_or_isotopologue):
    """Get hitran species and isotopologue codes from species name.
    Assumes primary isotopologue if not indicated."""
    if len(i:=find(molparam.match(isotopologue=species_or_isotopologue))) > 0:
        assert len(i) == 1
        i = i[0]
    elif len(i:=find(molparam.match(chemical_species=species_or_isotopologue))) > 0:
        i = i[np.argmax(molparam['natural_abundance'][i])]
    else:
        raise Exception(f"Cannot find {species_or_isotopologue=}")
    return (int(molparam['species_ID'][i]),
            int(molparam['local_isotopologue_ID'][i]))

def download_linelist(
        species, 
        νbeg,νend,
        data_directory='td',
        table_name=None,        # defaults to species
):
    from hapi import hapi
    if table_name is None:
        table_name = species
    MOL,ISO = translate_species_to_codes(species)
    mkdir(data_directory)
    hapi.db_begin(data_directory)
    hapi.fetch(table_name,int(MOL),int(ISO),νbeg,νend)

def calc_spectrum(
        species,
        data_directory,
        T,p,
        νbeg,νend,νstep,
        table_name=None,
        make_plot= True,
):
    """Plot data. Must be already downloaded with download_linelist."""
    from hapi import hapi
    if table_name is None:
        table_name = species
    MOL,ISO = translate_species_to_codes(species)
    hapi.db_begin(data_directory)
    ## calculate spectrum
    ν,coef = hapi.absorptionCoefficient_Lorentz(
        SourceTables=table_name,
        WavenumberRange=[νbeg,νend],
        WavenumberStep=νstep,
        Environment={"T":T,"p":p},)
    if make_plot:
        from . import plotting
        ax = plotting.gca()
        ax.plot(ν,coef,label=species)
        ax.set_yscale('log')
        ax.set_xlabel('Wavenumber (cm-1)')
        ax.set_ylabel('Absorption coefficient')
    return ν,coef
    


def load(filename):
    """Load HITRAN .data file into a dictionary of numpy arrays."""
    data = np.genfromtxt(
        expand_path(filename),
        dtype=[
            ('Mol',int),    # molecule code number
            ('Iso','U1'),    # isotopologue code number
            ('ν',float),    # wavenumber
            ('S',float), # spectral line intensity at 296K cm-1(molecular.cm-2), the integrated cross section at 296K
            ('A',float), # Einstein A-coefficient
            ('γair',float),  # air broadening coefficient
            ('γself',float), # self broadening coefficient
            ('E_l',float), # lower level energy
            ('nair',float), # broadening temperature dependence
            ('δair',float), # pressure shift in ari (cm-1/atm)
            ('V_u','U15'),       # Upper-state “global” quanta 
            ('V_l','U15'),      # Lower-state “global” quanta 
            ('Q_u','U15'),       # Upper-state “local” quanta  
            ('Q_l','U15'),      # Lower-state “local” quanta  
            ('Ierr',int),     # Uncertainty indices         
            ('Iref',int),     # Reference indices           
            ('asterisk','U1'), # Flag                           
            ('g_u',float),    # lower level degeneracy
            ('g_l',float),    # upper level degeneracy
        ],
        delimiter=(2,1,12,10,10,5,5,10,4,8,15,15,15,15,6,12,1,7,7), # column widths
    )
    retval = {key:data[key] for key in data.dtype.names}
    iso_translate = {'1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7,
                     '8':8, '9':9, '0':10, 'A':11, 'B':12, 'C':13, 'D':14, 'E':15,
                     'F':16, 'G':17, 'H':18, 'I':19, 'J':20,}
    retval['Iso'] = [iso_translate[t] for t in retval['Iso']]
    return retval

# def get_spectrum(
        # species_or_isotopologue,
        # νbeg,νend,
        # spectrum_type='absorption',
        # data_directory=None,
        # table_name=None,        # defaults to species
# ):
    # if data_directory is None:
        # data_directory = f'/home/heays/data/databases/HITRAN/data/{species}/'
    # if table_name is None:
        # table_name = 'linelist'
    # molecule_id = _map_species_to_codes[species]
    # isotopologue_id = 1         # could be generalised
    # mkdir_if_necessary(data_directory)
    # hapi.db_begin(data_directory)
    # print(hapi.tableList())
    # ν,coef = hapi.absorptionCoefficient_Lorentz(SourceTables=table_name)
    # if spectrum_type == 'absorption':
        # return(ν,coef)
    # elif spectrum_type == 'emission':
        # ν,radi = hapi.radianceSpectrum(ν,coef)
        # return(ν,radi)
    # else:
        # raise Exception(f'unknown spectrum type: {spectrum_type}')

@functools.lru_cache
def get_lines(species_or_isotopologue):
    """Load a preconstructed linelists.  If species_or_isotopologue is not
    an isotopologue return natural abundance."""
    l = lines.Generic(f'species_or_isotopologue')
    if species_or_isotopologue in molparam['chemical_species']:
        filename = f'~/data/databases/HITRAN/data/{species_or_isotopologue}/natural_abundance/lines.h5'
    elif species_or_isotopologue in molparam['isotopologue']:
        filename = f'~/data/databases/HITRAN/data/{species_or_isotopologue}/lines.h5'
    else:
        raise Exception(f"could not determine HITRAN linelist filename for {repr(species_or_isotopologue)}")
    l.load(filename)
    l['Zsource'] = 'HITRAN'
    return l


## downloaded from https://hitran.org/docs/iso-meta/ 2020-10-15
## I also added my own classname row -- which lines class should be used
molparam = Dataset()
molparam.load_from_string('''

species_ID| global_isotopologue_ID| local_isotopologue_ID| chemical_species| isotopologue   | AFGL_code| natural_abundance| molar_mass   | Q_296K  | gi   | dataset_classname
1         | 1                     | 1                    | H2O             | [1H]2[16O]     | 161      | 0.997317         | 18.010565    | 174.58  | 1    | lines.Generic
1         | 2                     | 2                    | H2O             | [1H]2[18O]     | 181      | 0.002000         | 20.014811    | 176.05  | 1    | lines.Generic
1         | 3                     | 3                    | H2O             | [1H]2[17O]     | 171      | 3.718840e-4      | 19.01478     | 1052.14 | 6    | lines.Generic
1         | 4                     | 4                    | H2O             | [1H][2H][16O]  | 162      | 3.106930e-4      | 19.01674     | 864.74  | 6    | lines.Generic
1         | 5                     | 5                    | H2O             | [1H][2H][18O]  | 182      | 6.230030e-7      | 21.020985    | 875.57  | 6    | lines.Generic
1         | 6                     | 6                    | H2O             | [1H][2H][17O]  | 172      | 1.158530e-7      | 20.020956    | 5226.79 | 36   | lines.Generic
1         | 129                   | 7                    | H2O             | [2H]2[16O]     | 262      | 2.419700e-8      | 20.022915    | 1027.80 | 1    | lines.Generic
2         | 7                     | 1                    | CO2             | [12C][16O]2    | 626      | 0.984204         | 43.98983     | 286.09  | 1    | lines.Generic
2         | 8                     | 2                    | CO2             | [13C][16O]2    | 636      | 0.011057         | 44.993185    | 576.64  | 2    | lines.Generic
2         | 9                     | 3                    | CO2             | [16O][12C][18O]| 628      | 0.003947         | 45.994076    | 607.81  | 1    | lines.Generic
2         | 10                    | 4                    | CO2             | [16O][12C][17O]| 627      | 7.339890e-4      | 44.994045    | 3542.61 | 6    | lines.Generic
2         | 11                    | 5                    | CO2             | [16O][13C][18O]| 638      | 4.434460e-5      | 46.997431    | 1225.46 | 2    | lines.Generic
2         | 12                    | 6                    | CO2             | [16O][13C][17O]| 637      | 8.246230e-6      | 45.9974      | 7141.32 | 12   | lines.Generic
2         | 13                    | 7                    | CO2             | [12C][18O]2    | 828      | 3.957340e-6      | 47.998322    | 323.42  | 1    | lines.Generic
2         | 14                    | 8                    | CO2             | [17O][12C][18O]| 827      | 1.471800e-6      | 46.998291    | 3766.58 | 6    | lines.Generic
2         | 121                   | 9                    | CO2             | [12C][17O]2    | 727      | 1.368470e-7      | 45.998262    | 10971.57| 1    | lines.Generic
2         | 15                    | 10                   | CO2             | [13C][18O]2    | 838      | 4.446000e-8      | 49.001675    | 652.24  | 2    | lines.Generic
2         | 120                   | 11                   | CO2             | [18O][13C][17O]| 837      | 1.653540e-8      | 48.001646    | 7595.04 | 12   | lines.Generic
2         | 122                   | 12                   | CO2             | [13C][17O]2    | 737      | 1.537500e-9      | 47.0016182378| 22120.47| 2    | lines.Generic
3         | 16                    | 1                    | O3              | [16O]3         | 666      | 0.992901         | 47.984745    | 3483.71 | 1    | lines.Generic
3         | 17                    | 2                    | O3              | [16O][16O][18O]| 668      | 0.003982         | 49.988991    | 7465.68 | 1    | lines.Generic
3         | 18                    | 3                    | O3              | [16O][18O][16O]| 686      | 0.001991         | 49.988991    | 3647.08 | 1    | lines.Generic
3         | 19                    | 4                    | O3              | [16O][16O][17O]| 667      | 7.404750e-4      | 48.98896     | 43330.85| 6    | lines.Generic
3         | 20                    | 5                    | O3              | [16O][17O][16O]| 676      | 3.702370e-4      | 48.98896     | 21404.96| 6    | lines.Generic
4         | 21                    | 1                    | N2O             | [14N]2[16O]    | 446      | 0.990333         | 44.001062    | 4984.90 | 9    | lines.Generic
4         | 22                    | 2                    | N2O             | [14N][15N][16O]| 456      | 0.003641         | 44.998096    | 3362.01 | 6    | lines.Generic
4         | 23                    | 3                    | N2O             | [15N][14N][16O]| 546      | 0.003641         | 44.998096    | 3458.58 | 6    | lines.Generic
4         | 24                    | 4                    | N2O             | [14N]2[18O]    | 448      | 0.001986         | 46.005308    | 5314.74 | 9    | lines.Generic
4         | 25                    | 5                    | N2O             | [14N]2[17O]    | 447      | 3.692800e-4      | 45.005278    | 30971.79| 54   | lines.Generic
5         | 26                    | 1                    | CO              | [12C][16O]     | 26       | 0.986544         | 27.994915    | 107.42  | 1    | lines.Generic
5         | 27                    | 2                    | CO              | [13C][16O]     | 36       | 0.011084         | 28.99827     | 224.69  | 2    | lines.Generic
5         | 28                    | 3                    | CO              | [12C][18O]     | 28       | 0.001978         | 29.999161    | 112.77  | 1    | lines.Generic
5         | 29                    | 4                    | CO              | [12C][17O]     | 27       | 3.678670e-4      | 28.99913     | 661.17  | 6    | lines.Generic
5         | 30                    | 5                    | CO              | [13C][18O]     | 38       | 2.222500e-5      | 31.002516    | 236.44  | 2    | lines.Generic
5         | 31                    | 6                    | CO              | [13C][17O]     | 37       | 4.132920e-6      | 30.002485    | 1384.66 | 12   | lines.Generic
6         | 32                    | 1                    | CH4             | [12C][1H]4     | 211      | 0.988274         | 16.0313      | 590.48  | 1    | lines.Generic
6         | 33                    | 2                    | CH4             | [13C][1H]4     | 311      | 0.011103         | 17.034655    | 1180.82 | 2    | lines.Generic
6         | 34                    | 3                    | CH4             | [12C][1H]3[2H] | 212      | 6.157510e-4      | 17.037475    | 4794.73 | 3    | lines.Generic
6         | 35                    | 4                    | CH4             | [13C][1H]3[2H] | 312      | 6.917850e-6      | 18.04083     | 9599.16 | 6    | lines.Generic
7         | 36                    | 1                    | O2              | [16O]2         | 66       | 0.995262         | 31.98983     | 215.73  | 1    | lines.Generic
7         | 37                    | 2                    | O2              | [16O][18O]     | 68       | 0.003991         | 33.994076    | 455.23  | 1    | lines.Generic
7         | 38                    | 3                    | O2              | [16O][17O]     | 67       | 7.422350e-4      | 32.994045    | 2658.12 | 6    | lines.Generic
8         | 39                    | 1                    | NO              | [14N][16O]     | 46       | 0.993974         | 29.997989    | 1142.13 | 3    | lines.Generic
8         | 40                    | 2                    | NO              | [15N][16O]     | 56       | 0.003654         | 30.995023    | 789.26  | 2    | lines.Generic
8         | 41                    | 3                    | NO              | [14N][18O]     | 48       | 0.001993         | 32.002234    | 1204.44 | 3    | lines.Generic
9         | 42                    | 1                    | SO2             | [32S][16O]2    | 626      | 0.945678         | 63.961901    | 6340.30 | 1    | lines.Generic
9         | 43                    | 2                    | SO2             | [34S][16O]2    | 646      | 0.041950         | 65.957695    | 6368.98 | 1    | lines.Generic
10        | 44                    | 1                    | NO2             | [14N][16O]2    | 646      | 0.991616         | 45.992904    | 13577.48| 3    | lines.Generic
## appears in https://hitran.org/docs/iso-meta/ but data appeard to be missing from hapy or something
#10       | 130                   | 2                    | NO2    | [15N][16O]2            | 656      | 0.003646         | 46.989938    | 9324.70 | 2     | lines.Generic
11        | 45                    | 1                    | NH3    | [14N][1H]3             | 4111     | 0.995872         | 17.026549    | 1725.22 | 3     | lines.Generic
11        | 46                    | 2                    | NH3    | [15N][1H]3             | 5111     | 0.003661         | 18.023583    | 1153.30 | 2     | lines.Generic
12        | 47                    | 1                    | HNO3   | [1H][14N][16O]3        | 146      | 0.989110         | 62.995644    | 2.14e5  | 6     | lines.Generic
12        | 117                   | 2                    | HNO3   | [1H][15N][16O]3        | 156      | 0.003636         | 63.99268     | 1.43e5  | 4     | lines.Generic
13        | 48                    | 1                    | OH     | [16O][1H]              | 61       | 0.997473         | 17.00274     | 80.35   | 2     | lines.Generic
13        | 49                    | 2                    | OH     | [18O][1H]              | 81       | 0.002000         | 19.006986    | 80.88   | 2     | lines.Generic
13        | 50                    | 3                    | OH     | [16O][2H]              | 62       | 1.553710e-4      | 18.008915    | 209.32  | 3     | lines.Generic
14        | 51                    | 1                    | HF     | [1H][19F]              | 19       | 0.999844         | 20.006229    | 41.47   | 4     | lines.Generic
14        | 110                   | 2                    | HF     | [2H][19F]              | 29       | 1.557410e-4      | 21.012404    | 115.91  | 6     | lines.Generic
15        | 52                    | 1                    | HCl    | [1H][35Cl]             | 15       | 0.757587         | 35.976678    | 160.65  | 8     | lines.Generic
15        | 53                    | 2                    | HCl    | [1H][37Cl]             | 17       | 0.242257         | 37.973729    | 160.89  | 8     | lines.Generic
15        | 107                   | 3                    | HCl    | [2H][35Cl]             | 25       | 1.180050e-4      | 36.982853    | 462.78  | 12    | lines.Generic
15        | 108                   | 4                    | HCl    | [2H][37Cl]             | 27       | 3.773500e-5      | 38.979904    | 464.13  | 12    | lines.Generic
16        | 54                    | 1                    | HBr    | [1H][79Br]             | 19       | 0.506781         | 79.92616     | 200.17  | 8     | lines.Generic
16        | 55                    | 2                    | HBr    | [1H][81Br]             | 11       | 0.493063         | 81.924115    | 200.23  | 8     | lines.Generic
16        | 111                   | 3                    | HBr    | [2H][79Br]             | 29       | 7.893840e-5      | 80.932336    | 586.40  | 12    | lines.Generic
16        | 112                   | 4                    | HBr    | [2H][81Br]             | 21       | 7.680160e-5      | 82.930289    | 586.76  | 12    | lines.Generic
17        | 56                    | 1                    | HI     | [1H][127I]             | 17       | 0.999844         | 127.912297   | 388.99  | 12    | lines.Generic
17        | 113                   | 2                    | HI     | [2H][127I]             | 27       | 1.557410e-4      | 128.918472   | 1147.06 | 18    | lines.Generic
18        | 57                    | 1                    | ClO    | [35Cl][16O]            | 56       | 0.755908         | 50.963768    | 3274.61 | 4     | lines.Generic
18        | 58                    | 2                    | ClO    | [37Cl][16O]            | 76       | 0.241720         | 52.960819    | 3332.29 | 4     | lines.Generic
19        | 59                    | 1                    | OCS    | [16O][12C][32S]        | 622      | 0.937395         | 59.966986    | 1221.01 | 1     | lines.Generic
19        | 60                    | 2                    | OCS    | [16O][12C][34S]        | 624      | 0.041583         | 61.96278     | 1253.48 | 1     | lines.Generic
19        | 61                    | 3                    | OCS    | [16O][13C][32S]        | 632      | 0.010531         | 60.970341    | 2484.15 | 2     | lines.Generic
19        | 62                    | 4                    | OCS    | [16O][12C][33S]        | 623      | 0.007399         | 60.966371    | 4950.11 | 4     | lines.Generic
19        | 63                    | 5                    | OCS    | [18O][12C][32S]        | 822      | 0.001880         | 61.971231    | 1313.78 | 1     | lines.Generic
20        | 64                    | 1                    | H2CO   | [1H]2[12C][16O]        | 126      | 0.986237         | 30.010565    | 2844.53 | 1     | lines.Generic
20        | 65                    | 2                    | H2CO   | [1H]2[13C][16O]        | 136      | 0.011080         | 31.01392     | 5837.69 | 2     | lines.Generic
20        | 66                    | 3                    | H2CO   | [1H]2[12C][18O]        | 128      | 0.001978         | 32.014811    | 2986.44 | 1     | lines.Generic
21        | 67                    | 1                    | HOCl   | [1H][16O][35Cl]        | 165      | 0.755790         | 51.971593    | 19274.79| 8     | lines.Generic
21        | 68                    | 2                    | HOCl   | [1H][16O][37Cl]        | 167      | 0.241683         | 53.968644    | 19616.20| 8     | lines.Generic
22        | 69                    | 1                    | N2     | [14N]2                 | 44       | 0.992687         | 28.006148    | 467.10  | 1     | lines.Generic
22        | 118                   | 2                    | N2     | [14N][15N]             | 45       | 0.007478         | 29.003182    | 644.10  | 6     | lines.Generic
23        | 70                    | 1                    | HCN    | [1H][12C][14N]         | 124      | 0.985114         | 27.010899    | 892.20  | 6     | lines.Generic
23        | 71                    | 2                    | HCN    | [1H][13C][14N]         | 134      | 0.011068         | 28.014254    | 1830.97 | 12    | lines.Generic
23        | 72                    | 3                    | HCN    | [1H][12C][15N]         | 125      | 0.003622         | 28.007933    | 615.28  | 4     | lines.Generic
24        | 73                    | 1                    | CH3Cl  | [12C][1H]3[35Cl]       | 215      | 0.748937         | 49.992328    | 57916.12| 4     | lines.Generic
24        | 74                    | 2                    | CH3Cl  | [12C][1H]3[37Cl]       | 217      | 0.239491         | 51.989379    | 58833.90| 4     | lines.Generic
25        | 75                    | 1                    | H2O2   | [1H]2[16O]2            | 1661     | 0.994952         | 34.00548     | 9847.99 | 1     | lines.Generic
26        | 76                    | 1                    | C2H2   | [12C]2[1H]2            | 1221     | 0.977599         | 26.01565     | 412.45  | 1     | lines.Generic
26        | 77                    | 2                    | C2H2   | [1H][12C][13C][1H]     | 1231     | 0.021966         | 27.019005    | 1656.18 | 8     | lines.Generic
26        | 105                   | 3                    | C2H2   | [1H][12C][12C][2H]     | 1222     | 3.045500e-4      | 27.021825    | 1581.84 | 6     | lines.Generic
27        | 78                    | 1                    | C2H6   | [12C]2[1H]6            | 1221     | 0.976990         | 30.04695     | 70882.52| 1     | lines.Generic
27        | 106                   | 2                    | C2H6   | [12C][1H]3[13C][1H]3   | 1231     | 0.021953         | 31.050305    | 36191.80| 2     | lines.Generic
28        | 79                    | 1                    | PH3    | [31P][1H]3             | 1111     | 0.999533         | 33.997238    | 3249.44 | 2     | lines.Generic
29        | 80                    | 1                    | COF2   | [12C][16O][19F]2       | 269      | 0.986544         | 65.991722    | 70028.43| 1     | lines.Generic
29        | 119                   | 2                    | COF2   | [13C][16O][19F]2       | 369      | 0.011083         | 66.995083    | 1.40e5  | 2     | lines.Generic
30        | 126                   | 1                    | SF6    | [32S][19F]6            | 29       | 0.950180         | 145.962492   | 1.62e6  | 1     | lines.Generic
31        | 81                    | 1                    | H2S    | [1H]2[32S]             | 121      | 0.949884         | 33.987721    | 505.79  | 1     | lines.Generic
31        | 82                    | 2                    | H2S    | [1H]2[34S]             | 141      | 0.042137         | 35.983515    | 504.35  | 1     | lines.Generic
31        | 83                    | 3                    | H2S    | [1H]2[33S]             | 131      | 0.007498         | 34.987105    | 2014.94 | 4     | lines.Generic
32        | 84                    | 1                    | HCOOH  | [1H][12C][16O][16O][1H]| 126      | 0.983898         | 46.00548     | 39132.76| 4     | lines.Generic
33        | 85                    | 1                    | HO2    | [1H][16O]2             | 166      | 0.995107         | 32.997655    | 4300.39 | 2     | lines.Generic
34        | 86                    | 1                    | O      | [16O]                  | 6        | 0.997628         | 15.994915    | 6.72    | 1     | lines.Generic
35        | 127                   | 1                    | ClONO2 | [35Cl][16O][14N][16O]2 | 5646     | 0.749570         | 96.956672    | 4.79e6  | 12    | lines.Generic
35        | 128                   | 2                    | ClONO2 | [37Cl][16O][14N][16O]2 | 7646     | 0.239694         | 98.953723    | 4.91e6  | 12    | lines.Generic
36        | 87                    | 1                    | NO+    | [14N][16O]+            | 46       | 0.993974         | 29.997989    | 311.69  | 3     | lines.Generic
37        | 88                    | 1                    | HOBr   | [1H][16O][79Br]        | 169      | 0.505579         | 95.921076    | 28339.38| 8     | lines.Generic
37        | 89                    | 2                    | HOBr   | [1H][16O][81Br]        | 161      | 0.491894         | 97.919027    | 28237.98| 8     | lines.Generic
38        | 90                    | 1                    | C2H4   | [12C]2[1H]4            | 221      | 0.977294         | 28.0313      | 11041.54| 1     | lines.Generic
38        | 91                    | 2                    | C2H4   | [12C][1H]2[13C][1H]2   | 231      | 0.021959         | 29.034655    | 45196.89| 2     | lines.Generic
39        | 92                    | 1                    | CH3OH  | [12C][1H]3[16O][1H]    | 2161     | 0.985930         | 32.026215    | 70569.92| 2     | lines.Generic
40        | 93                    | 1                    | CH3Br  | [12C][1H]3[79Br]       | 219      | 0.500995         | 93.941811    | 83051.98| 4     | lines.Generic
40        | 94                    | 2                    | CH3Br  | [12C][1H]3[81Br]       | 211      | 0.487433         | 95.939764    | 83395.21| 4     | lines.Generic
41        | 95                    | 1                    | CH3CN  | [12C][1H]3[12C][14N]   | 2124     | 0.973866         | 41.026549    | 88672.19| 3     | lines.Generic
42        | 96                    | 1                    | CF4    | [12C][19F]4            | 29       | 0.988890         | 87.993616    | 1.21e5  | 1     | lines.Generic
43        | 116                   | 1                    | C4H2   | [12C]4[1H]2            | 2211     | 0.955998         | 50.01565     | 9818.97 | 1     | lines.Generic
44        | 109                   | 1                    | HC3N   | [1H][12C]3[14N]        | 1224     | 0.963346         | 51.010899    | 24786.84| 6     | lines.Generic
45        | 103                   | 1                    | H2     | [1H]2                  | 11       | 0.999688         | 2.01565      | 7.67    | 1     | lines.Generic
45        | 115                   | 2                    | H2     | [1H][2H]               | 12       | 3.114320e-4      | 3.021825     | 29.87   | 6     | lines.Generic
46        | 97                    | 1                    | CS     | [12C][32S]             | 22       | 0.939624         | 43.971036    | 253.62  | 1     | lines.Generic
46        | 98                    | 2                    | CS     | [12C][34S]             | 24       | 0.041682         | 45.966787    | 257.77  | 1     | lines.Generic
46        | 99                    | 3                    | CS     | [13C][32S]             | 32       | 0.010556         | 44.974368    | 537.50  | 2     | lines.Generic
46        | 100                   | 4                    | CS     | [12C][33S]             | 23       | 0.007417         | 44.970399    | 1022.97 | 4     | lines.Generic
47        | 114                   | 1                    | SO3    | [32S][16O]3            | 26       | 0.943400         | 79.95682     | 7783.30 | 1     | lines.Generic
48        | 123                   | 1                    | C2N2   | [12C]2[14N]2           | 4224     | 0.970752         | 52.006148    | 15582.44| 1     | lines.Generic
49        | 124                   | 1                    | COCl2  | [12C][16O][35Cl]2      | 2655     | 0.566392         | 97.9326199796| 1.48e6  | 1     | lines.Generic
49        | 125                   | 2                    | COCl2  | [12C][16O][35Cl][37Cl] | 2657     | 0.362235         | 99.9296698896| 3.04e6  | 16    | lines.LinearTriatomic
53        | 131                   | 1                    | CS2    | [12C][32S]2            | 222      | 0.892811         | 75.94414     | 1352.60 | 1     | lines.LinearTriatomic
53        | 132                   | 2                    | CS2    | [32S][12C][34S]        | 224      | 0.079260         | 77.93994     | 2798.00 | 1     | lines.LinearTriatomic
53        | 133                   | 3                    | CS2    | [32S][12C][33S]        | 223      | 0.014094         | 76.943256    | 1107.00 | 4     | lines.LinearTriatomic
53        | 134                   | 4                    | CS2    | [13C][32S]2            | 232      | 0.010310         | 76.947495    | 2739.70 | 2     | lines.LinearTriatomic

''')

