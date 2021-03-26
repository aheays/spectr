import subprocess

from . import *


def calculate_equilibrium_ggchem(
        elements_abundances,    # a dict e.g., {'H':1e-3,'C':1e-5,...}
        P=None,Tmin=None,Tmax=None, # T scan
        T=None,Pmin=None,Pmax=None, # or P scan
        npoints=100,            # number of points if a grid
        verbose=False,
        keep_tmpdir=False,
):
    ## make an input file for ggchem
    if P is not None and Tmin is not None and Tmax is not None:
        Pmin = Pmax = P
    elif T is not None and Pmin is not None and Pmax is not None:
        Tmin = Tmax = T
    else:
        raise Exception("Set (P,Tmin,Tmax) or (T,Pmin,Pma).")
    ## make tmpdir and input files
    tmpdir = tools.tmpdir()
    with open(f'{tmpdir}/abundances','w') as fid:
        for element,abundance in elements_abundances.items():
            fid.write(f'{element} {abundance}\t\n')
    element_list = ' '.join(elements_abundances)
    element_list += ' el'           # add electrons/ions
    with open(f'{tmpdir}/input','w') as fid:
        fid.write(f'# selected elements\n')
        fid.write(f'{element_list}\n')
        fid.write(f'\n')
        fid.write(f'# name of files with molecular kp-data\n')
        fid.write(f'dispol_BarklemCollet.dat                ! dispol_file\n')
        fid.write(f'dispol_StockKitzmann_withoutTsuji.dat   ! dispol_file2\n')
        fid.write(f'dispol_WoitkeRefit.dat                  ! dispol_file3\n')
        fid.write(f'\n')
        fid.write(f'# abundance options 1=EarthCrust, 2=Ocean, 3=Solar, 4=Meteorites\n')
        fid.write(f'0                     ! abund_pick\n')
        fid.write(f'abundances\n')
        fid.write(f'\n')
        fid.write(f'# equilibrium condensation?\n')
        fid.write(f'.false.               ! model_eqcond\n')
        fid.write(f'\n')
        fid.write(f'# model options\n')
        fid.write(f'1                     ! model_dim  (0,1,2)\n')
        fid.write(f'.true.                ! model_pconst\n')
        fid.write(f'{Tmax}                  ! Tmax [K]\n')
        fid.write(f'{Tmin}                  ! Tmin [K]      (if model_dim>0)\n')
        fid.write(f'{Pmax}                  ! pmax [bar]    (if pconst=.true.)\n')
        fid.write(f'{Pmin}                  ! pmin [bar]\n')
        fid.write(f'4.E+19                ! nHmax [cm-3]  (if pconst=.false.)\n')
        fid.write(f'4.E+19                ! nHmin [cm-3]\n')
        fid.write(f'{npoints}                    ! Npoints       \n')
        fid.write(f'5                     ! NewBackIt  \n')
        fid.write(f'1000.0                ! Tfast\n')
        fid.write(f'\n')
    shutil.copy(tools.expand_path('~/data/chemistry/thermochemistry/GGchem/src16/ggchem16'),tmpdir)
    shutil.copytree(tools.expand_path('~/data/chemistry/thermochemistry/GGchem/data'),f'{tmpdir}/data')
    ## get results
    status,output = subprocess.getstatusoutput([f'cd {tmpdir} && ./ggchem16 input'])
    if status!=0:
        print(output)
        raise Exception(f'ggchem error status: {status}')
    if verbose:
        print( output)
    ## put results in a Dataset
    data = Dataset(**tools.file_to_dict(f'{tmpdir}/Static_Conc.dat',skiprows=2,labels_commented=False))
    ## process results a bit
    not_log_keys=('Tg','nHges',
                  'pgas','SC','SCO','SCO2','nC','nCO','nCO2','epsC',
                  'epsN','epsO','dust/gas','dustVol/H','Jstar(W)','Nstar(W)',)
    not_species_keys=('Tg','nHges','pgas','SC','SCO',
                      'SCO2','nC','nCO','nCO2','epsC','epsN',
                      'epsO','dust/gas','dustVol/H','Jstar(W)','Nstar(W)',)
    for key in data.keys():
        if key not in not_log_keys:
            data[key] = 10**data[key]
    data['nt'] = np.sum([data[key] for key in data if key not in not_species_keys],0)
    data['patm'] = convert.units(data['pgas'],'dyn.cm-2','atm')
    ## delete tmpdir
    if keep_tmpdir:
        print(f'tmpdir: {tmpdir}')
    else:
        shutil.rmtree(tmpdir)
    return data
