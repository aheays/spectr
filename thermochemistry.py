import subprocess

from . import *

def ggchem(
        ## a dict e.g., {'H':1e-3,'C':1e-5,...}. if not constant
        ## pressure these are densities in cm-3 otherwise relative
        ## abundances
        elements,
        Tmin,Tmax,Tn=100, # K temperature scan, min ,max ,number of points
        p=None,              # bar, None for constant density 
        verbose=False,
        include_charge=True,     # include ionised species and electrons
        keep_tmpdir=False,
):
    """Compute thermochemical equilibrium gas phase products over a range
    of temperautures. Uses ggchem (woitke2015)."""
    ## H is the reference element in ggchem so it must be present,
    ## here I set it to a hopefully negligible level
    if 'H' not in elements:
        elements['H'] = 1e-20 
    ## make an input file for ggchem
    if p is not None:
        ## constant pressure
        pconst = True 
        nH = -1
    else:
        ## constant total elemental density. in ggchem this is
        ## referenced to H
        pconst = False 
        p = -1
        nH = elements['H']
    ## make tmpdir and input files
    tmpdir = tools.tmpdir()
    if verbose:
        print(f'tmpdir: {tmpdir}')
    with open(f'{tmpdir}/abundances','w') as fid:
        for element,abundance in elements.items():
            fid.write(f'{element} {abundance:0.3E}\n')
    ## Generate a ggchem in put file. Comments after ! are read by
    ## read_parameter.f to determine the meaning of each line (and
    ## possibly subsequent lines).
    with open(f'{tmpdir}/input','w') as fid:
        element_list = ' '.join(elements) 
        if include_charge:
            element_list = element_list+' el'
        fid.write(f'{element_list}\n')
        fid.write(f'dispol_BarklemCollet.dat              ! dispol_file\n')
        fid.write(f'dispol_StockKitzmann_withoutTsuji.dat ! dispol_file2\n')
        fid.write(f'dispol_WoitkeRefit.dat                ! dispol_file3\n')
        fid.write(f'0                                     ! abund_pick\nabundances\n')
        fid.write(f'.true.                                ! pick_mfrac\n')
        fid.write(f'.false.                               ! model_eqcond\n')
        fid.write(f'1                                     ! model_dim  (0,1,2)\n')
        fid.write(f'{Tmax}                                ! Tmax [K]\n')
        fid.write(f'{Tmin}                                ! Tmin [K]      (if model_dim>0)\n')
        if pconst:
            fid.write(f'.true.                            ! model_pconst\n')
            fid.write(f'{p}                               ! pmax [bar]    (if pconst=.true.)\n')
            fid.write(f'{p}                               ! pmin [bar]')
        else:
            fid.write(f'.false.                           ! model_pconst\n')
            fid.write(f'{nH}                              ! nHmax [cm-3]  (if pconst=.false.)\n')
            fid.write(f'{nH}                              ! nHmin [cm-3]\n')
        fid.write(f'{Tn}                                 ! Npoints       \n')
        fid.write(f'5                                     ! NewBackIt  \n')
        fid.write(f'1000.0                                ! Tfast\n')
    shutil.copy(tools.expand_path('~/data/chemistry/thermochemistry/GGchem/src16/ggchem16'),tmpdir)
    shutil.copytree(tools.expand_path('~/data/chemistry/thermochemistry/GGchem/data'),f'{tmpdir}/data')
    ## get results
    status,output = subprocess.getstatusoutput([f'cd {tmpdir} && ./ggchem16 input > t.out'])
    if status!=0:
        print(output)
        raise Exception(f'ggchem error status: {status}')
    if verbose:
        print( output)
    data = Dataset(**tools.file_to_dict(f'{tmpdir}/Static_Conc.dat',skiprows=2,labels_commented=False))
    ## Convert into a less complex standard Mixture datset.  Transfer
    ## log abundances to linear, discard other data
    mixture = kinetics.Mixture(
        T=data['Tg'],
        p=convert.units(data['pgas'],'dyn.cm-2','Pa'),)
    for key in data.keys():
        if key not in (
                'Tg','nHges',
                'pgas','SC','SCO','SCO2',
                'nC','nCO','nCO2','epsC',
                'epsN','epsO','dust/gas',
                'dustVol/H','Jstar(W)','Nstar(W)',):
            mixture[key] = 10**data[key]
    ## delete tmpdir
    if keep_tmpdir:
        print(f'tmpdir: {tmpdir}')
    else:
        shutil.rmtree(tmpdir)
    ## return raw ggchem data and mixture dataset
    return data,mixture
