import numpy as np
from . import tools
from . import lines

## codes at https://hitran.org/docs/iso-meta/
_map_codes_to_species = {
    1:'H2O', 2:'CO2', 3:'O3', 4:'N2O',
    5:'CO', 6:'CH4', 7:'O2', 8:'NO', 9:'SO2', 10:'NO2', 11:'NH3',
    12:'HNO3', 13:'OH', 14:'HF', 15:'HCl', 16:'HBr', 17:'HI',
    18:'ClO', 19:'OCS', 20:'H2CO', 21:'HOCl', 22:'N2', 23:'HCN',
    24:'CH3Cl', 25:'H2O2', 26:'C2H2', 27:'C2H6', 28:'PH3', 29:'COF2',
    30:'SF6', 31:'H2S', 32:'HCOOH', 33:'HO2', 34:'O', 35:'ClONO2',
    37:'HOBr', 38:'C2H4', 39:'CH3OH', 40:'CH3Br', 41:'CH3CN',
    42:'CF4', 43:'C4H2', 44:'HC3N', 45:'H2', 46:'CS', 47:'SO3',
    48:'C2N2', 49:'COCl2', (2,1):'[12C][16O]2', (2,2):'[13C][16O]2',
    (2,3):'[16O][12C][18O]', (2,4):'[16O][12C][17O]', (2,5):'[16O][13C][18O]', (2,6):'[16O][13C][17O]',
    (2,7):'[12C][18O]2', (2,8):'[17O][12C][18O]', (2,9):'[12C][17O]2',
    (2,10):'[13C][18O]2', (2,11):'[18O][13C][17O]', (2,12):'[13C][17O]2',
    (5,1):'12C16O',(5,2):'13C16O',(5,3):'12C18O',(5,4):'12C17O',(5,5):'13C18O',(5,6):'13C17O',
    (5,1):'[12C][16O]',(5,2):'[13C][16O]',(5,3):'[12C][18O]',(5,4):'[12C][17O]',(5,5):'[13C][18O]',(5,6):'[13C][17O]',
    (8,1):'14N16O', (8,2):'15N16O', (8,3):'14N18O', 
    (8,1):'[14N][16O]', (8,2):'[15N][16O]', (8,3):'[14N][18O]',
    (19,1):'[16O][12C][32S]', (19,2):'[16O][12C][34S]', (19,3):'[16O][13C][32S]', (19,4):'[16O][12C][33S]', (19,5):'[18O][12C][32S]', # OCS
    (24,1):'[12C][1H]3[35Cl]', (24,2):'[12C][1H]3[37Cl]', # CH3Cl
    (6,1):'[12C][1H]4', (6,2):'[13C][1H]4', (6,3):'[12C][1H]3[2H]', (6,4):'[13C][1H]3[2H]',
    (15,1):'[1H][35Cl]', (15,2):'[1H][37Cl]', (15,3):'[2H][35Cl]', (15,4):'[2H][37Cl]',


}
_map_species_to_codes = {val:key for key,val in _map_codes_to_species.items()}

@tools.vectorise_in_chunks
def get_partition_function(species,temperature):
    """Use hitran to get a partition function."""
    import hapi
    Mol,Iso = translate_species_to_codes(species)
    return(hapi.partitionSum(Mol,Iso,temperature))

def translate_species_to_codes(species):
    """Get Hitran Mol and Iso codes from a species mode."""
    if not np.isscalar(species):
        return([translate_codes_to_species(t) for t in species])
    t = _map_species_to_codes[species]
    if np.isscalar(t):
        Mol,Iso = t,1
    else:
        Mol,Iso = t
    return(Mol,Iso)

def translate_codes_to_species(Mol,Iso=None):
    """Get Hitran Mol and Iso codes from a species mode."""
    if Iso is None:
        if not np.isscalar(Mol):
            return([translate_codes_to_species(Moli) for Moli in Mol])
        else:
            return(_map_codes_to_species[Mol])
    else:
        if not np.isscalar(Mol):
            return([translate_codes_to_species(Moli,Isoi) for (Moli,Isoi) in zip(Mol,Iso)])
        else:
            return(_map_codes_to_species[(Mol,Iso)])

def download_linelist(
        species,
        νbeg,νend,
        data_directory='td',
        table_name=None,        # defaults to species
):
    import hapi
    if table_name is None:
        table_name = species
    t = _map_species_to_codes[species]
    if np.isscalar(t):
        molecule_id = t
        isotopologue_id = 1
    else:
        molecule_id,isotopologue_id = t
    tools.mkdir_if_necessary(data_directory)
    hapi.db_begin(data_directory)
    hapi.fetch(table_name,molecule_id,isotopologue_id,νbeg,νend)
    

def load_lines(filename):
    """Load HITRAN .data file recarray."""
    data = np.genfromtxt(
        tools.expand_path(filename),
        dtype=[
            ('Mol',int),    # molecule code number
            ('Iso',int),    # isotopologue code number
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
            ('g_l',float),    # ujpper level degeneracy
        ],
        delimiter=(2,1,12,10,10,5,5,10,4,8,15,15,15,15,6,12,1,7,7), # column widths
    )
    return(data)

# def load_lines(filename):
    # """Load HITRAN .data file into a Dynamic_Recarray linelist. INCOMPLETE."""
    # # import spectra
    # ## get raw HITRAN data
    # data = load_lines_as_recarray(filename)
    # species = np.unique(translate_codes_to_species(data['Mol']))
    # assert len(species)==1,'Cannot handle mixed species HITRAN linelist.'
    # species = species[0]
    # ## interpret into transition quantities common to all transitions
    # kw = dict(ν=data['ν'], Ae=data['A'],
              # E_l=data['E_l'],
              # gu=data['gu'],
              # g_l=data['g_l'],
              # γair=data['γair']*2, # HITRAN uses HWHM, I'm going to go with FWHM
              # nair=data['nair'],
              # δair=data['δair'],
              # γself=data['γself']*2, # HITRAN uses HWHM, I'm going to go with FWHM
    # )
    # ## full isotopologue
    # assert len(np.unique(data['Mol']))==1
    # try:
        # kw['species_l'] = kw['speciesu'] = translate_codes_to_species(data['Mol'],data['Iso'])
    # except KeyError:
        # assert len(np.unique(data['Iso']))==1,'Cannot identify isotopologues and multiple are present.'
        # kw['species_l'] = kw['speciesu'] = translate_codes_to_species(data['Mol'])
    # ## interpret quantum numbers and insert into some kind of transition, this logic is in its infancy
    # if species in ('CO',):
        # ## standin for diatomics
        # kw['vu'] = data['Vu']
        # kw['v_l'] = data['V_l']
        # branches = {'P':-1,'Q':0,'R':+1}
        # ΔJ,J_l = [],[]
        # for Q_l in data['Q_l']:
            # branchi,Jli = Q_l.split()
            # ΔJ.append(branches[branchi])
            # J_l.append(Jli)
        # kw['ΔJ'] = np.array(ΔJ,dtype=int)
        # kw['J_l'] = np.array(J_l,dtype=float)
        # return(lines.HeteronuclearDiatomicLines(**kw))
    # elif species=='CH4':
        # def decode_Qp(Qp):
            # return({'Ju':tools.string_to_number(Qp[2:5],np.nan),
                        # # 'Rgrouu':Qp[5:7].strip(),
                        # # 'αu':float(Qp[7:10]),
            # })
        # def decode_Vp(Vp):
            # return({'v1u' : tools.string_to_number(Vp[3:5],default=-1),
                    # 'v2u' : tools.string_to_number(Vp[5:7],default=-1),
                    # 'v3u' : tools.string_to_number(Vp[7:9],default=-1),
                    # 'v4u' :tools.string_to_number(Vp[9:11],default=-1),
                    # # 'nu'  : tools.string_to_number(Vp[11:13]),
                    # 'vgroup_l' : Vp[13:15].strip()})
        # def decode_Q_l(Q_l):
            # return({key+'u':val for key,val in decode_Qp(Q_l).items()})
        # def decode_V_l(V_l):
            # return({key+'u':val for key,val in decode_Vp(V_l).items()})
        # kw.update(tools.vectorise_dicts(*[decode_Qp(t) for t in data['Qu']]))
        # kw.update(tools.vectorise_dicts(*[decode_Q_l(t) for t in data['Q_l']]))
        # kw.update(tools.vectorise_dicts(*[decode_Vp(t) for t in data['Vu']]))
        # kw.update(tools.vectorise_dicts(*[decode_V_l(t) for t in data['V_l']]))
        # return(Td_Transition(**kw))
    # else:
        # ## anything else
        # return(Generic_Transition(**kw))

def get_spectrum(
        species,
        νbeg,νend,
        spectrum_type='absorption',
        data_directory=None,
        table_name=None,        # defaults to species
):
    import hapi
    if data_directory is None:
        data_directory = f'/home/heays/data/databases/HITRAN/data/{species}/'
    if table_name is None:
        table_name = 'linelist'
    molecule_id = _map_species_to_codes[species]
    isotopologue_id = 1         # could be generalised
    tools.mkdir_if_necessary(data_directory)
    hapi.db_begin(data_directory)
    print(hapi.tableList())
    ν,coef = hapi.absorptionCoefficient_Lorentz(SourceTables=table_name)
    if spectrum_type == 'absorption':
        return(ν,coef)
    elif spectrum_type == 'emission':
        ν,radi = hapi.radianceSpectrum(ν,coef)
        return(ν,radi)
    else:
        raise Exception(f'unknown spectrum type: {spectrum_type}')


# from . import *
# ## used commonly by some scripts
# # import my
# # import os,subprocess
# # import urllib.request, urllib.parse, urllib.error,os,re,argparse,datetime,sys,collections
# # import numpy as np
# # import periodictable
# # from matplotlib import pyplot as plt
# # import spectra
# # from scipy import constants

# from copy import copy,deepcopy

# def resample_to_finest_grid(x0,y0,x1,y1):
    # """Linearly intepolate (x0,y0) and (x1,y1) to a grid that contains
    # all points in x0 and x1. Any extrapolations are set to zero."""
    # i = np.argsort(x0)
    # x0,y0 = x0[i],y0[i]
    # i = np.argsort(x1)
    # x1,y1 = x1[i],y1[i]
    # ## get a grid containing all points in x0 and y0
    # xnew = np.unique(np.concatenate((x0,x1)))
    # y0new = np.zeros(xnew.shape)
    # i = (xnew>=x0[0])&(xnew<=x0[-1])
    # y0new[i] = tools.spline(x0,y0,xnew[i],order=1)   #linearly interpolate
    # y1new = np.zeros(xnew.shape)
    # i = (xnew>=x1[0])&(xnew<=x1[-1])
    # y1new[i] = tools.spline(x1,y1,xnew[i],order=1)   #linearly interpolate
    # return(xnew,y0new,y1new)
   #  
# def resample_data(x,y,xout,default=0.,
                  # check_limits=False,
                  # check_positive=False,
                  # ensure_positive=False,
                  # set_nan_to_zero=False,
                  # use_fortran=True,
# ):
    # """Spline or bin (as appropriate) (x,y) data to a given xout grid"""
    # ## sort original data
    # x,i = np.unique(x,return_index=True)
    # y = y[i]
    # if check_limits:
        # assert xout[0]>=x[0],'Missing lowest wavelengths: '+str(xout[0])+' < '+str(x[0])
        # assert xout[-1]<=x[-1],'Missing highest wavelengths: '+str(xout[-1])+' > '+str(x[-1])
    # if check_positive:
        # assert all(y>=0),'Negative y data'
    # if set_nan_to_zero:
        # y[np.isnan(y)] = 0.
    # ## calculate using compiled fortran code - much faster
    # if use_fortran:
        # from spectra import lib_molecules_fortran
        # yout = np.zeros(xout.shape,dtype=float)
        # # lib_molecules_fortran.lib_molecules.resample_data(
        # #     np.array(x,dtype=float),
        # #     np.array(y,dtype=float),
        # #     np.array(len(x)),
        # #     np.array(xout,dtype=float),
        # #     yout,
        # #     np.array(len(xout)]))
        # lib_molecules_fortran.lib_molecules.resample_data(
            # np.array(x,dtype=float),
            # np.array(y,dtype=float),
            # # np.array(len(x)),
            # np.array(xout,dtype=float),
            # yout,
            # # np.array(len(xout)),
        # )
    # ## use python functions only
    # else:
       # ## find out which new bin each original data point should go in
       # ibin = np.digitize(x[(x>=xout[0])&(x<=xout[-1])],xout,right=True)
       # ## count how many original datapoints in each new bin, might need to
       # ## extend to full size of xout
       # t_icount = np.bincount(ibin)
       # icount = np.zeros(xout.shape)
       # icount[0:len(t_icount)] = t_icount
       # ## initalise the output array
       # yout = np.ones(xout.shape)*default
       # ## all new data bins that contain 2 or less old data points will
       # ## be linearly interpolated
       # i = icount<3
       # j = i&(xout>=x[0])&(xout<=x[-1])
       # if any(j):
           # yout[j] = tools.spline(x,y,xout[j],order=1)
       # ## all others, calculate the mean of the original datapoints they
       # ## contain A WEIGHTED MEAN WOULD BE BETTER - BUT LIKELY TOO SLOW
       # for j in np.argwhere(~i):
           # yout[j] = np.mean(y[(x>=xout[j-1])&(x<xout[min(len(xout)-1,j+1)])])
    # if ensure_positive:
        # yout = np.abs(yout)
    # return yout

# def dwim_load_cross_sections(wl,cross_section_filename,species=None,verbose=True):
    # """This is a bit of a hack. Load cross sections from filename in some idealised way."""
    # ## Leiden photo website, expects .pd and .pi files in this directory
    # if cross_section_filename=='photo website':
        # photo_website_directory = '~/projects/photo_website/photo/pd/'
        # pd_filename = tools.expand_path(photo_website_directory+species.lower()+'.pd')
        # pi_filename = tools.expand_path(photo_website_directory+species.lower()+'.pi')
        # if verbose:
            # print("loading photodissociation cross section from: "+pd_filename)
            # print("loading photoionisation cross section from: "+pi_filename)
        # ## return zeros if file doesn't exist
        # if os.path.exists(pd_filename):
            # pd_data = load_cross_section_leiden_photodissoc_database(pd_filename)
            # pd = resample_data(pd_data['wavelength'],pd_data['cross_section'],wl)
        # else:
            # pd = np.zeros(wl.shape)
        # if os.path.exists(pi_filename):
            # pi_data = load_cross_section_leiden_photodissoc_database(pi_filename)
            # pi = resample_data(pi_data['wavelength'],pi_data['cross_section'],wl)
        # else:
            # pi = np.zeros(wl.shape)
        # cross_sections = collections.OrderedDict()
        # cross_sections['photoabsorption'] = pd + pi
        # cross_sections['photodissociation'] = pd
        # cross_sections['photoionisation'] = pi
    # ## else assume in my readable format
    # elif cross_section_filename=='old photo website broadened':
        # cross_section_filename = tools.expand_path('~/projects/photodissoc_database/results/old_website_data_2015-03-11_converted_to_continuum/'+species+'.hdf5')
        # if verbose: print("load_cross_sections from file:",cross_section_filename)
        # cross_sections = collections.OrderedDict()
        # t = tools.file_to_recarray(cross_section_filename)
        # for key in t.dtype.names:
            # if key=='wavelength': continue
            # cross_sections[key] = resample_data(t['wavelength'],t[key],wl,ensure_positive=True)
        # if verbose: print()
    # elif cross_section_filename=='data molecules':
        # for cross_section_filename in (
                # tools.expand_path('~/data/molecules/'+species+'/cross_sections/all_cross_sections.hdf5'),
                # tools.expand_path('~/data/molecules/'+species+'/cross_sections/all_cross_sections'),
        # ):
            # if os.path.exists(cross_section_filename): break
        # else:
            # raise Exception('File could not be found in '+'~/data/molecules/'+species+'/cross_sections')
        # if verbose: print("load_cross_sections from file:",cross_section_filename)
        # cross_sections = collections.OrderedDict()
        # t = tools.file_to_recarray(cross_section_filename)
        # for key in t.dtype.names:
            # if key=='wavelength': continue
            # cross_sections[key] = resample_data(t['wavelength'],t[key],wl,ensure_positive=True)
        # if verbose: print()
    # elif cross_section_filename=='data atoms':
        # for cross_section_filename in (
                # tools.expand_path('~/data/atoms/'+species+'/cross_sections/all_cross_sections.hdf5'),
                # tools.expand_path('~/data/atoms/'+species+'/cross_sections/all_cross_sections'),
        # ):
            # if os.path.exists(cross_section_filename): break
        # else:
            # raise Exception('File could not be found in '+'~/data/atoms/'+species+'/cross_sections')
        # if verbose: print("load_cross_sections from file:",cross_section_filename)
        # cross_sections = collections.OrderedDict()
        # t = tools.file_to_recarray(cross_section_filename)
        # for key in t.dtype.names:
            # if key=='wavelength': continue
            # cross_sections[key] = resample_data(t['wavelength'],t[key],wl,ensure_positive=True)
        # if verbose: print()
    # else:
        # if verbose: print("load_cross_sections from file:",cross_section_filename)
        # cross_sections = collections.OrderedDict()
        # t = tools.file_to_recarray(cross_section_filename)
        # for key in t.dtype.names:
            # if key=='wavelength': continue
            # cross_sections[key] = resample_data(t['wavelength'],t[key],wl,ensure_positive=True)
        # if verbose: print()
    # return(cross_sections)

# def load_cross_section_from_file(filename,output_x_units='nm',output_y_units='cm2',):
    # """Load a cross section from a file. Return (x,y). x is either
    # wavelength or an energy or a freuqency, judged by the governing
    # units. y is a cross section. Attempts to heuristically convert
    # file data into requesting outputs. Probably this is better than dwim_load_cross_section HEURISTICS CURRENTLY VERY
    # LIMITED."""
    # ## load data
    # data = tools.file_to_recarray(filename)
    # ## heurisitically get correct outputs
    # if   output_x_units=='nm'   and 'wavelength'  in data.dtype.names:      x = data['wavelength']
    # elif   output_x_units=='nm'   and 'wavelength(nm)'  in data.dtype.names:      x = data['wavelength(nm)']
    # elif output_x_units=='cm-1' and 'wavenumbers' in data.dtype.names:      x = data['wavenumbers']
    # elif output_x_units=='cm-1' and 'wavelength'  in data.dtype.names:      x = tools.nm2k(data['wavelength'])
    # else: raise Exception("Could not determine x output")
    # if   output_y_units=='cm2'  and 'cross_section' in data.dtype.names:    y = data['cross_section']
    # if   output_y_units=='cm2'  and 'cross_section(cm2)' in data.dtype.names:    y = data['cross_section(cm2)']
    # elif output_y_units=='cm2'  and 'photoabsorption' in data.dtype.names:  y = data['photoabsorption']
    # else: raise Exception("Could not determine y output")
    # ## sort and uniquify
    # x,i = np.unique(x,return_index=True)
    # y = y[i]
    # return(x,y)

# def get_nonzero_limits(x,y):
    # t = x[y>0]
    # if any(t):
        # return(t[0],t[-1])
    # else:
        # return(np.inf,-np.inf)

# def plot_cross_sections(ax,wl,axs,dxs=None,ixs=None,yscale='linear'):
    # """If axs is a dictionary/recarray then assume might contain keys
    # photoabsorptionn / photodissociation /photoionisation"""
    # if isinstance(axs,dict) or isinstance(axs,np.recarray):
        # try:
            # dxs = axs['photodissociation']
        # except KeyError:
            # dxs = None
        # try:
            # ixs = axs['photoionisation']
        # except KeyError:
            # ixs = None
        # try:
            # axs = axs['photoabsorption']
        # except KeyError:
            # axs = None
    # if axs is not None:
        # axs_min_wl,axs_max_wl = get_nonzero_limits(wl,axs)
        # ax.plot(wl,axs,label='axs',color='red',ls='-',)
        # ax.plot(wl,np.cumsum(axs)/axs.sum()*axs.max(),color='red',ls=':',)
    # if dxs is not None:
        # dxs_min_wl,dxs_max_wl = get_nonzero_limits(wl,dxs)
        # ax.plot(wl,dxs,label='dxs',color='blue')
        # ax.plot(wl,np.cumsum(dxs)/dxs.sum()*dxs.max(),color='blue',ls=':',)
    # if ixs is not None:
        # ixs_min_wl,ixs_max_wl = get_nonzero_limits(wl,ixs)
        # ax.plot(wl,ixs,label='ixs',color='green')
        # ax.plot(wl,np.cumsum(ixs)/ixs.sum()* ixs.max(),color='green',ls=':',)
    # ax.set_xlim(wl[0],wl[-1])
    # tools.simpleTickLabels()
    # # ax.plot([dxs_min_wl,dxs_min_wl],ax.get_ylim(),ls=':',color='black',lw=3,)
    # # ax.plot([ixs_min_wl,ixs_min_wl],ax.get_ylim(),ls=':',color='black',lw=3,)
    # tools.legend(ax=ax)
    # ax.set_yscale(yscale)
    # if yscale=='linear':
        # ax.set_ylim(ymin=0)
    # elif yscale=='log':
        # ax.set_ylim(*get_nonzero_limits(np.concatenate(axs)))
    # return ax

# def mainz_download_data_and_header(url,output_dir='./'):
    # """Takes a Mainz database molecule page, and downloads all
    # datasets into text files with the info pagea s a header."""
    # ## parse webpage extracting header information
    # import datetime
    # output_lines = []
    # output_lines.append('# Extracted from MAINZ UV/VIS website '+datetime.datetime.now().isoformat())
    # output_lines.append('#')
    # for l in tools.wget(url).split('\n'):
        # ## look for data file url
        # r = re.match('.*href="([^"]*\.txt).*',l)
        # if r:
            # ## expand % symbls, spline into domains
            # datafile_url = r.groups()[0]
            # ## t = urllib.unquote(r.groups()[0]).split('/')
            # # datafile_url = datafile_url.replace(' ','+',1) # replace first ' ' with +, only the first!
            # # datafile_url = datafile_url.replace('(','%28').replace(')','%29') # replace parenthese with html codes
            # datafile_url = datafile_url.replace('T+dep','T dep')
            # # datafile_url = 'joseba.mpch-mainz.mpg.de/spectral_atlas_data/'+datafile_url
            # subprocess.call(['wget',datafile_url],cwd=output_dir)
            # output_filename = os.path.basename(datafile_url).replace('%28','(').replace('%29',')')
            # output_lines.append('# URL:          '+url)
            # output_lines.append('# DATA_URL:     '+datafile_url)
            # output_lines.append('# LOCAL_FILE:   '+output_filename)
            # output_lines.append('#')
            # ## get data -- something wrong with urllib, using wget externally instead -- NOT PORTABLE
            # # datafile_url = 'joseba.mpch-mainz.mpg.de/spectral_atlas_data/'+datafile_url
            # tools.wget(datafile_url)        
            # output_lines.append(tools.wget(datafile_url))
            # ## write to file or stdout
            # if output_dir==None:
                # print('\n'.join(output_lines))
            # else:
                # assert not os.path.isfile(output_dir), 'Bad output_dir, is a file'
                # if not os.path.isdir(output_dir):
                    # os.mkdir(output_dir)
                # f = open(output_dir+'/'+output_filename,'w')
                # f.write('\n'.join([str(t) for t in output_lines]))
                # f.close()
            # continue
        # ## look for reference data - append to output data
        # r = re.match(r'<TR valign="top"><TD>(.*:)</TD>( +)<TD>(.*)</TD></TR>',str(l))
        # if r:
            # ## save molecular formula
            # if r.groups()[0]=='FORMULA:':
                # formula = r.groups()[2]
            # output_lines.append('# '+''.join(r.groups()))

# def mainz_load_data(filename):
    # """Load whatever data can be got from a mainz database file."""
    # try: # try to load as numeric array with various dimensions
        # t = tools.file_to_array(filename,dtype=float).squeeze()
        # if t.shape==(2,):       # one (x,y) datapoint
            # x,y,dy = t[0:1],t[1:],None
        # elif t.shape==(3,):     # one (x,y,dy) datapoint in numeric form
            # x,y,dy = t[0:1],t[1:2],t[2:]
        # elif t.ndim==2 and t.shape[1]==2: # array of (x,y) data
            # x,y,dy = t[:,0],t[:,1],None   
        # elif t.ndim==2 and t.shape[1]==3: # array of (x,y,dy) data
            # x,y,dy = t[:,0],t[:,1],t[:,2]
        # elif t.ndim==2 and t.shape[1]>3:
            # print('bad data dimension, loading first two columns: '+filename+' '+repr(t.shape))
            # x,y,dy = t[:,0],t[:,1],None
        # else:
            # print('bad data dimension, giving up: '+filename+' '+repr(t.shape))
            # x,y,dy = None,None,None
        # return (x,y,dy)
    # except: 
        # pass
    # try: # try to load from the format e.g., 121.567 (1.5 +- 0.1)e-17
        # f = open(filename,'r')
        # x,y,dy = [],[],[]
        # for line in f:
            # r = re.match(r'^\s*([-+0-9.]+)\s+\(\s*([-+0-9.]+)\b\s*\D+\s*([0-9.+-]+)\s*\)\s*e([0-9+-]+)\b',line)
            # if r:
                # a,b,c,d = r.groups()
                # x.append(float(a))
                # y.append(float(b)*10**float(d))
                # dy.append(float(c)*10**float(d))
        # return (np.array(x,ndmin=1),np.array(y,ndmin=1),np.array(dy,ndmin=1))
    # except:
        # pass
    # ## failed
    # print('Couldnt plot filename: '+filename)

# def phidrates_load_cross_sections(filename,convert_wavelength_units=True,parse_and_replace_keys=True):
    # """Load a file downloaded from the phidrates database. Retrun a
    # dictionatry containing the data. If convert_wavelength_units then field
    # Lambda in A is removed and replaced with wavelength in nm."""
    # ## open, awk filter the header
    # data = tools.file_to_recarray(filename,
           # awkfilter='''
           # BEGIN{data_found=0};
           # data_found==1{print};
           # #/^ *Lambda +[a-zA-Z+]/{print "#",$0;data_found=1;};
           # /^0 *Branching ratio for/{printf("#");data_found=1;};
           # '''
           # )
    # if convert_wavelength_units:
        # data['Lambda'] = tools.A2nm(data['Lambda'])
        # t = list(data.dtype.names)
        # t[t.index('Lambda')] = 'wavelength'
        # data.dtype.names = tuple(t)
    # ## rejig keys to my standard type
    # if parse_and_replace_keys:
        # new_data = collections.OrderedDict()
        # new_data['wavelength'] = data['wavelength']
        # new_data['photoabsorption'] = np.zeros(new_data['wavelength'].shape)
        # new_data['photodissociation'] = np.zeros(new_data['wavelength'].shape)
        # new_data['photoionisation'] = np.zeros(new_data['wavelength'].shape)
        # for key in data.dtype.names:
            # if key in ('Total','wavelength',): continue
            # if '+' in key:      # look for (dissociative) ionisation partial cross sections
                # new_data['photoionisation'] += data[key] # add to total photoionisation
                # new_data[key.replace('/',' + ')] = data[key] # translate key
            # else:               # else neutral photodissociation
                # new_data['photodissociation'] += data[key] # add to total photoionisation
                # new_data[key.replace('/',' + ')] = data[key] # translate key
        # new_data['photoabsorption'] = new_data['photoionisation'] + new_data['photodissociation']
        # new_data = tools.dict_to_recarray(new_data)
        # data = new_data
    # return(data)

# def phidrates_plot_cross_sections(filename,convert_wavelength_units=True):
    # """Load cross sections from a file from the phidrates database and
    # plot them all."""
    # data = phidrates_load_cross_sections(filename,convert_wavelength_units)
    # ax = plt.gca()
    # for key in list(data.keys()):
        # if key=='wavelength': continue
        # ax.plot(data['wavelength'],data[key],label=key)
    # tools.legend()
    # return(data)

# def phidrates_calc_branching_ratios(filename):
    # """Load a phidrates cross section file, convert partical cross
    # sections into branching ratios, also for subcategories of all
    # neutral products and ionised products."""
    # data = tools.recarray_to_dict(phidrates_load_cross_sections(filename))
    # wavelength = data.pop('wavelength')
    # Total = data.pop('Total')

    # all_ratios = collections.OrderedDict()
    # all_ratios['wavelength'] = wavelength
    # for key in data: all_ratios[key] = data[key]/Total
    # keys = [key for key in data if not re.match(r'.*\+.*',key)]
    # norm = sum([data[key] for key in keys])
    # neutral_ratios = collections.OrderedDict()
    # neutral_ratios['wavelength'] = wavelength
    # for key in keys: neutral_ratios[key] = data[key]/norm
    # keys = [key for key in data if re.match(r'.*\+.*',key)]
    # norm = sum([data[key] for key in keys])
    # ionic_ratios = collections.OrderedDict()
    # ionic_ratios['wavelength'] = wavelength
    # for key in keys: ionic_ratios[key] = data[key]/norm
    # ## ensure nans are zero
    # for t in list(all_ratios.values())+list(neutral_ratios.values())+list(ionic_ratios.values()):
        # t[np.isnan(t)] = 0.
    # return(all_ratios,neutral_ratios,ionic_ratios)

# def phidrates_plot_branching_ratios(filename,fig=None):
    # """Load a phidrates cross section file, convert partical cross
    # sections into branching ratios, also for subcategories of all
    # neutral products and ionised products."""
    # all_ratios,neutral_ratios,ionic_ratios = phidrates_calc_branching_ratios(filename)
    # if fig==None: fig = plt.gcf()
    # fig.clf()
    # ax = tools.subplot_append()
    # for key in all_ratios:
        # if key=='wavelength':continue
        # ax.plot(all_ratios['wavelength'],all_ratios[key],label=key)
    # tools.legend()
    # tools.annotate_corner('all branching')
    # ax.grid(True)
    # ax = tools.subplot_append()
    # for key in neutral_ratios:
        # if key=='wavelength':continue
        # ax.plot(neutral_ratios['wavelength'],neutral_ratios[key],label=key)
    # tools.legend()
    # tools.annotate_corner('neutral species')
    # ax.grid(True)
    # ax = tools.subplot_append()
    # for key in ionic_ratios:
        # if key=='wavelength':continue
        # ax.plot(ionic_ratios['wavelength'],ionic_ratios[key],label=key)
    # tools.legend()
    # tools.annotate_corner('ionic species')
    # ax.grid(True)
   #  
# def plot_shielding_function(filename):
    # """Requires a standrad file format"""
    # ax = plt.gca()
    # d = tools.file_to_recarray(filename)
    # x = d['N']
    # for key in d.dtype.names:
        # if key=='N': continue
        # y = d[key]
        # if all(np.isnan(y)): continue
        # ax.plot(x,y,label=key)
    # tools.legend()
    # ax.grid(True)
    # if len(filename) > 50:
        # ax.set_title(tools.rootname(filename),fontsize=10)
    # else:
        # ax.set_title(filename,fontsize=10)
    # ax.set_ylabel('Shielding function')
    # ax.set_xlabel('Column density (cm-2)')
    # ax.set_xscale('log')
    # ax.set_ylim(0.,1.)
    # return ax

# def black_body_radiation(
        # temperature,            # K
        # wavelength,             # nm
        # output_units='photons.s-1.cm-2.A-1'):
    # """Calculate black-body emissive power (or number of photons per
    # second) per unit area per spectral unit at a certain wavelength
    # and temperature. Input temperature in K, wavelength in nm.
    # \nPossible output units:
        # - J.s-1.m-2.Hz-1
        # - J.s-1.m-2.nm-1
        # - erg.s-1.cm-2.A-1
        # - photon.s-1.cm-2.A-1
        # - photon.s-1.cm-2.(cm-1)-1
        # - J.s-1.cm-2.(cm-1)-1
    # """
    # frequency = tools.nm2Hz(wavelength)
    # wavelengthSI = wavelength*1e-9 # m
    # if output_units=='J.s-1.m-2.Hz-1':
        # intensity = 2.*constants.pi*constants.h*frequency**3/constants.c**2 / (np.exp(constants.h*frequency/constants.k/temperature)-1)
    # elif output_units=='J.s-1.m-2.nm-1':
        # intensity = 2.*constants.pi*constants.h*constants.c**2/wavelengthSI**5 / (np.exp(constants.h*constants.c/wavelengthSI/constants.k/temperature)-1)
        # intensity = intensity*1e-9
    # elif output_units=='erg.s-1.cm-2.A-1':
        # intensity = black_body_radiation(temperature,wavelength,output_units='J.s-1.m-2.nm-1')
        # intensity = tools.J2erg(intensity)*1e-4*1e-1
    # elif output_units=='photon.s-1.cm-2.A-1':
        # intensity = black_body_radiation(temperature,wavelength,output_units='J.s-1.m-2.nm-1')
        # intensity = intensity/constants.h/frequency*1e-4*1e-1
    # elif output_units=='photon.s-1.cm-2.(cm-1)-1':
        # intensity = black_body_radiation(temperature,wavelength,output_units='J.s-1.m-2.nm-1')/constants.h/frequency*1e-4/tools.dnm2dk(1,wavelength)
    # elif output_units=='J.s-1.cm-2.(cm-1)-1':
        # intensity = black_body_radiation(temperature,wavelength,output_units='J.s-1.m-2.nm-1')*1e-4/tools.dnm2dk(1,wavelength)
    # else:
        # raise Exception("output_units not known: "+repr(output_units))
    # return intensity

# def load_cross_section(
        # species,
        # products='photoabsorption', # 'photoabsorption','photodissociation','photoionisation', or perhaps others
        # xunits='nm',                # or cm-1
        # yunits='cm2',
        # x=None   # specify a y grid to resample onto, in xunits, None for whatever is in data file
# ):
    # """Load a cross section from a standard location."""
    # xgrid = x                    # the output x grid requested, preserve x name for below
    # data_filename = tools.expand_path(f'~/data/species/{species}/cross_sections/all_cross_sections.hdf5')
    # assert os.path.exists(data_filename),f'Cross sections for {repr(species)} not found, expected in {repr(data_filename)}'
    # data = tools.file_to_recarray(data_filename) # load data
    # x,y = data['wavelength'],data[products]   # get requested cross section
    # if   xunits=='nm': pass                   # convert units
    # elif xunits=='cm-1': x,y = tools.nm2k(x)[::-1],y[::-1]
    # else: raise Exception(f"Unknown xunits: {repr(xunits)}")
    # if   yunits=='cm2': pass    # convert units
    # else: raise Exception(f"Unknown yunits: {repr(yunits)}")
    # if xgrid is not None: x,y = xgrid,resample_data(x,y,xgrid) # resample xgrid if requested
    # return(x,y)

# def load_radiation_field(radiation_field_name,units='photon.cm-2.s-1.nm-1',x=None):
    # """Load a radiation field into an array by name, units
    # [wavelength(nm),intensity(photon.cm-2.s-1.nm-1). """
    # xout = x                    # output x, or None for whatever is in data file
    # def identity(x): return x
    # def times10(x): return x*10
    # for (name,filename,convert_wavelength,convert_intensity) in (
            # ('unit',None,None,None),
            # ('draine1978','~/projects/astrochem/radiation_fields/draine1978',tools.A2nm,times10,),
            # ('gondhalekar1980','~/projects/astrochem/radiation_fields/gondhalekar1980',tools.A2nm,times10,),
            # ('mathis1983','~/projects/astrochem/radiation_fields/mathis1983/DG=10',tools.A2nm,times10,),
            # ('mathis1983_normalised_to_draine1978','~/projects/astrochem/radiation_fields/mathis1983/DG=10_normalised_to_draine1978',tools.A2nm,times10,),
            # ('habing1968','~/projects/astrochem/radiation_fields/habing1968',tools.A2nm,times10,),
            # ('ISRF','~/projects/astrochem/radiation_fields/van_dishoeck1988',tools.A2nm,times10,),
            # ('van_dishoeck1988','~/projects/astrochem/radiation_fields/van_dishoeck1988',tools.A2nm,times10,),
            # ('blackbody_4000K','~/projects/astrochem/radiation_fields/black_body_4000K',tools.A2nm,times10,),
            # ('blackbody_5778K','~/projects/astrochem/radiation_fields/black_body_5778K',tools.A2nm,times10,),
            # ('blackbody_10000K','~/projects/astrochem/radiation_fields/black_body_10000K',tools.A2nm,times10,),
            # ('blackbody_20000K','~/projects/astrochem/radiation_fields/black_body_20000K',tools.A2nm,times10,),
            # ('blackbody_4000K_cutoff_912A','~/projects/astrochem/radiation_fields/black_body_4000K_cutoff_912A',tools.A2nm,times10,),
            # ('blackbody_5778K_cutoff_912A','~/projects/astrochem/radiation_fields/black_body_5778K_cutoff_912A',tools.A2nm,times10,),
            # ('blackbody_10000K_cutoff_912A','~/projects/astrochem/radiation_fields/black_body_10000K_cutoff_912A',tools.A2nm,times10,),
            # ('blackbody_20000K_cutoff_912A','~/projects/astrochem/radiation_fields/black_body_20000K_cutoff_912A',tools.A2nm,times10,),
            # ('Lyman-alpha_100km.s-1','~/data/astro/radiation_fields/Lyman-alpha_100km.s-1',tools.A2nm,times10,),
            # ('Lyman-alpha_400km.s-1','~/data/astro/radiation_fields/Lyman-alpha_400km.s-1',tools.A2nm,times10,),
            # ('solar_heays2017a','~/data/astro/radiation_fields/solar_spectrum/combined_spectrum',None,None,),
            # ('solar_chance2010','~/data/astro/radiation_fields/solar_spectrum/chance2010_spectrum/spectrum.mod',None,None,),
            # ('solar_normalised_draine1978','~/data/astro/radiation_fields/solar_spectrum/combined_spectrum_normalised_to_draine1978',tools.A2nm,times10,),
            # ('solar_claire2012_modern','~/data/astro/radiation_fields/solar_spectrum/claire2012_spectra/SunModern.txt',None,None,),
            # ('france2014_V4046SGR','~/data/astro/radiation_fields/france2014/V4046SGR_allFUV_int_n_2013a.txt',tools.A2nm,times10,),
            # ('france2014_AATAU','~/data/astro/radiation_fields/france2014/AATAU_allFUV_int_n_2013a.txt.hdf5',tools.A2nm,times10,),
            # ('france2014_BPTAU','~/data/astro/radiation_fields/france2014/BPTAU_allFUV_int_n_2013a.txt.hdf5',tools.A2nm,times10,),
            # ('france2014_DETAU','~/data/astro/radiation_fields/france2014/DETAU_allFUV_int_n_2013a.txt.hdf5',tools.A2nm,times10,),
            # ('france2014_DFTAU','~/data/astro/radiation_fields/france2014/DFTAU_allFUV_int_n_2013a.txt.hdf5',tools.A2nm,times10,),
            # ('france2014_DMTAU','~/data/astro/radiation_fields/france2014/DMTAU_allFUV_int_n_2013a.txt.hdf5',tools.A2nm,times10,),
            # ('france2014_DRTAU','~/data/astro/radiation_fields/france2014/DRTAU_allFUV_int_n_2013a.txt.hdf5',tools.A2nm,times10,),
            # ('france2014_GMAUR','~/data/astro/radiation_fields/france2014/GMAUR_allFUV_int_n_2013a.txt.hdf5',tools.A2nm,times10,),
            # ('france2014_HNTAU','~/data/astro/radiation_fields/france2014/HNTAU_allFUV_int_n_2013a.txt.hdf5',tools.A2nm,times10,),
            # ('france2014_LKCA15','~/data/astro/radiation_fields/france2014/LKCA15_allFUV_int_n_2013a.txt.hdf5',tools.A2nm,times10,),
            # ('france2014_RECX11','~/data/astro/radiation_fields/france2014/RECX11_allFUV_int_n_2013a.txt.hdf5',tools.A2nm,times10,),
            # ('france2014_RECX15','~/data/astro/radiation_fields/france2014/RECX15_allFUV_int_n_2013a.txt.hdf5',tools.A2nm,times10,),
            # ('france2014_RULUPI','~/data/astro/radiation_fields/france2014/RULUPI_allFUV_int_n_2013a.txt.hdf5',tools.A2nm,times10,),
            # ('france2014_SUAUR','~/data/astro/radiation_fields/france2014/SUAUR_allFUV_int_n_2013a.txt.hdf5',tools.A2nm,times10,),
            # ('france2014_TWHYA','~/data/astro/radiation_fields/france2014/TWHYA_allFUV_int_n_2013a.txt.hdf5',tools.A2nm,times10,),
            # ('france2014_UXTAU','~/data/astro/radiation_fields/france2014/UXTAU_allFUV_int_n_2013a.txt.hdf5',tools.A2nm,times10,),
            # ('TW-Hya_normalised_draine1978','~/data/astro/radiation_fields/synthetic_stellar_spectra/TW-Hya_nomura2005_france2014_normalised_to_draine1978',tools.A2nm,times10,),
            # ('cosmic_ray_field_oH2-pH2_0-1_b_5km.s-1','~/data/astro/radiation_fields/cosmic_ray_induced_radiation/oH2-pH2_0-1_b_5km.s-1_Angstroms.hdf5',tools.A2nm,times10,),
            # ('cosmic_ray_field_oH2-pH2_3-1_b_5km.s-1','~/data/astro/radiation_fields/cosmic_ray_induced_radiation/oH2-pH2_3-1_b_5km.s-1_Angstroms.hdf5',tools.A2nm,times10,),
            # ('cosmic_ray_field_oH2-pH2_0-1_b_1km.s-1','~/data/astro/radiation_fields/cosmic_ray_induced_radiation/oH2-pH2_0-1_b_1km.s-1_Angstroms.hdf5',tools.A2nm,times10,),
            # ('cosmic_ray_field_oH2-pH2_3-1_b_1km.s-1','~/data/astro/radiation_fields/cosmic_ray_induced_radiation/oH2-pH2_3-1_b_1km.s-1_Angstroms.hdf5',tools.A2nm,times10,),
            # ('cosmic_ray_field_oH2-pH2_0-1_low_resolution','~/data/astro/radiation_fields/cosmic_ray_induced_radiation/from_Ewine_low_resolution_2016-03-23/crfield01-ang.txt',tools.A2nm,times10,),
            # ('cosmic_ray_field_oH2-pH2_3-1_low_resolution','~/data/astro/radiation_fields/cosmic_ray_induced_radiation/from_Ewine_low_resolution_2016-03-23/crfield11-ang.txt',tools.A2nm,times10,),
    # ):
        # if radiation_field_name==name:
            # if name=='unit':
                # x = np.arange(10.,1000.,100.)
                # y = np.ones(x.shape,dtype=float)
            # else:
                # x,y = tools.file_to_array(filename,usecols=(0,1),unpack=True)
                # if convert_wavelength is not None: x = convert_wavelength(x)
                # if convert_intensity is not None: y = convert_intensity(y)
            # break
    # else:
        # raise Exception('Radiation field not found among known radiation fields: '+repr(radiation_field_name))
    # if units=='photon.cm-2.s-1.nm-1':
        # pass
    # elif units=='J.m-3.nm-1':
        # y = tools.photon_intensity_to_energy_density(y,x)
    # if xout is not None:
        # x,y = xout,tools.spline(x,y,xout,order=1)
    # return(x,y)

# def roberge1991_load_grain_properties(filename):
    # """Load a file in the format of the dissoc2 program of
    # roberg1991. E.g., ~/data/astro/grain_properties/calculate_extinction/grains.std"""
    # data = tools.file_to_array('grains.std',comments='c')
    # retval = collections.OrderedDict()
    # retval['index'] = data[:,0]
    # retval['lambda'] = data[:,1]
    # retval['A(l)/Av'] = data[:,2]
    # retval['o(l)']    = data[:,3]
    # retval['g(l)']    = data[:,4]
    # return(retval)

# def roberge1991_plot_grain_properties(filename,fig=None):
    # """Plot all data from a file in the format of the dissoc2 program of
    # roberg1991. E.g., ~/data/astro/grain_properties/calculate_extinction/grains.std"""
    # data = load_roberge1991_grain_properties(filename)
    # if fig==None: fig = plt.gcf()
    # fig.clf()
    # for key in ['A(l)/Av','o(l)','g(l)']:
        # ax = tools.subplot_append()
        # ax.plot(data['lambda'],data[key],label=key)
        # ax.grid(True)
        # tools.annotate_corner(key)
    # ax.set_xlabel('lambda')

# def load_cross_section_leiden_photodissoc_database(
        # filename,
        # linewidth_fwhm=0.1,
        # lineshape='lorentzian', # lorentzian or gaussian
        # fwhms_to_specify = 10. # create this many points around line
# ):
    # """Load a file downloaded from the Leiden photodissoication database. Return a
    # dictionary containing the data with keys:
       # -- lines_wavelength - discrete lines (nm)
       # -- lines_integrated_cross_section - their integrated cross section (cm2 nm)
       # -- continuum_wavelength - continuum data in file (nm)
       # -- continuum_cross_section - cross section data in file (cm2)
       # -- wavelength - combined line and continuum (nm)
       # -- cross_section - combined line and continuum, lines broadened by linewidth_fwhm (nm)
       # -- lineshape - 'gaussian' or 'lorentzian' to broaden lines
    # File is expected to be in Angstrom units.
    # """
    # linewidth_fwhm = float(linewidth_fwhm)
    # fid = open(tools.expand_path(filename),'r')
    # data = dict()
    # ## header information 
    # data['header'] = fid.readline()
    # ## integrated cross sections of lines
    # number_of_lines = int(fid.readline())
    # t = [fid.readline().split()[1:] for i in range(number_of_lines)]
    # data['lines_wavelength'] = np.array([tt[0] for tt in t],dtype=float)
    # data['lines_integrated_cross_section'] = np.array([tt[1] for tt in t],dtype=float)
    # ## change units form [A,cm2.A] to [nm,cm.nm]
    # data['lines_wavelength'] /= 10.
    # data['lines_integrated_cross_section'] /= 10.
    # ## continuum cross section
    # number_of_continuum_points = int(fid.readline())
    # fid.readline()              # skip a -1
    # t = [fid.readline().split()[1:] for i in range(number_of_continuum_points)]
    # data['continuum_wavelength'] = np.array([tt[0] for tt in t],dtype=float)
    # data['continuum_cross_section'] = np.array([tt[1] for tt in t],dtype=float)
    # ## change units to [A,cm2] to [nm,cm2]
    # data['continuum_wavelength'] = data['continuum_wavelength']/10.
    # fid.close()
    # ## if lines, make a new continuum cross section
    # if number_of_lines>0:
        # ## get domains of line data - possible overlapping
        # domains = []
        # width = linewidth_fwhm*fwhms_to_specify
        # lines = np.sort(data['lines_wavelength'])
        # domains.append([lines[0]-width])
        # for (line1,line2) in zip(lines[0:-1],lines[1:]):
            # if line2-line1 < width*2:
                # continue
            # else:
                # domains[-1].append(line1+width)
                # domains.append([line2-width])
        # domains[-1].append(lines[-1]+width)
        # ## get combined wavelength
        # wavelength = np.unique(np.concatenate((data['continuum_wavelength'],
             # # np.concatenate([np.arange(t0,t1,linewidth_fwhm/10.) for (t0,t1) in domains])
           # np.concatenate([
               # np.concatenate(([t0-0.1*linewidth_fwhm,t1+0.1*linewidth_fwhm], np.arange(t0,t1,linewidth_fwhm/10.) ))
               # for (t0,t1) in domains])
        # )))
        # ## get total cross seciton on this wavelength scale
        # tmp_wavelength = np.concatenate((wavelength,np.array([wavelength[-1]+0.0001]))) # A HACKY BUGFIX TO SOLVE A PROBLEM IN RESAMPLE DATA
        # tmp_cross_section = resample_data(data['continuum_wavelength'],data['continuum_cross_section'],tmp_wavelength)
        # cross_section = tmp_cross_section[0:-1] # A HACKY BUGFIX TO SOLVE A PROBLEM IN RESAMPLE DATA
        # for (t0,t1) in zip(data['lines_wavelength'],data['lines_integrated_cross_section']):
            # i = (wavelength>=(t0-width))&(wavelength<=(t0+width))
            # if lineshape=='gaussian':
                # cross_section[i] += tools.gaussian(wavelength[i],fwhm=linewidth_fwhm,mean=t0)*t1
            # elif lineshape=='lorentzian':
                # cross_section[i] += tools.lorentzian(wavelength[i],Gamma=linewidth_fwhm,k0=t0)*t1
            # else:
                # raise InputError('Invald lineshape: '+repr(lineshape))
        # data['wavelength'],data['cross_section'] = wavelength,cross_section
    # else:
        # data['wavelength'],data['cross_section'] = data['continuum_wavelength'],data['continuum_cross_section']
    # return(data)

# def save_cross_section_leiden_photodissoc_database(
        # filename,
        # header,
        # lines_wavelength=[],
        # lines_integrated_cross_section=[],
        # continuum_wavelength=[],
        # continuum_cross_section=[],
# ):
    # """Save a cross section into the file format of the Leiden
    # photodissoication database. LINES NOT IMPLEMENTED!!!. Input
    # wavelengths are expected in nm, but printed in files in Angstroms."""
    # lines_wavelength,lines_integrated_cross_section = np.array(lines_wavelength),np.array(lines_integrated_cross_section)
    # continuum_wavelength,continuum_cross_section = np.array(continuum_wavelength),np.array(continuum_cross_section)
    # ## ## do not include zero cross section
    # ## i = continuum_cross_section>0
    # ## continuum_wavelength,continuum_cross_section = continuum_wavelength[i],continuum_cross_section[i]
    # ## convert to Angstroms
    # lines_wavelength = tools.nm2A(np.array(lines_wavelength,ndmin=1,dtype=float)) # convert to Angstroms
    # lines_integrated_cross_section *= 10. # convert to cm2.Angstroms
    # continuum_wavelength = tools.nm2A(np.array(continuum_wavelength,ndmin=1,dtype=float)) # convert to Angstroms
    # ## begin file data
    # lines = []
    # lines.append(header.strip())
    # ## IMPLEMENT LINES HERE
    # lines.append(format(len(lines_wavelength))) # number of discrete lines
    # for i,(wavelength,cross_section) in enumerate(zip(lines_wavelength,lines_integrated_cross_section)):
        # lines.append(' '.join([format(i,'4d'),format(wavelength,'15.5f'),format(cross_section,'15.5e')]))
    # ## continuum cross section
    # lines.append(str(len(continuum_wavelength))) # number of continuum points
    # if len(continuum_wavelength)>0:
        # # lines.append(format(-(int(np.floor(continuum_wavelength[-1])+1)),'d')) # longest-wavelength threshold of continuum 
        # lines.append('-1')
        # lines.extend([
            # format(i,'>4d')+'  '+format(wavelength,'10.5f')+'  '+format(cross_section,'0.3e')
            # for (i,(wavelength,cross_section)) in enumerate(zip(continuum_wavelength,continuum_cross_section))
        # ])
    # ## write to file
    # fid = open(filename,'w')
    # fid.write('\n'.join(lines))
    # fid.close()

# def plot_cross_section_leiden_photodissoc_database(
        # filename,
        # plot_total=True,
        # ylog=False,
        # label = None,
        # linestyle = '-',
        # ax=None,
        # color=None,
        # **load_kwargs):
    # """Load and plot a cross section file in the format of the Leiden
    # photodissoication database"""
    # data = load_cross_section_leiden_photodissoc_database(filename,**load_kwargs)
    # if label is None: label = filename
    # label = label+' datapoints: '+str(len(data['continuum_wavelength'])+len(data['lines_wavelength'])) # TEMPORARY HACK
    # if ax is None:
        # fig = plt.gcf()
        # fig.clf()
        # ax = fig.gca()
        # # ax.set_title(filename)
    # if plot_total:
        # ax.plot(data['wavelength'],data['cross_section'],
                # color=(tools.newcolor(0) if color is None else color),
                # zorder=5)
    # else:
        # if len(data['continuum_wavelength'])>0:
            # ax.plot(data['continuum_wavelength'],data['continuum_cross_section'],
                    # color=(tools.newcolor(1) if color is None else color),
            # )
        # if len(data['lines_wavelength'])>0:
            # for (x,y) in zip(data['lines_wavelength'],data['lines_integrated_cross_section']):
                # ax.plot([x,x],[0,y],
                        # color=(tools.newcolor(2) if color is None else color),
                        # )
    # ax.plot([],[],
            # color=('black' if color is None else color),
            # label=label,
            # linestyle=linestyle,
    # ) # for legend
    # tools.legend(ax=ax)
    # ax.grid(True,color='gray')
    # if ylog:
        # ax.set_yscale('log')
        # ax.set_ylim(ymin=1e-20)
    # else:
        # ax.set_ylim(ymin=0)
    # ax.set_xlim((data['wavelength'][0],data['wavelength'][-1]))
    # ax.set_xlabel("Wavelength (nm)")
    # # ax.set_title("Cross section (cm$^2$)\nIntegrated cross section ( cm²$\AA$)")
    # ax.set_ylabel("Cross section (cm$^2$)\nIntegrated cross section ( cm²$\AA$)")
    # return(data)

# def load_bruker_opus_data(filename,datablock=None):
    # """Load a binary Bruker Opus file, returning a specific datablock as well as all data in a
    # dictionary. Useful datablocks:
    # IgSm:  the single-channel sample interferogram
    # ScSm:  the single-channel sample spectrum
    # IgRf:  the reference (background) interferogram
    # ScRf:  the reference (background) spectrum
    # """
    # import brukeropusreader
    # d = brukeropusreader.read_file(tools.expand_path(filename))
    # if datablock is None:
        # return(d)
    # assert f'{datablock}' in d, f'Cannot find datablock: {repr(datablock)}'
    # parameters = d[f'{datablock} Data Parameter']
    # x = np.linspace(parameters['FXV'], parameters['LXV'], parameters['NPT'])
    # y = d[datablock]
    # return(x,y,parameters)

# def load_bruker_opus_spectrum(filename,datablock='ScSm'):
    # return(load_bruker_opus_data(filename,datablock))
   #  
# ## dictionary mapping for converting species names
# _species_name_translation_dict = dict(
    # matplotlib = bidict({
        # '14N2':'${}^{14}$N$_2$',
        # '12C18O':r'${}^{12}$C${}^{18}$O',
        # '32S16O':r'${}^{32}$S${}^{16}$O',
        # '33S16O':r'${}^{33}$S${}^{16}$O',
        # '34S16O':r'${}^{34}$S${}^{16}$O',
        # '36S16O':r'${}^{36}$S${}^{16}$O',
        # '32S18O':r'${}^{32}$S${}^{18}$O',
        # '33S18O':r'${}^{33}$S${}^{18}$O',
        # '34S18O':r'${}^{34}$S${}^{18}$O',
        # '36S18O':r'${}^{36}$S${}^{18}$O',
    # }),
    # leiden = bidict({'Ca':'ca', 'He':'he',
                              # 'Cl':'cl', 'Cr':'cr', 'Mg':'mg', 'Mn':'mn',
                              # 'Na':'na', 'Ni':'ni', 'Rb':'rb', 'Ti':'ti',
                              # 'Zn':'zn', 'Si':'si', 'Li':'li', 'Fe':'fe',
                              # 'HCl':'hcl', 'Al':'al', 'AlH':'alh',
                              # 'LiH':'lih', 'MgH':'mgh', 'NaCl':'nacl',
                              # 'NaH':'nah', 'SiH':'sih', 'Co':'cob'}),
    # meudon_pdr = ({'Ca':'ca', 'Ca+':'ca+', 'He':'he', 'He+':'he+',
                   # 'Cl':'cl', 'Cr':'cr', 'Mg':'mg', 'Mn':'mn', 'Na':'na', 'Ni':'ni',
                   # 'Rb':'rb', 'Ti':'ti', 'Zn':'zn', 'Si':'si', 'Si+':'si+',
                   # 'Li':'li', 'Fe':'fe', 'Fe+':'fe+', 'HCl':'hcl', 'HCl+':'hcl+',
                   # 'Al':'al', 'AlH':'alh', 'h3+':'H3+', 'l-C3H2':'h2c3' ,
                   # 'l-C3H':'c3h' , 'l-C4':'c4' , 'l-C4H':'c4h' , 'CH3CN':'c2h3n',
                   # 'CH3CHO':'c2h4o', 'CH3OCH3':'c2h7o', 'C2H5OH':'c2h6o',
                   # 'CH2CO':'c2h2o', 'HC3N':'c3hn', 'e-':'electr', # not sure
                   # ## non chemical processes
                   # 'phosec':'phosec', 'phot':'phot', 'photon':'photon', 'grain':'grain',
                   # # '?':'c3h4',                 # one of either H2CCCH2 or H3CCCH
                   # # '?':'c3o',                  # not sure
                   # # '?':'ch2o2',                  # not sure
# }))

# ## functions for converting a species name
# _species_name_translation_functions = {}

# def _f(name):
    # """Translate form my normal species names into something that
    # looks nice in matplotlib."""
    # name = re.sub(r'([0-9]+)',r'$_{\1}$',name) # subscript multiplicity 
    # name = re.sub(r'([+-])',r'$^{\1}$',name) # superscript charge
    # return(name)
# _species_name_translation_functions[('standard','matplotlib')] = _f

# def _f(leiden_name):
    # """Translate from Leidne data base to standard."""
    # ## default to uper casing
    # name = leiden_name.upper()
    # name = name.replace('C-','c-')
    # name = name.replace('L-','l-')
    # ## look for two-letter element names
    # name = name.replace('CL','Cl')
    # name = name.replace('SI','Si')
    # name = name.replace('CA','Ca')
    # ## look for isotopologues
    # name = name.replace('C*','13C')
    # name = name.replace('O*','18O')
    # name = name.replace('N*','15N')
    # ## assume final p implies +
    # if name[-1]=='P' and name!='P':
        # name = name[:-1]+'+'
    # return name
# _species_name_translation_functions[('leiden','standard')] = _f

# def _f(standard_name):
    # """Translate form my normal species names into the Leiden database
    # equivalent."""
    # standard_name  = standard_name.replace('+','p')
    # return standard_name.lower()
# _species_name_translation_functions[('standard','leiden')] = _f

# def _f(name):
    # return(name)
# _species_name_translation_functions[('kida','standard')] = _f

# def _f(name):
    # """Makes a nice latex version of species. THIS COULD BE EXTENDED"""
    # try:
        # return(database.get_species_property(name,'latex'))
    # except:
        # return(r'\ce{'+name.strip()+r'}')
# _species_name_translation_functions[('standard','latex')] = _f

# def _f(name):
    # """Standard to Meudon PDR with old isotope labellgin."""
    # name = re.sub(r'\([0-9][SPDF][0-9]?\)','',name) # remove atomic terms e.g., O(3P1) ⟶ O
    # for t in (('[18O]','O*'),('[13C]','C*'),('[15N]','N*'),): # isotopes
        # name = name.replace(*t)
    # return(name.lower())
# _species_name_translation_functions[('standard','meudon old isotope labelling')] = _f

# def _f(name):
    # """Standard to Meudon PDR."""
    # name = re.sub(r'\([0-9][SPDF][0-9]?\)','',name) # remove atomic terms e.g., O(3P1) ⟶ O
    # for t in (('[18O]','_18O'),('[13C]','_13C'),('[15N]','_15N'),): # isotopes
        # name = name.replace(*t)
    # name = re.sub(r'^_', r'' ,name)
    # name = re.sub(r' _', r' ',name)
    # name = re.sub(r'_ ', r' ',name)
    # name = re.sub(r'_$', r'' ,name)
    # name = re.sub(r'_\+',r'+',name)
    # return(name.lower())
# _species_name_translation_functions[('standard','meudon')] = _f
           #  
# def translate_species(name,input_encoding,output_encoding):
    # """Translate species name between different formats. standard, kida, leiden, meudon, latex, matplotlib. """
    # ## vectorise
    # if not np.isscalar(name):
        # return([translate_species(namei,input_encoding,output_encoding) for namei in name])
    # ## check translation dictionaries first
    # if (output_encoding in _species_name_translation_dict
        # and name in _species_name_translation_dict[output_encoding]):
        # return(_species_name_translation_dict[output_encoding][name])
    # ## use a function
    # if (input_encoding,output_encoding) in _species_name_translation_functions:
        # return(_species_name_translation_functions[(input_encoding,output_encoding)](name))
    # ## try passing through 'standard' form
    # if input_encoding=='standard' or output_encoding=='standard':
        # raise Exception(f'Do not know how to translate species name betwen {repr((input_encoding,output_encoding))}')
    # return(
        # translate_species(
            # translate_species(name,input_encoding,'standard'),
            # 'standard',output_encoding))

# def translate_species_to_standard(encoding,name):
    # """Translate some species encoding to the standard one."""
    # if not np.isscalar(name):
        # return([translate_reaction_to_standard(encoding,namei) for namei in name])
    # if encoding=='meudon':
        # for t0,t1 in _direct_translations_meudon_pdr: # see if in explicit translation dict
            # if t1==name: return(t0)
        # else:                # a chemical species
            # name = name.upper() # convert to capitalised element labels 
            # name = name.replace('CL','Cl')
            # name = name.replace('HE','He') 
            # name = name.replace('AR','Ar') 
            # name = name.replace('AL','Al') 
            # name = name.replace('CR','Cr') 
            # name = name.replace('CA','Ca') 
            # name = name.replace('C-','c-') # strucutre
            # name = name.replace('LI','Li')
            # name = name.replace('MG','Mg')
            # name = name.replace('MN','Mn')
            # name = name.replace('RB','Rb')
            # name = name.replace('NA','Na')
            # name = name.replace('L-','l-')
            # name = name.replace('C*','[13C]')
            # name = name.replace('O*','[18O]')
            # name = name.replace('N*','[15N]')
            # return(name)
    # else:
        # raise Exception(f"Unknown reaction format: {repr(encoding)}")

# def translate_reaction_to_standard(encoding,name):
    # """Translate some other reaction encoding to the standard one."""
    # if not np.isscalar(name): return([translate_reaction_to_standard(encoding,namei) for namei in name])
    # if encoding in ('meudon','meudon_alternative_isotopes'):
        # tokens = name.split()
        # for i,token in enumerate(tokens):
            # if token=='+':
                # pass
            # elif token=='=':
                # tokens[i] = '⟶' # different reactands/products separator
            # else:                                  # a chemical species
                # tokens[i] = translate_species_to_standard(encoding,token)
        # return(' '.join(tokens))
    # else:
        # raise Exception(f"Unknown reaction encoding: {repr(encoding)}")

# def translate_reaction_from_standard(encoding,name):
    # """Translate some other reaction encoding to the standard one."""
    # if not np.isscalar(name): return([translate_reaction_from_standard(encoding,namei) for namei in name])
    # if encoding in ('meudon','meudon_alternative_isotopes'):
        # tokens = name.split()
        # for i,token in enumerate(tokens):
            # if token=='+':
                # pass
            # elif token=='⟶':
                # tokens[i] = '=' # different reactands/products separator
            # else:                                  # a chemical species
                # tokens[i] = translate_species_from_standard(encoding,token)
        # return(' '.join(tokens))
    # else:
        # raise Exception(f"Unknown reaction encoding: {repr(encoding)}")

# def top_base_load_all_cross_sections(filename='~/data/atoms/opacity_project_TOPbase/all_data_level_ordering_2015-05-01'):
    # """Load opacity project top base photoionisation cross section for
    # atomic number NZ, electron number NE, and energy level ILV, data in
    # given filename."""
    # fid = open(tools.expand_path(filename),'r')
    # data = []
    # global x,y
    # def add_data():
        # global x,y
        # x=np.array(x)
        # y=np.array(y)
        # wavelength = (tools.eV2nm(tools.Ry2eV(x)))[-1::-1]
        # cross_section = tools.Mb2cm2(y)[-1::-1]
        # data.append(dict(wavelength=wavelength,cross_section=cross_section,x=x,y=y,I=I,NZ=NZ,NE=NE,ISLP=ISLP,ILV=ILV,E=E,NP=NP,S=S,L=L,P=P))
    # data_found = False
    # for line in fid.readlines():
        # tokens = line.split()
        # if len(tokens)==0 or tokens[0][0]in ['=','I']: continue 
        # if len(tokens)==7:
            # if int(tokens[0])==0: continue #  no data
            # if data_found: add_data()
            # else:          data_found = True
            # I = int(tokens[0])  # arbitrary index
            # NZ = int(tokens[1]) # atomic number
            # NE = int(tokens[2]) # electron number
            # ISLP = int(tokens[3]) # code for quatnum numb ers
            # S = ((ISLP-ISLP%100)/100. - 1. ) /2. # decompose code into spin
            # L = int(((ISLP-ISLP%10)%100)/10.)    # angular mom
            # P = int(ISLP%10)                     # parity
            # ILV = int(tokens[4])                 # excited level index
            # E = float(tokens[5])                 # threshold energy
            # NP = int(tokens[6])                  # number of data points
            # x = []                               # photon energy (Ry)
            # y = []                               # cross section Mb
        # if len(tokens)==2:
            # x.append(float(tokens[0]))
            # y.append(float(tokens[1]))
    # ## final dataset
    # add_data()
    # return(data)

# def top_base_load_cross_section(NZ,NE,ILV,S,L,P,filename='~/data/atoms/opacity_project_TOPbase/all_data_level_ordering_2015-05-01'):
    # """Convenience function - get one cross section out of top base
            # NZ = atomic number
            # NE = electron number
            # ILV = excited ion level
            # S =  spin multiplicity (actually 2S+1)
            # L = angular mom
            # P = parity """
    # for data in top_base_load_all_cross_sections(filename=filename):
        # if(
                # (data['NZ']<=NZ) 
                # &(data['NE']==NE) 
                # &(data['ILV']==ILV) 
                # &(data['S']==S) 
                # &(data['L']==L) 
                # &(data['P']==P)
        # ):
            # return(data)
            # # print data['NZ'],data['NE'],data['ISLP'],data['ILV'],data['S'],data['L'],data['P']


# elements = [ 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt',]
# atoms = elements + [t+'+' for t in elements] + [t+'++' for t in elements] + [t+'+++' for t in elements] + [t+'++++' for t in elements] + [t+'-' for t in elements]
# molecules = ['H2', 'H2+', 'H3+', 'HeH+',
             # 'CH', 'CH+', 'CH2', 'CH2+', 'CH3', 'CH4', 'CH4+',
             # 'C2', 'C2H', 'C2H2', 'C2H4', 'C2H6',
             # 'C3', 'l-C3H', 'c-C3H', 'HC3H', 'l-C3H2', 'c-C3H2',
             # 'l-C4', 'l-C4H', 'l-C5H',
             # 'OH', 'OH+', 'H2O','H2O+',
             # 'O2', 'O2+',
             # 'HO2', 'H2O2',
             # 'O3',
             # 'CO', 'CO+','CO2',
             # 'HCO', 'HCO+', 'H2CO',
             # 'NH', 'NH+', 'NH2', 'NH3',
             # 'N2', 'NO', 'NO2', 'N2O',
             # 'CN', 'HCN', 'HC3N',
             # 'CH3OH', 'CH3CN', 'CH3SH', 'CH3CHO', 'CH3NH2', 'NH2CHO',
             # 'C2H5OH', 'C3H7OH',
             # 'SH', 'SH+', 'H2S',
             # 'CS', 'CS2', 'COS', 'OCS',
             # 'S2','SO', 'SO2',
             # 'CH3SH',
             # 'SiH', 'SiH+', 'SiO',
             # 'HCl', 'HCl+', 'NaCl',
             # 'HF', 'CF+', 
             # 'PH', 'PH+', 'PN?',
             # 'AlH', 'ArH+', 'LiH', 'MgH', 'NaH',]

# def sort_species(list_of_species):
    # """Sort chemical species into some arbitrary order. Originally for
    # presenation in heays2016."""
    # ## atoms
    # # ## atomic anions
    # # groups.append(['H-',])
    # # ## hydrides
    # # groups.append(['HCl', 'LiH', 'MgH', 'AlH', 'NaH', 'NH', 'OH', 'PH', 'SH', 'SiH',])
    # # ## other diatomics
    # # groups.append(['H2', 'CO', 'N2', 'CN', 'CS', 'NO', 'O2', 'SO', 'SiO', 'NaCl',])
    # # ## unsaturated carbon chains and 
    # # ## molecular cations
    # # groups.append(['H2+', 'CO+', 'CH+', 'H3+', 'HCO+', 'HCl+', 'NH+', 'O2+', 'OH+', 'PH+', 'SH+', 'SiH+', 'CH2+ CH4+',])
    # ## look for species not included
    # for t in list_of_species:
        # if t not in atoms + molecules:
            # raise Exception('species not in reference lists: '+str(t))
    # ## sort atoms, return indices leading to this sorted subset
    # i = tools.find(tools.isin(list_of_species,atoms))
    # j = np.argsort([atoms.index(t)for t in list_of_species[i]])
    # iatoms = i[j]
    # ## sort molecules, return indices leading to this sorted subset
    # i = tools.find(tools.isin(list_of_species,molecules))
    # j = np.argsort([molecules.index(t)for t in list_of_species[i]])
    # imolecules = i[j]
    # return(iatoms,imolecules)

# def get_common_reactions(reaction_network1,reaction_network2):
    # return(tools.common([t.name for t in reaction_network1.reactions],
                     # [t.name for t in reaction_network2.reactions]))


# def calibrate_integral_factor(x0,y0,x1,y1,xbeg=None,xend=None):
    # """Calculate scale factor integrate(x0,y0)/integrate(x1,y1). If xbeg
    # and xend not given then calibration is made over the mutual domain
    # of x. """
    # from scipy import interpolate
    # ## ensure data is sorted and unique
    # x0,i = np.unique(x0,return_index=True);y0=y0[i]
    # x1,i = np.unique(x1,return_index=True);y1=y1[i]
    # ## reduce to mutual x domain
    # if xbeg is None: xbeg = max(x0.min(),x1.min())
    # if xend is None: xend = min(x0.max(),x1.max())
    # ## get integral, linearly interpolating
    # int0 = interpolate.UnivariateSpline(x0,y0,s=0,k=1).integral(xbeg,xend)
    # int1 = interpolate.UnivariateSpline(x1,y1,s=0,k=1).integral(xbeg,xend)
    # return(int1/int0)
   #  
# def get_interpolated_N2_spectrum(
        # temperature,    # K
        # species='14N2', # or '14N15N' or '15N2'
        # process='photoabsorption',   # or 'photodissociation'
        # λbeg=None,λend=None, # restrict to this range (nm)
        # λ=None,              # spline to this grid (nm)
# ):
    # """Get an N2 cross section interpolated in temperature."""
    # assert species in ('14N2','14N15N','15N2'),"bad species must be 14N2','14N15N','15N2'"
    # ## catalogue all precomputed N2 cross sections
    # if process=='photoabsorption':
        # filenames = np.array(tools.myglob('/home/heays/cse/model/thesis/cross_sections/'+species+r'/convolved_to_temperature/*K.hdf5'))
        # temperatures = np.array([float(re.sub(r'/home/heays/cse/model/thesis/cross_sections/'+species+r'/convolved_to_temperature/(.*)K\.hdf5',r'\1',t)) for t in filenames])
    # elif process=='photodissociation':
        # filenames = np.array(tools.myglob('/home/heays/projects/N2_dissociation_fraction_correction/'+species+r'/dissociation_fraction_corrected_cross_sections/*K.hdf5'))
        # temperatures = np.array([float(re.sub(r'/home/heays/projects/N2_dissociation_fraction_correction/'+species+r'/dissociation_fraction_corrected_cross_sections/(.*)K\.hdf5',r'\1',t)) for t in filenames])
    # else:
        # raise Exception("Unknown process "+repr(process))
    # i = np.argsort(temperatures)
    # filenames,temperatures = filenames[i],temperatures[i]
    # ## load nearest temperature cross sections
    # i0,i1 = np.argsort(np.abs(temperature-temperatures))[0:2]
    # T0,T1 = temperatures[i0],temperatures[i1]
    # x,y = tools.file_to_array_unpack(filenames[i0])
    # λ0,σ0 = tools.k2nm(x)[-1::-1],y[-1::-1]
    # x,y = tools.file_to_array_unpack(filenames[i1])
    # λ1,σ1 = tools.k2nm(x)[-1::-1],y[-1::-1]
    # if λ is not None:
        # σ0 = tools.spline(λ0,σ0,λ)
        # σ1 = tools.spline(λ1,σ1,λ)
    # else:
        # if λbeg is not None:
            # i = λ0>λbeg
            # λ0,σ0 = λ0[i],σ0[i]
            # i = λ1>λbeg
            # λ1,σ1 = λ1[i],σ1[i]
        # if λend is not None:
            # i = λ0<λend
            # λ0,σ0 = λ0[i],σ0[i]
            # i = λ1<λend
            # λ1,σ1 = λ1[i],σ1[i]
        # if len(λ0)!=len(λ1) or np.any(np.abs(λ0-λ1)>1e-4): λ1,σ1 = λ0,tools.spline(λ1,σ1,λ0) # spline to common grid if necessary
        # λ = λ1
    # return(λ, (T1-temperature)/(T1-T0)*σ1+(temperature-T0)/(T1-T0)*σ0 ) # linearly interpolate between them and return

# def get_infrared_transition(
        # species,
        # νbeg=None,νend=None,
        # σmax=None,
        # Aemax=None,
        # **set_keys_vals
# ):
    # """Convenience function"""
    # import spectra
    # d = load_level_transition(f'~/data/databases/HITRAN/data/{species}/transition.h5')
    # if νbeg is not None:
        # d.limit_to(d['ν']>=νbeg)
    # if νend is not None:
        # d.limit_to(d['ν']<=νend)
    # for key,val in set_keys_vals.items():
        # d[key] = val
    # if σmax is not None:
        # d.limit_to(d['σ']>=σmax)
    # if Aemax is not None:
        # d.limit_to(d['Ae']>=Aemax)
    # return(d)


# def ggchem_calculate_thermochemical_equilibrium(
        # elements_abundances,    # a dict e.g., {'H':1e-3,'C':1e-5,...}
        # P=None,Tmin=None,Tmax=None, # T scan
        # T=None,Pmin=None,Pmax=None, # or P scan
        # Npoints=100,            # number of points if a grid
        # verbose=False,
        # keep_tmpdir=False,
# ):
    # ## inputs
    # if P is not None and Tmin is not None and Tmax is not None:
        # Pmin = Pmax = P
    # elif T is not None and Pmin is not None and Pmax is not None:
        # Tmin = Tmax = T
    # else:
        # raise Exception("Set (P,Tmin,Tmax) or (T,Pmin,Pma).")
    # ## make tmpdir and input files
    # tmpdir = tools.tmpdir()
    # with open(f'{tmpdir}/abundances','w') as fid:
        # for element,abundance in elements_abundances.items():
            # fid.write(f'{element} {abundance}\t\n')
    # element_list = ' '.join(elements_abundances)
    # element_list += ' el'           # add electrons/ions
    # with open(f'{tmpdir}/input','w') as fid:
        # fid.write(f'# selected elements\n')
        # fid.write(f'{element_list}\n')
        # fid.write(f'\n')
        # fid.write(f'# name of files with molecular kp-data\n')
        # fid.write(f'dispol_BarklemCollet.dat                ! dispol_file\n')
        # fid.write(f'dispol_StockKitzmann_withoutTsuji.dat   ! dispol_file2\n')
        # fid.write(f'dispol_WoitkeRefit.dat                  ! dispol_file3\n')
        # fid.write(f'\n')
        # fid.write(f'# abundance options 1=EarthCrust, 2=Ocean, 3=Solar, 4=Meteorites\n')
        # fid.write(f'0                     ! abund_pick\n')
        # fid.write(f'abundances\n')
        # fid.write(f'\n')
        # fid.write(f'# equilibrium condensation?\n')
        # fid.write(f'.false.               ! model_eqcond\n')
        # fid.write(f'\n')
        # fid.write(f'# model options\n')
        # fid.write(f'1                     ! model_dim  (0,1,2)\n')
        # fid.write(f'.true.                ! model_pconst\n')
        # fid.write(f'{Tmax}                  ! Tmax [K]\n')
        # fid.write(f'{Tmin}                  ! Tmin [K]      (if model_dim>0)\n')
        # fid.write(f'{Pmax}                  ! pmax [bar]    (if pconst=.true.)\n')
        # fid.write(f'{Pmin}                  ! pmin [bar]\n')
        # fid.write(f'4.E+19                ! nHmax [cm-3]  (if pconst=.false.)\n')
        # fid.write(f'4.E+19                ! nHmin [cm-3]\n')
        # fid.write(f'{Npoints}                    ! Npoints       \n')
        # fid.write(f'5                     ! NewBackIt  \n')
        # fid.write(f'1000.0                ! Tfast\n')
        # fid.write(f'\n')
    # tools.cp('~/data/chemistry/thermochemistry/GGchem/src16/ggchem16',tmpdir)
    # tools.cptree('~/data/chemistry/thermochemistry/GGchem/data',f'{tmpdir}/data')
    # ## get results
    # status,output = subprocess.getstatusoutput([f'cd {tmpdir} && ./ggchem16 input'])
    # if status!=0:
        # print(output)
        # raise Exception(f'ggchem error status: {status}')
    # if verbose:
        # print( output)
    # ## get resutls in a nice format
    # results = Dynamic_Recarray(**tools.file_to_dict(f'{tmpdir}/Static_Conc.dat',skiprows=2,labels_commented=False))
    # ## delete tmpdir
    # if keep_tmpdir:
        # print(f'tmpdir: {tmpdir}')
    # else:
        # shutil.rmtree(tmpdir)
    # return(results)

# _get_species_cache = {}
# def get_species(name):
    # """Get species from name, assume immutable and potentially
    # cached."""
    # if name not in _get_species_cache:
        # _get_species_cache[name] = Species(name=name)
    # return(_get_species_cache[name])
   #  

# class Species:
    # """Info about a species. Currently assumed to be immutable data only."""

    # def __init__(
            # self,
            # name,
            # elements=None,
            # charge=None,
            # isomer=None,
            # encoding=None,
    # ):

        # ## look for isomer, e..,g c-C3 vs l-C3
        # if r:=re.match('^(.+)-(.*)$',name):
            # self.isomer,name = r.groups()
        # else:
            # self.isomer = None
        # ## decode name
        # ## translate diatomic special cases -- remove one day, e.g., 32S16O to [32]S[16]O
        # name = re.sub(r'^([0-9]+)([A-Z][a-z]?)([0-9]+)([A-Z][a-z]?)$',r'[\1\2][\3\4]',name)
        # name = re.sub(r'^([0-9]+)([A-Z][a-z]?)2$',r'[\1\2]2',name)
        # if encoding is not None:
            # name = translate_species_to_standard(encoding,name)
        # self.name = name # must be unique -- use a hash instead?
        # self._reduced_elements = None            # keys are names of elements, values are multiplicity in ONE molecule
        # self._elements = None            # list of elements, possibly repeated
        # self._element_species = None            # list of elements as Species objects
        # self._charge = None
        # self._nelectrons = None
        # self._mass = None            # of one molecule
        # self._reduced_mass = None   # reduced mass (amu), actually a property
        # self._nonisotopic_name = None # with mass symbols removed, e.g., '[32S][16O]' to 'SO'
        # self.density = None     # cm-3
       #  
    # def _get_nonisotopic_name(self):
        # """NOT WELL TESTED, ADD SPECIAL CASE FOR DEUTERIUM"""
        # if self._nonisotopic_name is None:
            # if r:=re.match(r'^[0-9]+([A-Z].*)',self.name):
                # ## simple atom ,e.g., 13C
                # self._nonisotopic_name = r.group(1)
            # else:
                # self._nonisotopic_name = r.sub(r'\[[0-9]*([A-Za-z]+)\]',r'\1',self.name)
        # return self._nonisotopic_name
       #  
    # nonisotopic_name = property(_get_nonisotopic_name)

    # def _get_reduced_mass(self):
        # if self._reduced_mass is None:
            # self._reduced_mass = database.get_species_property(self.name,'reduced_mass')
        # return(self._reduced_mass)

    # def _set_reduced_mass(self,reduced_mass):
        # self._reduced_mass = reduced_mass

    # reduced_mass = property(_get_reduced_mass,_set_reduced_mass)
   #  
    # def _get_mass(self):
        # if self._mass is None:
            # self._mass = database.get_species_property(self.name,'mass')
        # return(self._mass)

    # mass = property(_get_mass)
   #  
    # def _get_charge(self):
        # if self._charge is not None:
            # pass
        # elif self.name=='e-':
            # self._charge = -1
        # elif self.name=='photon':
            # self.charge = 0
        # elif r:=re.match('^(.*)\^([-+][0-9]+)$',self.name): # ddd^+3 / ddd^-3 / ddd^3
            # self._charge = int(r.group(2))
        # elif r:=re.match('^(.*)\^([0-9]+)([+-])$',self.name): # ddd^3+ / ddd^3-
            # self._charge = int(r.group(3)+r.group(2))
        # elif r:=re.match('^(.*[^+-])(\++|-+)$',self.name): # ddd+ / ddd++
            # self._charge = r.group(2).count('+') - r.group(2).count('-')
        # else:
            # self._charge = 0
        # return self._charge

    # charge = property(_get_charge)

    # def _get_elements(self):
        # if self._elements is not None:
            # pass
        # else:
            # self._elements = []
            # self._reduced_elements = {}
            # for part in re.split(r'(\[[0-9]*[A-Z][a-z]?\][0-9]*|[A-Z][a-z]?[0-9]*)',self.name):
                # if len(part)==0: continue
                # r = re.match('^(.+?)([0-9]*)$',part)
                # if r.group(2)=='':
                    # multiplicity = 1
                # else:
                    # multiplicity = int(r.group(2))
                # element = r.group(1)
                # element = element.replace(']','').replace('[','')
                # for i in range(multiplicity):
                    # self._elements.append(element)
                # if element in self.reduced_elements:
                    # self.reduced_elements[element] += multiplicity
                # else:
                    # self.reduced_elements[element] = multiplicity
        # return self._elements

    # elements = property(_get_elements)

    # def _get_element_species(self):
        # if self._element_species is None:
            # self._element_species = [Species(element) for element in self.elements]
        # return(self._element_species)
   #  
    # element_species = property(_get_element_species)
   #  
    # def _get_reduced_elements(self):
        # if self._reduced_elements is None:
            # self._get_elements()
        # return self._reduced_elements
   #  
    # reduced_elements = property(_get_reduced_elements)

    # def _get_nelectrons(self):
        # if self._nelectrons is not None:
            # pass
        # elif self.name == 'e-':
            # self.charge = -1
            # self._nelectrons = 1
        # elif self.name == 'photon':
            # self.charge = 0
            # self.nelectrons = 0
        # else:
            # self._nelectrons = 0
            # for element,multiplicity in self.reduced_elements.items():
                # element = re.sub(r'^\[?[0-9]*([A-Za-z]+)\]?',r'\1',element)
                # self._nelectrons += multiplicity*getattr(periodictable,element).number # add electrons attributable to each nucleus
            # self._nelectrons -= self.charge # account for ionisation
        # return self._nelectrons

    # nelectrons = property(_get_nelectrons)
           #  
    # def encode_elements(self,elements,charge=None,isomer=None):
        # ## convert list of elements into elemets with degeneracies
        # element_degeneracy = []
        # for element in elements:
            # if len(element_degeneracy)==0 or element_degeneracy[-1][0]!=element: # first or new
                # element_degeneracy.append([element,1])
            # elif element_degeneracy[-1][0]==element: # continuation
                # element_degeneracy[-1][1] += 1
        # ## concatenate
        # name = ''.join(
            # [(element if element[0] not in '0123456789' else '['+element+']') # bracket isotopes
             # +('' if degeneracy==1 else str(degeneracy)) # multiplicity
             # for element,degeneracy in element_degeneracy])
        # if charge is not None:
            # if charge>0:
                # for i in range(charge): name += '+'
            # if charge<0:
                # for i in range(-charge): name += '-'
        # if isomer is not None and isomer!='':
            # name = isomer+'-'+name
        # self.decode_name(name)
           #  
# ##    def decode_name(self,name,encoding=None):
# ##        """Decode a name and set all internal variables."""
# ##        ## translate diatomic special cases -- remove one day, e.g., 32S16O to [32]S[16]O
# ##        name = re.sub(r'^([0-9]+)([A-Z][a-z]?)([0-9]+)([A-Z][a-z]?)$',r'[\1\2][\3\4]',name)
# ##        name = re.sub(r'^([0-9]+)([A-Z][a-z]?)2$',r'[\1\2]2',name)
# ##        ## translate from some other encoding if requested
# ##        if encoding is not None:
# ##            name = translate_species_to_standard(encoding,name)
# ##        self.name = name
# ##        self.reduced_elements={}            # keys are names of elements, values are multiplicity in ONE molecule
# ##        self.elements=[]            # keys are names of elements, values are multiplicity in ONE molecule
# ##        self.charge=0
# ##        self.isomer=''
# ##        self.nelectrons=0
# ##        self.mass=np.nan            # of one molecule
# ##        ## short cut for electrons and photons
# ##        if name=='e-':
# ##            self.charge = -1
# ##            self.nelectrons = 1
# ##            self.mass = constants.m_e
# ##        elif name=='photon':
# ##            self.charge = 0
# ##            self.nelectrons = 0
# ##            self.mass = 0
# ##        else:
# ##            ## get charge
# ##            for t in [0]:
# ##                r = re.match('^(.*)\^([-+][0-9]+)$',name) # ddd^+3 / ddd^-3 / ddd^3
# ##                if r:
# ##                    self.charge = int(r.group(2))
# ##                    name = r.group(1)
# ##                    break
# ##                r = re.match('^(.*)\^([0-9]+)([+-])$',name) # ddd^3+ / ddd^3-
# ##                if r:
# ##                    self.charge = int(r.group(3)+r.group(2))
# ##                    name = r.group(1)
# ##                    break
# ##                r = re.match('^(.*[^+-])(\++|-+)$',name) # ddd+ / ddd++
# ##                if r:
# ##                    self.charge = r.group(2).count('+') - r.group(2).count('-')
# ##                    name = r.group(1)
# ##                    break
# ##            ## get isomer
# ##            r = re.match('^(.+)-(.*)$',name)
# ##            if r is not None:
# ##                self.isomer,name = r.groups()
# ##            ## get elements and their multiplicity
# ##            for part in re.split(r'(\[[0-9]*[A-Z][a-z]?\][0-9]*|[A-Z][a-z]?[0-9]*)',name):
# ##                if len(part)==0: continue
# ##                r = re.match('^(.+?)([0-9]*)$',part)
# ##                if r.group(2)=='':
# ##                    multiplicity=1
# ##                else:
# ##                    multiplicity = int(r.group(2))
# ##                element = r.group(1)
# ##                # element = element.replace(']','').replace('[','')
# ##                for i in range(multiplicity):
# ##                    self.elements.append(element)
# ##                if element in self.reduced_elements:
# ##                    self.reduced_elements[element] += multiplicity
# ##                else:
# ##                    self.reduced_elements[element] = multiplicity
# ##            ## try determine mass in amu and number of electrons --
# ##            ## MOVE THIS TO A PROPERTY?
# ##            self.mass = 0.
# ##            self.nelectrons = 0
# ##            for element,multiplicity in self.reduced_elements.items():
# ##                r = re.match(r'([0-9]+)([A-Za-z])+',element)
# ##                if r:
# ##                    isotope,element = r.groups()
# ##                    self.mass += multiplicity*getattr(periodictable,element)[int(isotope)].mass # isotopic mass
# ##                else:
# ##                    self.mass += multiplicity*getattr(periodictable,element).mass # natural mixture mass
# ##                self.nelectrons += multiplicity*getattr(periodictable,element).number # add electrons attributable to each nucleus
# ##            self.nelectrons -= self.charge # account for ionisation
       #  
    # def get_isotopologues(self,element_from,element_to):
        # """Find disctinct single-substitutions of one element."""
        # isotopologues = []      # list of Species, one for each distinct isotopologue
        # for i,element in enumerate(self.elements):
            # if i<(len(self.elements)-1) and element==self.elements[i+1]: continue # skip identical subsitutiotns, i.,e., keep rightmost, CO[18O] and not C[18O]O for CO2 substitution
            # if element==element_from:
                # t = copy(self.elements)
                # t[i] = element_to
                # isotopologues.append(Species(elements=t,charge=self.charge,isomer=self.isomer))
        # return(isotopologues)         

    # # def __str__(self):
    # #     return('\n'.join([
    # #         f'{"name":<10s} = {repr(self.name)}',
    # #         f'{"elements":<10s} = {repr(self.elements)}',
    # #         f'{"reduced_elements":<10s} = {repr(self.reduced_elements)}',
    # #         f'{"charge":<10s} = {repr(self.charge)}',
    # #         f'{"isomer":<10s} = {repr(self.isomer)}',
    # #         f'{"nelectrons":<10s} = {repr(self.nelectrons)}',
    # #         f'{"mass":<10s} = {repr(self.mass)}',
    # #         ]))

    # def __str__(self): return(self.name)

    # ## for sorting a list of Species objects
    # def __lt__(self,other): return(self.name<other.name)
    # def __gt__(self,other): return(self.name>other.name)
   #  

# class Mixture:
    # """A mixture of chemical species.  PERHAPS USABLE BY Reaction."""
   #  
    # def __init__(self,name='mixture'):
        # self.name = name
        # self.species = {}       # dictionary list {Species:amount}
        # self.other_data = {}

    # def __getitem__(self,key):
        # if key in self.other_data:
            # return(self.other_data[key])
        # elif key in self.species:
            # return(self.species[key])
        # elif (t:=get_species(key)) in self.species:
            # return(self.species[t])

    # def __setitem__(self,key,val):
        # if key in self.species:
            # self.species[key] = val
        # elif key in self.other_data:
            # self.other_data[key] = val
        # else:
            # self.add_species(key,val)

    # def add_data(self,key,val):
        # """Add some kind of data."""
        # self.other_data[key] = val

    # def add_species(self,name_or_Species,amount):
        # """Add a species to the mixture."""
        # if isinstance(name_or_Species,str):
            # name_or_Species = get_species(name_or_Species)
        # name,species = name_or_Species.name,name_or_Species
        # ## if already in mixture add amount, else add new element to
        # ## self.species
        # for tspecies in self.species:
            # if tspecies.name==name:
                # break
        # else:
            # self.species[species] = amount
        # return(species)

    # def get_elements(self):
        # elements = {}
        # for species,amount in self.species.items():
            # for element,multiplicity in species.reduced_elements.items():
                # if element in elements:
                    # elements[element] += multiplicity*amount
                # else:
                    # elements[element] = multiplicity*amount
        # return(elements)

    # def __str__(self):
        # return('\n'.join(
            # [f'{str(species):>10s} = {amount}'
            # for species,amount in self.species.items()]))

    # def get_atom_number(self):
        # elements = self.get_elements()
        # total_atom_number = sum(elements.values())
        # return(total_atom_number)

# class Reaction:

    # def __init__(
            # self,
            # name=None,    # correctly encoded
            # reaction_type='constant', # type of reaction, defined in get_rate_coefficient
            # encoding=None,            # of name
            # reactants_products=None, # ([list of reactant Species],[list of product Species])
            # reference=None,
            # **c,     # used for computing rate coefficient
    # ):
        # self.name = name
        # self.reactants,self.products = [],[]
        # self.reaction_type = reaction_type
        # self.c = c # may be scalar or multidimensional
        # self.rate_coefficient = None        # not yet computed
        # if name is not None:
            # self.decode_name(name,encoding)
        # elif reactants_products is not None:
            # self.encode_reactants_products(*reactants_products)

    # def encode_reactants_products(self,reactants,products):
        # self.reactants = reactants
        # self.products = products
        # self.name = ' + '.join([t.name for t in reactants])+' ⟶ '+' + '.join([t.name for t in products])
       #  
    # def get_hash(self):
        # """A convenient way to summarise a reaction by its name. I don't use
        # __hash__ because I worry that the reactants/products might mutate."""
        # return(hash(' + '.join(sorted([t.name for t in self.reactants]))+' ⟶ '+' + '.join(sorted([t.name for t in self.products]))))
       #  
    # def decode_name(self,name,encoding=None):
        # """Decode a string indicating a reaction, e.g, H + H ⟶ H2. Returns two
        # lists (reactants,products)."""
        # if encoding is not None:
            # name = translate_reaction_to_standard(encoding,name)
        # reactants,products = [],[]
        # for reactants_or_products,terms in zip([reactants,products],name.split('⟶')):
            # for term in re.split(r' +\+ +',terms.strip()):
                # ## get initial multiplier
                # r = re.match('^([0-9]+)(.*)$',term)
                # if r is not None:
                    # multiplicity,term = int(r.group(1)),r.group(2)
                # else:
                    # multiplicity = 1
                # species = Species(term)
                # for t in range(multiplicity):
                    # reactants_or_products.append(species)
        # self.reactants = reactants
        # self.products = products
       #  
    # def get_isotopologues(self,element_from,element_to,reduced=False):
        # """Find disctinct single-substitutions of one element balancing reaction."""
        # reactions = []
        # for ireactant,reactant in enumerate(self.reactants):
            # for reactant_isotopologue in reactant.get_isotopologues(element_from,element_to):
                # for iproduct,product in enumerate(self.products):
                    # for product_isotopologue in product.get_isotopologues(element_from,element_to):
                        # reactants = copy(self.reactants)
                        # reactants[ireactant] = reactant_isotopologue
                        # products = copy(self.products)
                        # products[iproduct] = product_isotopologue
                        # reactions.append(Reaction(reactants_products=(reactants,products)))
        # ## Currently a list of potentially non-unique products and
        # ## reactant subsitutions (given reorderings). If reduced=True
        # ## then combine those reorderings and reutrn with a degeneracy
        # ## factor.
        # if not reduced: return(reactions)
        # d = {}
        # for r in reactions:
            # if r.get_hash() not in d:
                # d[r.get_hash()] = [r,1]
            # else:
                # d[r.get_hash()][1] += 1
        # reactions_degeneracies = d.values()
        # return(reactions_degeneracies)

    # def __getitem__(self,key):
        # return(self.c[key])

    # def __setitem__(self,key,val):
        # self.c[key] = val


    # def get_rate_coefficient(self,**p):
        # """Calculate rate coefficient from parameters p."""
        # if self.reaction_type=='constant':
            # self.rate_coefficient = self['k']
        # elif self.reaction_type=='function': # k==a function with arguments in p
            # self.rate_coefficient = self['f'](**p)
        # elif self.reaction_type=='Arrhenius': # a la KIDA
            # self.rate_coefficient = self['A']*(p['T']/300.)**self['B']
        # elif self.reaction_type=='modified Arrhenius': # a la KIDA
            # self.rate_coefficient = self['A']*(p['T']/300.)**self['B']*np.exp(-self['C']*p['T'])
        # elif self.reaction_type=='NIST':
            # self.rate_coefficient = self['A']*(p['T']/298.)**self['n']*np.exp(-self['Ea']/8.314472e-3/p['T'])
        # elif self.reaction_type=='NIST_3rd_body_hack':
            # self.rate_coefficient = 1e19*self['A']*(p['T']/298.)**self['n']*np.exp(-self['Ea']/8.314472e-3/p['T'])
        # elif self.reaction_type=='photoreaction':
            # import scipy
            # self.rate_coefficient = scipy.integrate.trapz(self['σ'](p['T'])*p['I'],self['λ'])
        # else:
            # raise Exception(f"Unknown reaction_type: {repr(self.reaction_type)}")
        # return(self.rate_coefficient)

    # def format_aligned_reaction_string(
            # number_of_reactants=3, # how many columns of reactants to assume
            # number_of_products=3, # products
            # species_name_length=12, # character length of names
            # ):
        # ## get reactants and products padded with blanks if required
        # reactants = copy(self.reactants)+['' for i in range(number_of_reactants-len(reactants))]
        # products = copy(self.products)+['' for i in range(number_of_products-len(products))]
        # return((' + '.join([format(t,f'<{species_name_length}s') for t in reactants])
                # +' ⟶ '+' + '.join([format(t,f'<{species_name_length}s') for t in reactants])))

    # def __str__(self):
        # return(format(self.name,'80')+' :: '+' '.join(
            # [format(key+':'+str(val),'12')
             # for key,val in self.c.items()]))

    # # def generate_all_single_isotopic_substitutions(self,subsitute_from,subsitute_to):
        # # subsitute_from = Species(subsitute_from)
        # # subsitute_to = Species(subsitute_to)
        # # print( subsitute_from,subsitute_to)
        # # for reactant in self.reactants:
            # # for element in reactant.elements:
                # # if element==subsitute_from.name:
                    # # for product in self.products:
                        # # for element,multiplicity in product.elements.items():
                            # # if element==subsitute_from.name:
                                # # print( )
                                # # print( reactant)
                                # # print( product)
                                # # print( element,multiplicity)


# class Reaction_Network:

    # def __init__(self,kida_filename=None):
        # self.reactions = []
        # self.T = 300            # constant temperature or a function of time
        # self.species = {}       # indexd by species name
        # self.n = dict()         # indexd by species name
        # self.rate = dict()      # indexd by species name
        # if kida_filename is not None:
            # self.load_from_kida(kida_filename)

    # def print_rate_coefficients(self):
        # for reaction in self.reactions:
            # print( format(reaction.name,'40'),reaction.get_rate_coefficient(T=self.T))
       #  
    # def get_rates(self,time,density):
        # """Calculate all rate coefficients in the network at a given time and
        # with given densities."""
        # rates = np.zeros(len(density),dtype=float) # rates for all species
        # self.reaction_rates = np.zeros(len(self.reactions),dtype=float)
        # self.rate_coefficients = np.zeros(len(self.reactions),dtype=float)
        # ## T is a constant or function fo time
        # if np.isscalar(self.T):
            # T = self.T
        # else:
            # T = self.T(time)
        # for ireaction,reaction in enumerate(self.reactions):
            # ## compute rates of reaction
            # rate_coefficient = reaction.get_rate_coefficient(T=T)
            # reaction_rate = (rate_coefficient*np.prod([
                                # density[self._species_index[reactant.name]]
                                 # for reactant in reaction.reactants]))
            # self.reaction_rates[ireaction] = reaction_rate
            # self.rate_coefficients[ireaction] = rate_coefficient
            # ## add contribution to product/reactant
            # ## formation/destruction rates
            # for reactant in reaction.reactants:
                # rates[self._species_index[reactant.name]] -= reaction_rate
            # for product in reaction.products:
                # rates[self._species_index[product.name]] += reaction_rate
        # self.rates = rates
        # return(rates)

    # def integrate(self,time,nsave_points=10,**initial_densities):
        # ## collect all species names
        # species = set()
        # for reaction in self.reactions:
            # for reactant in reaction.reactants:
                # species.add(reactant.name)
            # for product in reaction.products:
                # species.add(product.name)
        # for name in initial_densities:
            # species.add(name)
        # ## species_name:index_of_species_in_internal_arrays
        # self._species_index = {name:i for (i,name) in enumerate(species)} 
        # ## time steps
        # time = np.array(time,dtype=float)
        # if time[0]!=0:
            # time = np.concatenate(([0],time))
        # density = np.full((len(species),len(time)),0.)
        # ## set initial conditions
        # for key,val in initial_densities.items():
            # density[self._species_index[key],0] = val
        # ## initialise integrator
        # r = integrate.ode(self.get_rates)
        # r.set_integrator('lsoda', with_jacobian=True)
        # r.set_initial_value(density[:,0])
        # ## run saving data at requested number of times
        # for itime,timei in enumerate(time[1:]):
            # r.integrate(timei)
            # density[:,itime+1] = r.y
        # ##
        # retval = Dynamic_Recarray(
            # time=time,
            # T=self.T(time),
        # )
        # for speciesi,densityi in zip(species,density):
            # retval[speciesi] = densityi
        # self.species = species
        # self.density = density
        # self.time = time
        # self.rates = self.get_rates(self.time[-1],self.density[:,-1])
        # return(retval)
   #  
    # def print_rates(self):
        # for reaction,rate,rate_coefficient in zip(self.reactions,self.reaction_rates,self.rate_coefficients):
            # print( format(reaction.name,'35'),format(rate_coefficient,'<+10.2e'),format(rate,'<+10.2e'))
   #  
    # def append(self,reaction):
        # self.reactions.append(reaction)
       #  

    # def extend(self,reactions):
        # self.reactions.extend(reactions)

    # ## a list of unique reactants, useful where these branch
    # unique_reactants = property(lambda self:set([t.reactants for t in self.reactions]))

    # def get_species(self):
        # """Return a list of all species in this network."""
        # return(list(np.unique(np.concatenate([list(t.products)+list(t.reactants) for t in self.reactions]))))
   #  
    # def get_product_branches(self,reactants,with_reactants=[],without_reactants=[],with_products=[],without_products=[]):
        # """Get a list of reactions with different products and the
        # same reactants. Restricut to some products"""
        # return([t for t in self.reactions
                # if t.reactants==reactants
                # and np.all([t1 in t.products  for t1 in with_products])
                # and not np.any([t1 in t.products for t1 in without_products])
                # and np.all([t1 in t.reactants for t1 in with_reactants])
                # and not np.any([t1 not in t.reactants for t1 in without_reactants])])

    # def __iter__(self):
        # for t in self.reactions: yield(t)

    # def __len__(self): return(len(self.reactions))
           #  
    # def get_matching_reactions(self,reactants=(),products=(),not_reactants=(),not_products=()):
        # """Return a list of reactions with this reactant."""
        # return([t for t in self.reactions if
                # np.all(
                    # [t0 in t.reactants for t0 in reactants]
                    # +[t0 in t.products for t0 in products]
                    # +[t0 not in t.reactants for t0 in not_reactants]
                    # +[t0 not in t.products for t0 in not_products]
                # )])

    # def get_reaction(self,name):
        # for t in self.reactions:
            # if t.name==name:
                # return(t)
        # else:
            # raise IndexError('Could not find reaction: '+repr(name))    

    # def __str__(self):
        # retval = []
        # for name,species in self.species.items():
            # retval.append(f'{name:20} {species.density}')
        # for t in self.reactions:
            # retval.append(str(t))
        # return('\n'.join(retval))
#  
    # # def get_rates(self):
        # # self.rates = dict()
        # # for reaction in self.reactions:
            # # self.rates[reaction] = reaction.get_rate(T=self.T,n=self.n)
            # # for multiplicity,species in reaction.reactants:
                # # self.rate[species] -= multiplicity*self.rates[reaction]
            # # for multiplicity,species in reaction.products:
                # # self.rate[species] += multiplicity*self.rates[reaction]
               #  
    # def load_from_kida(self,filename):
        # """E.g, ~/data/reactions/kida/kida.uva.2014/kida.uva.2014.dat"""
        # if filename[-4:]=='.csv':
            # for data in tools.file_to_recarray(filename): # loop through reactions
                # if len(data['Reactant 1'])==0: continue   # indicates blank line
                # if data['Reactant 1']=='nan': continue   # indicates blank line
                # reaction_type = 'KIDA modified Arrhenius' # default
                # if data['Reactant 2']=='CR': reaction_type = 'KIDA primary cosmic ray'
                # elif data['Reactant 2']=='CRP': reaction_type = 'KIDA secondary cosmic ray'
                # elif data['Reactant 2']=='Photon': reaction_type = 'KIDA photo process'
                # coefficients = collections.OrderedDict()
                # coefficients['α'] = float(data['Alpha'])
                # coefficients['β'] = float(data['Beta'])
                # coefficients['γ'] = float(data['Gamma'])
                # for key in ('Uncert', 'g', 'Uncert Type', 'Channel type', 'Formula',
                            # 'Tmin', 'Tmax', 'Field (for UV)', 'Method', 'Origin',
                            # 'Database', 'BibTex file', 'Description', 'Application'):
                    # coefficients[key] = data[key]
                # self.reactions.append(
                    # Reaction(reactants=[data[key].strip() for key in ['Reactant 1','Reactant 2'] if str(data[key]).strip() not in ['','nan']],
                             # products=[data[key].strip() for key in ['Product 1','Product 2','Product 3','Product 4','Product 5','Product 6'] if str(data[key]).strip() not in ['','nan']],
                             # reaction_type=reaction_type, coefficients=coefficients))
        # ## space formatted file, e..g, ~/data/reactions/kida/kida.uva.2014/kida.uva.2014.dat
        # else:
            # with open(tools.expand_path(filename),'r') as fid:
                # for line in fid:
                    # if line[0]=='!':continue # skip first line
                    # reaction_type = 'KIDA modified Arrhenius'
                    # ## get species
                    # reactants, products = [],[]
                    # for t in line[0:34].split():
                        # if t=='CR': reaction_type = 'KIDA primary cosmic ray'
                        # elif t=='CRP': reaction_type = 'KIDA secondary cosmic ray'
                        # elif t=='Photon': reaction_type = 'KIDA photo process'
                        # reactants.append(translate_species_kida_to_standard(t))
                    # for t in line[34:91].split():
                        # products.append(translate_species_kida_to_standard(t))
                    # ## get coefficients
                    # A,B,C = line[91:123].split()
                    # coefficients = collections.OrderedDict()
                    # coefficients['α'] = float(A)
                    # coefficients['β'] = float(B)
                    # coefficients['γ'] = float(C) 
                    # self.reactions.append(Reaction(reactants,products,reaction_type,coefficients))

    # def get_recarray(self,keys=None):
        # data = collections.OrderedDict()
        # ## list reactants up to max number of reactants
        # for i in range(np.max([len(t.reactants) for t in self.reactions])):
            # data['reactant'+str(i)] = [t.reactants[i] if len(t.reactants)>i else ''for t in self.reactions]
        # ## list products up to max number of products
        # for i in range(np.max([len(t.products) for t in self.reactions])):
            # data['product'+str(i)] = [t.products[i] if len(t.products)>i else '' for t in self.reactions]
        # ## all coefficients
        # if keys is None:
            # keys = np.unique(np.concatenate(
                # [list(t.coefficients.keys()) for t in self.reactions]))
        # for key in keys:
            # data[key] = [t[key] if key in t.coefficients else np.nan for t in self.reactions]
        # return(tools.dict_to_recarray(data))
                   #  
    # def save_to_kida_csv(self,filename):
        # """Save to KIDA csv format."""
        # lines = [['Reactant 1', 'Reactant 2', 'Product 1', 'Product 2', 'Product 3',
                 # 'Product 4', 'Product 5', 'Product 6', 'Alpha ', 'Beta', 'Gamma',
                 # 'Uncert', 'g', 'Uncert Type', 'Channel type', 'Formula', 'Tmin',
                 # 'Tmax', 'Field (for UV)', 'Method ', 'Origin ', 'Database',
                  # 'BibTex file ', 'Description', 'Application']]
        # def format_strings(x):
            # if t['Field (for UV)']=='nan' or (tools.isnumeric(x) and np.isnan(x)):
                # return('""')
            # return('"'+str(x)+'"')
        # for t in self.reactions:
            # lines.append([
                # format(t.reactants[0] if len(t.reactants)>0 else '','>8'),
                # format(t.reactants[1] if len(t.reactants)>1 else '','>8'),
                # format(t.products[0] if len(t.products)>0 else '','>8'),
                # format(t.products[1] if len(t.products)>1 else '','>8'),
                # format(t.products[2] if len(t.products)>2 else '','>2'),
                # format(t.products[3] if len(t.products)>3 else '','>2'),
                # format(t.products[4] if len(t.products)>4 else '','>2'),
                # format(t.products[5] if len(t.products)>5 else '','>2'),
                # format(t['α'],'10.2e'),
                # format(t['β'],'5g'),
                # format(t['γ'],'5g'),
                # format(t['Uncert']),
                # format(t['g']),
                # format(t['Uncert Type']),
                # format(t['Channel type']),
                # format(t['Formula']),
                # format(t['Tmin']),
                # format(t['Tmax']),
                # format_strings(t['Field (for UV)']),
                # format_strings(t['Method']),
                # format_strings(t['Origin']),
                # format_strings(t['Database']),
                # format_strings(t['BibTex file']),
                # format_strings(t['Description']),
                # format_strings(t['Application']),
            # ])
        # lines = [' , '.join([format(t,'>10') for t in line]) for line in lines]
        # tools.string_to_file(filename,'\n'.join(lines))    
       #  


