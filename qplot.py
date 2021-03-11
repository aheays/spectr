#!/usr/bin/env python3 

import argparse,warnings,os,signal,sys,collections,re
from spectr import tools,plotting
from spectr.conversions import convert
import numpy as np
import matplotlib as mpl
from copy import deepcopy

## get command line agruments
parser = argparse.ArgumentParser(description='Quickly plot data in files.')
parser.add_argument('-s','--skip-header', dest='skip_header',type=int,default=0, help='How many rows at beginning of file to skip.')
parser.add_argument('-r','--read-names', dest='read_names',action="store_true",default=False, help='Extract names from first nonskipped row.')
parser.add_argument('-x','--xkey', dest='xkey',type=str,default=None, help='Data label to use for x coordinate.')
parser.add_argument("-y","--ykey",type=str,dest="ykeys",nargs=1,action='append', default=[],help="Add data label to plot on y axis.")
parser.add_argument('--xcolumn', dest='xindex',type=int,default=0, help='Column index to use as x-axis, either a column index or column name.')
parser.add_argument("--ycolumn",type=int,dest="yindices",nargs=1,action='append', default=[],help="Add Column index to plot on y axis.")
parser.add_argument('-n','--names', dest='names',type=str,default=None, help='List names in comma separated list.')
parser.add_argument('-o','--output-to-file', dest='output_to_file',type=str,default=None, help='Filename to output to, alternative on screen.')
parser.add_argument('-t','--filetype', dest='filetype',action="store", default=None,help='What type of file this is.')
parser.add_argument('-f', dest='fastplot',default='False',action='store_true',help='Use fastplot.')
parser.add_argument('-F', '--fullscreen',dest='fullscreen',default='False',action='store_true',help='Plot full screen.')
parser.add_argument('-v','--verbose-mode', dest='verbose',action="store_true", help='Increased error messages.')
parser.add_argument('-a','--awkscript', dest='awkscript',type=str,default=None, help='Pipe file through an awkscript first.')
parser.add_argument('-l','--linewidth', dest='linewidth',default=1,type=float,help='Set linewidth.')
parser.add_argument('-D','--down-sample', dest='down_sample',default=None,type=int,help='Integer for simple down sampling before plotting.')
parser.add_argument('-m','--marker', dest='marker',default='',type=str,help='Marker type.')
parser.add_argument('--encoding', dest='encoding',default=None,type=str,help='File encoding.')
parser.add_argument('-O','--offset-plots', dest='offset_plots',default=0.,type=float,help='Offset each successive plot by this much.')
parser.add_argument('--legend', dest='legend_on',action="store_true",default=True,help='Legend.')
parser.add_argument('--no-legend', dest='legend_on',action="store_false",default=True,help='No legend.')
parser.add_argument('-A','--annotate-lines',dest='annotate_lines',action="store_true",default=False,help='Annotate lines.')
parser.add_argument('--ylog', dest='ylog',action="store_true",default=False, help='Use y-axis log scale.')
parser.add_argument('--xlog', dest='xlog',action="store_true",default=False, help='Use x-axis log scale.')
parser.add_argument('--grid', dest='grid',action="store_false",default=True, help='Do not plot grid.')
parser.add_argument('--delimiter', dest='delimiter',type=str,default=None, help='Column delimiter in data file.')
parser.add_argument('--xbeg', dest='xbeg',type=float,default=None, help='Lower bound of x-axis scale.')
parser.add_argument('--xend', dest='xend',type=float,default=None, help='Lower bound of x-axis scale.')
parser.add_argument('--ybeg', dest='ybeg',type=float,default=None, help='Lower bound of y-axis scale.')
parser.add_argument('--yend', dest='yend',type=float,default=None, help='Lower bound of y-axis scale.')
parser.add_argument('--xconvert', dest='xconvert',type=str,nargs=2,default=None, help='Convert x units (from,to).')
parser.add_argument('--yconvert', dest='yconvert',type=str,nargs=2,default=None, help='Convert y units (from,to).')
parser.add_argument('--xaltaxis', dest='xaltaxis',type=str,default=None, help='Display an alternative x axis with this transform of units. TRANSFORM BREAKS ON ZOOM.')
parser.add_argument('--linestyle', dest='linestyle',type=str,default=None, help='Use this linestyle, else adjust for uniqueness.')
parser.add_argument('--color', dest='color',type=str,default=None, help='Use this color, else adjust for uniqueness.')
parser.add_argument('--contaminants', dest='contaminants_to_plot',type=str,default=None, help='Plot contaminant spectral lines, comma separated list of species, or "default". Requires cm-1 units.')
parser.add_argument('--labels-commented', dest='labels_commented',default="True",action="store_true",help='Label line is commented.')
parser.add_argument('--labels-uncommented', dest='labels_commented',default="True",action="store_false",help='Label line is not commented.')
parser.add_argument('--hitran', dest='hitran',default=False,action="store_true",help='Plot HITRAN spetrum for this species')
parser.add_argument('filenames', metavar='file', type=str, nargs='*',help='name of a band data file')
args = parser.parse_args()

## reduce spew to console
if not args.verbose:
    np.seterr(divide='ignore')
    np.seterr(invalid='ignore')
    sys.tracebacklimit = 0   # will supress traceback in python 3, doesn't work 2.7
    def signal_handler(sig,trace):
        print()
        sys.exit(1)
    signal.signal(signal.SIGINT,signal_handler)
 
## default to stdinput 
if len(args.filenames)==0:
    args.filenames = ['/dev/stdin']

## convert filenames to hitran paths
if args.hitran:
    args.filenames = [f'~/data/databases/HITRAN/data/{species}/natural_abundance/absorption_cross_section_296K_1atm.h5' for species in args.filenames]

## list of strings, not list of lists
args.ykeys = [t[0] for t in args.ykeys]

## set plot_library
if args.fastplot is True:
    args.plot_library = 'fastplot'
else:
    args.plot_library = 'matplotlib'
    
## plotting header
if args.plot_library=='fastplot':
    import fastplot as plt
    fig = plt.figure()
    ax = plt.gca()
else:
    import matplotlib.pyplot as plt
    if args.output_to_file == None:
        plotting.presetRcParams('screen')
        fig = plotting.qfig(fullscreen=args.fullscreen,)
    else:
        plotting.presetRcParams('a4landscape')
        fig = plt.figure()
    ax = fig.gca()

## loop through each file, plotting as best one can
plot_count = 0
original_args = deepcopy(args)  # cache args because they can be changed below, and each file should begin the same
for f in args.filenames:
    args = deepcopy(original_args)

    ## include filename in label if multiple files plotted
    label_prefix = ''
    if len(args.filenames)>1:
        label_prefix = f+' -- '

    ## if only one file use title
    if len(args.filenames)==1:
        if args.plot_library=='matplotlib':
            t = ax.set_title(f)
            t.set_in_layout(False)



    ## load data into a dictionary of arrays
    if args.xkey is not None or (len(f)>3 and f[-3:]=='.h5'):
        ## load as dictionary
        data = tools.file_to_dict(f,labels_commented=args.labels_commented,filetype=args.filetype)
        ## special cases one data series only 'data' which must be a
        ## 1- or 2-dimensional array then load columns a 'column0' etc
        if len(list(data.keys()))==1 and 'data' in data:
            data = {f'column{t0}':t1 for (t0,t1) in enumerate(data['data'].transpose())}
        
    else:
        ## load as array, convert to dictionary with index labels
        file_to_array_kwargs = dict(unpack=True, awkscript=args.awkscript, skip_header=args.skip_header)
        if args.delimiter is not None:
            file_to_array_kwargs['delimiter'] = args.delimiter
        t = tools.file_to_array(f,**file_to_array_kwargs)
        if t.ndim==0:
            data = {'column0':np.array(float(t))}
        elif t.ndim==1:
            data = {'column0':t}
        else:
            data = collections.OrderedDict()
            for (t0,t1) in enumerate(t):
                data['column'+str(t0)] = t1


    ## get a deafult xkey
    if args.xkey is None:
        if len(data)==1:
            assert 'index' not in data
            args.xkey = 'index'
            data['index'] = np.arange(len(list(data.values())[0])) 
        else:
            args.xkey = list(data.keys())[0]

    ## get default ykeys
    if len(args.ykeys) == 0:
        xlen = len(data[args.xkey])
        args.ykeys = [key for key in list(data.keys()) if key!=args.xkey and len(data[key])==xlen]

    ## unit conversions
    if args.xconvert is not None:
        data[args.xkey] = convert(data[args.xkey],args.xconvert[0],args.xconvert[1])
    if args.yconvert is not None:
        for key in args.ykeys:
            data[key] = convert(data[key],args.yconvert[0],args.yconvert[1])

    ## loop through all ykeys and plot
    for ykey in args.ykeys:
        color,linestyle = plotting.newcolorlinestyle()
        if args.linestyle is not None:
            linestyle = args.linestyle
        if args.color is not None:
            color = args.color
        ## data to plot
        x,y =  data[args.xkey],data[ykey]
        ## shift by offset if requested
        if args.offset_plots!=0:
            y = y+plot_count*args.offset_plots
        ## down sample
        if args.down_sample is not None:
            x,y = x[::args.down_sample],y[::args.down_sample]
        ## the actual plot command
        plot_count += 1
        if args.plot_library=='matplotlib':
            ax.plot(x,y,label=label_prefix+ykey,color=color,linestyle=linestyle,linewidth=args.linewidth,marker=args.marker)
        elif args.plot_library=='fastplot':
            plt.plot(x,y,label=label_prefix+ykey,color=color,)

## plot contaminants spectral lines if requested
if args.contaminants_to_plot is not None:
    from spectra import database
    contaminant_linelist = database.get_spectral_contaminant_linelist(
        *args.contaminants_to_plot.split(','),
        νbeg=ax.get_xlim()[0],  # BAD TIME TO GET XLIM?
        νend=ax.get_xlim()[1],) # BAD TIME TO GET XLIM?
    for line in contaminant_linelist:
        x,y = line['ν'],ax.get_ylim()[0]/2.
        plotting.annotate_vline(line['name'],line['ν'],color='gray',fontsize='x-small',zorder=-5)
            # ax.plot(x,y,ls='',marker='o',color='red',markersize=6)
        # ax.annotate(line['name'],(x,1.1*y),ha='center',va='top',color='gray',fontsize='x-small',rotation=90,zorder=-5)

            
if args.plot_library=='matplotlib':

    if args.xbeg is not None:  ax.set_xlim(xmin=args.xbeg)
    if args.xend is not None:  ax.set_xlim(xmax=args.xend)
    if args.ybeg is not None:  ax.set_ylim(ymin=args.ybeg)
    if args.yend is not None:  ax.set_ylim(ymax=args.yend)
    if args.ylog: ax.set_yscale('log')
    if args.xlog: ax.set_xscale('log')
    if args.grid: ax.grid(True,color='gray')
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%0.10g')) # reliable x,y braph
    if args.annotate_lines:
        for t in plotting.annotate_line(ax=ax,fontsize='x-small',xpos='right',ypos='bottom'):
            t.set_in_layout(False)
        args.legend_on = False
    plotting.extra_interaction(fig=fig) # make lines pickable etc
    ## add alternative axis ticks with different units if requested -- should come afer all other plotting commands
    if args.xaltaxis is not None:
        plotting.add_xaxis_alternative_units(ax=ax,transform=getattr(my,args.xaltaxis),fmt='0.0f')
    ## print or display to screen
    
ax.set_xlabel(args.xkey)
if args.legend_on:
    plotting.legend(loc='upper right')

if args.output_to_file==None:
    # ## open fig maximised or full screen
    # mng = plt.get_current_fig_manager()
    # mng.window.showMaximized()
    # # mng.full_screen_toggle()
    # plt.ioff()
    plotting.show()
else:
    plotting.savefig(args.output_to_file)

