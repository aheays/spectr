
# Table of Contents

1.  [Installation](#org5a4dfc1)
    1.  [Source code](#org50036bc)
    2.  [Python dependencies](#org523172d)
    3.  [Building the fortran extensions](#org6fabff1)
    4.  [Matplotlib dependencies](#orgbfaf1b3)
    5.  [Installing in a linux virtual environment](#orgd4140fb)
    6.  [Testing the installation](#org3298cf3)
2.  [Usage](#org147597b)
    1.  [Importing `spectr`](#org464376e)
    2.  [Optimising things](#org9b30690)
    3.  [Encoding linear molecule quantum numbers](#orga1eb910)
    4.  [`qplot`](#org1895833)
3.  [Examples](#org2110ef9)
4.  [Submodules](#orgbb75b11)
    1.  [`env.py`](#org89f5cc3)
    2.  [`dataset.py`](#org56f7c45)
    3.  [`tools.py`](#orgaaa27c7)
    4.  [`plotting.py`](#org7b92b0d)
    5.  [`convert.py`](#org05a186c)
    6.  [`optimise.py`](#org60446f5)
    7.  [`atmosphere.py`](#orgd9bcc3d)
    8.  [`lines.py`](#org20edc6c)
    9.  [`levels.py`](#org1223f31)
    10. [`bruker.py`](#orgd636383)
    11. [`database.py`](#org1a7110f)
    12. [`electronic_states.py`](#orgdc1843f)
    13. [`exceptions.py`](#orgf9e2f8a)
    14. [`hitran.py`](#org4bd04b2)
    15. [`lineshapes.py`](#org79dedd5)
    16. [`quantum_numbers.py`](#orga5fce86)
    17. [`spectrum.py`](#org750f7ad)
    18. [`thermochemistry.py`](#org0b5c98f)
    19. [`viblevel.py`](#org5f368e1)
    20. [`fortran_tools.f90`](#orgbaa288a)
5.  [Bugs / improvements](#org12464c8)
    1.  [optimise.py](#org28258b0)
    2.  [viblevel.py](#orgbdaebe1)



<a id="org5a4dfc1"></a>

# Installation


<a id="org50036bc"></a>

## Source code

GPL source code is publically available on github at  <https://github.com/aheays/spectr>.
The repository can be cloned locally with `git clone --depth=1 https://github.com/aheays/spectr.git` (this will require git to be installed `sudo apt install git` on ubuntu). A subsequent `git pull` while inside the `spectr` source directory will download any updates. Otherwise, zip-archives of particular version can be downloaded e.g., <https://github.com/aheays/spectr/archive/refs/tags/v1.3.0.zip>.


<a id="org523172d"></a>

## Python dependencies

This module has been tested and works with python >=3.9. Many non-standard python libraries are used and are available from linux distributions or via pip, with the following package names:

    bidict cycler hitran-api brukeropusreader dill h5py matplotlib
    numpy openpyxl periodictable scipy sympy xmltodict pyqt5 py3nj

Note: py3nj can be successfully installed only if gfortran is already installed


<a id="org6fabff1"></a>

## Building the fortran extensions

Fortran code used to generate python extensions with `f2py` and needed to speed up some calculations.  Some submodules will still import and run fine without these.
To compile the extensions run `make` within the spectr source directory. The options in the `Makefile` are for the gfortran compiler and use the `lapack` library.  The following distribution packages might be sufficient to compile the code under linux:

-   Debian / Ubuntu: `gfortran python3-dev liblapack-dev`
-   Arch: `gcc-gfortran lapack`

If the Makefile is not working, then the following should be enough to get things running:

    f2py3 -c --quiet -llapack --f90flags="-Wall -ffree-line-length-none" --opt="-O3" fortran_tools.f90


<a id="orgbfaf1b3"></a>

## Matplotlib dependencies

Under ubuntu, installing the packages `pyqt5` and `qt5-default` seem to be enough to get matplotlib to draw windowed figures.


<a id="orgd4140fb"></a>

## Installing in a linux virtual environment

This is a recipe for installing spectr and its dependencies in a virtual environment, and not messing up the system python.  The script `install_in_venv.sh` attempts to do everything necessary, requiring only python3, gfortran, and lapack to be installed: `bash install_in_venv.sh`.

Individual steps:

-   Install python3 (or >=3.9 required) somehow. Under ubuntu with `sudo apt install python3`.
-   Create a virtual environment using python3: `python3 -m venv venv3`
-   Start the virtual environment: `source venv3/bin/activate`
-   Install the python dependencies with pip: `pip install bidict cycler hitran-api brukeropusreader dill h5py matplotlib numpy openpyxl periodictable scipy sympy xmltodict ; pip install py3nj` *For some reason py3nj must be intalled after everything else (2022-01-18).*
-   The matplotlib Qt dependency can also be installed with pip, `pip install pyqt5`, but Qt must be installed on the system itself, e.g, with `sudo apt install qt5-default`
-   Clone spectr code with `git clone --depth=1 https://github.com/aheays/spectr.git`
-   Add `spectr` to the python path so it can be imported from anywhere
    
        cd venv3/lib/python3/site-packages/
        ln -s ../../../../spectr .
        cd -
-   Add `qplot` to the path so it can be run from within the virtual environment
    
        cd venv3/bin
        ln -s ../../spectr/qplot .
        cd -


<a id="org3298cf3"></a>

## Testing the installation

Test by importing spectr and trying to plot something

    source venv3/bin/activate
    echo "from spectr.env import *" | python
    qplot absorption/data/2021_11_30_bcgr.0


<a id="org147597b"></a>

# Usage


<a id="org464376e"></a>

## Importing `spectr`

A command to import all submodules and many common functions directly into the working namespace is `from spectr.env import *`.  Otherwise only the needed submodules can be imported, e.g., `import spectr.spectrum`


<a id="org9b30690"></a>

## Optimising things

The `optimiser.Optimiser` class is used to conveniently construct model objects with parameters that can be fit to experimental data. The real-number input arguments of most methods of objects base-classed on `Optimiser` can be marked for optimisation by replacing their values with a optimiser.Parameter object.  This has the abbreviated definition:

    P(value=float,
      vary=True|False,
      step=float,
      uncertainty=float,
      bounds=(float,float))

Only the first argument is required. For example, `x=P(2,True,1e-5,bounds=(0,100))` defines a parameter `x` that will be varied from an initial value of 2 but constrained to the range 0 to 100.  When computing the finite-difference approximation to the linear dependence of model error on `x` a step size of \num{e-5} will be used.  The fitting uncertainty `unc` will be set automatically after optimisation.
Multiple `Optimiser` objects can be combined in a hierarchy, so that multiple spectra can be fit at once to optimise a common parameter, for example a temperature-dependence coefficient fit to spectra at multiple temperatures.


<a id="orga1eb910"></a>

## Encoding linear molecule quantum numbers

TBD


<a id="org1895833"></a>

## `qplot`

<a id="org00e4ee0"></a>
This is a command line programming for making line plots, e.g., `qplot datafile`, or `qplot -h` for a list of options.


<a id="org2110ef9"></a>

# Examples

Some examples scripts are provided in the repository <https://github.com/aheays/spectr_examples>


<a id="orgbb75b11"></a>

# Submodules


<a id="org89f5cc3"></a>

## `env.py`

Conveniently import all submodules.


<a id="org56f7c45"></a>

## `dataset.py`

Storage, manipulation, and plotting of tabular data. Allows for the
recursive calculation of derived quantities


<a id="orgaaa27c7"></a>

## `tools.py`

Functions for performing common mathematical and scripting tasks.


<a id="org7b92b0d"></a>

## `plotting.py`

Functions for plotting built on matplotlib.


<a id="org05a186c"></a>

## `convert.py`

Unit conversion, species name conversion, and various conversion formulae.


<a id="org60446f5"></a>

## `optimise.py`

General class for conveniently and hierarchically building numerical
models with optimisable parameters.


<a id="orgd9bcc3d"></a>

## `atmosphere.py`

Classes for analysing atmospheric photochemistry.


<a id="org20edc6c"></a>

## `lines.py`

Dataset subclasses for storing atomic and molecular line data.


<a id="org1223f31"></a>

## `levels.py`

Dataset subclasses for storing atomic and molecular level data.


<a id="orgd636383"></a>

## `bruker.py`

Interact with output files of Bruker OPUS spectroscopic acquisition
and analysis software. 


<a id="org1a7110f"></a>

## `database.py`

Interface to internal spectroscopic and chemistry database.  


<a id="orgdc1843f"></a>

## `electronic_states.py`

Calculation of diatomic level energies from potential-energy curves.


<a id="orgf9e2f8a"></a>

## `exceptions.py`

Exception used to internally communicate failure conditions.


<a id="org4bd04b2"></a>

## `hitran.py`

Access HITRAN spectroscopic data with hapy.


<a id="org79dedd5"></a>

## `lineshapes.py`

Simulate individual and groups of spectra lines of various shapes.


<a id="orga5fce86"></a>

## `quantum_numbers.py`

Functions for manipulating atomic and molecular quantum numbers.


<a id="org750f7ad"></a>

## `spectrum.py`

Classes for manipulating and modelling of experimental spectroscopic datea.


<a id="org0b5c98f"></a>

## `thermochemistry.py`

Functions for computing thermochemical equilibrium with ggchem.


<a id="org5f368e1"></a>

## `viblevel.py`

Classes for simulating diatomic levels and lines defined by effective Hamiltonians.


<a id="orgbaa288a"></a>

## `fortran_tools.f90`

Various fortran functions and subroutines.


<a id="org12464c8"></a>

# Bugs / improvements


<a id="org28258b0"></a>

## optimise.py


### inhibit `add_input_function` in `input_function_method`?


<a id="orgbdaebe1"></a>

## viblevel.py


### Implement general Λ-doubling formula of brown1979

Currently the o/p/q Λ-doubling is handled with effective
(S,Λ)-dependent forumulae.  Instead implement the last three terms of
Eq. 18 of brown1979 into \_get<sub>linear</sub><sub>H</sub>()
.


### Phase error in ⟨³Π|LS|¹Δ⟩

When comparing thismodel with pgopher, everything works find except
the sign of the interactions a³Π(v=12)~D¹Δ(v=1), a³Π(v=12)~d³Δ(v=5),
and a³Π(v=12)~d³Δ(v=6) needs to be reversed. There is a phase error
between these interactions and others.

    
    ##rafals draft 2021-06-24
    ## 
    ## crossing states
    upper_13C18O.add_level('A¹Π(v=1)',Tv=66175.53765,Bv=1.43761743,Dv=6.11179e-06,Hv=-22.39e-12,)
    upper_13C18O.add_level('D¹Δ(v=1)',Tv=66442.5076,Bv=1.12,Dv=5.79e-6,Hv=-0.22e-12,)
    upper_13C18O.add_level('I¹Σ⁻(v=2)',Tv=66595.57091,Bv=1.1146473,Dv=5.68e-6,Hv=2.25e-12,)
    upper_13C18O.add_level('d³Δ(v=6)',Tv=66956.97424,Bv=1.09416857,Dv=5.31e-6,Hv=-0.60e-12,Av=-16.097,ADv=-9.17e-5,λv=0.94,γv=0.76e-2,)
    upper_13C18O.add_level('e³Σ⁻(v=3)',Tv=66811.0988,Bv=1.1126549,Dv=5.55e-6,Hv=-1.50e-12,λv=0.5278,)
    # ## non-crossing states
    upper_13C18O.add_level('d³Δ(v=5)',Tv=65949.55,Bv=1.11,Dv=5.33e-6,Hv=-0.60e-12,Av=-15.91,ADv=-9.17e-5,λv=0.85,γv=0.69e-2,)
    upper_13C18O.add_level('e³Σ⁻(v=2)',Tv=65802.44,Bv=1.13,Dv=5.58e-6,Hv=-1.50e-12,λv=0.54,)
    upper_13C18O.add_level('I¹Σ⁻(v=1)',Tv=65593.17,Bv=1.13,Dv=5.67e-6,Hv=2.25e-12,)
    upper_13C18O.add_level('a′³Σ⁺(v=10)',Tv=66066.95,Bv=1.07,Dv=5.17e-6,Hv=-0.30e-12,)
    upper_13C18O.add_level('a′³Σ⁺(v=11)',Tv=67037.79,Bv=1.05,Dv=5.16e-6,Hv=-0.30e-12,λv=-108.84e-2,γv=-0.50e-2,)
    upper_13C18O.add_level('a³Π(v=12)',Tv=66355.00,Bv=1.32,Dv=5.67e-6,Av=36.97,ADv=-20.58e-5,λv=-0.49e-2,γv=0.33e-2,ov=0.64,pv=2.73e-3,qv=2.95e-5,)
    # ## interactions with crossing states
    upper_13C18O.add_coupling('A¹Π(v=1)','D¹Δ(v=1)',ξv=-6.1688e-2),
    upper_13C18O.add_coupling('A¹Π(v=1)','I¹Σ⁻(v=2)',ξv=7.630e-2)
    upper_13C18O.add_coupling('A¹Π(v=1)','d³Δ(v=6)',ηv=18.0838)
    upper_13C18O.add_coupling('A¹Π(v=1)','e³Σ⁻(v=3)',ηv=-5.4206)# ## interactions with non-crossing states
    upper_13C18O.add_coupling('A¹Π(v=1)','d³Δ(v=5)',ηv=15.57)
    upper_13C18O.add_coupling('A¹Π(v=1)','e³Σ⁻(v=2)',ηv=14.05)
    upper_13C18O.add_coupling('A¹Π(v=1)','I¹Σ⁻(v=1)',ξv=9.89e-2)
    upper_13C18O.add_coupling('A¹Π(v=1)','a′³Σ⁺(v=10)',ηv=-5.29)
    upper_13C18O.add_coupling('A¹Π(v=1)','a′³Σ⁺(v=11)',ηv=3.836)
    ## interactions not including A
    upper_13C18O.add_coupling('a³Π(v=12)','I¹Σ⁻(v=2)',ηv=-7.604)
    # upper_13C18O.add_coupling('a³Π(v=12)','D¹Δ(v=1)',ηv=-7.955)
    # upper_13C18O.add_coupling('a³Π(v=12)','d³Δ(v=5)',ηv=-38.48,ξv=7e-2)
    # upper_13C18O.add_coupling('a³Π(v=12)','d³Δ(v=6)',ηv=26.31,ξv=5.80e-2)
    upper_13C18O.add_coupling('a³Π(v=12)','D¹Δ(v=1)',ηv=7.955)
    upper_13C18O.add_coupling('a³Π(v=12)','d³Δ(v=5)',ηv=38.48,ξv=-7e-2)
    upper_13C18O.add_coupling('a³Π(v=12)','d³Δ(v=6)',ηv=-26.31,ξv=-5.80e-2)
    upper_13C18O.add_coupling('a³Π(v=12)','e³Σ⁻(v=2)',ηv=5.09,ξv=1.00e-2)
    upper_13C18O.add_coupling('a³Π(v=12)','e³Σ⁻(v=3)',ηv=8.24,ξv=1.60e-2)

