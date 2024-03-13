# `spectr`

## Installation

### Description

A real grab-bag of tools for spectroscopic data analysis and modelling.

### Source code

Source code is available at  <https://github.com/aheays/spectr>.
The repository can be cloned locally with `git clone --depth=1 https://github.com/aheays/spectr.git` or a zip archive downloaded from [GitHub](https://github.com/aheays/spectr/tags).

### External dependencies

#### Python

Version 3.11 or above will be necessary. 

#### Fortran extensions

It is necessary to have gfortran and lapack available on the system before installation to enable fortran extensions.

The necessary linux packages are
-   Debian / Ubuntu: `gfortran python3-dev liblapack-dev`
-   Arch: `gcc-gfortran lapack`


The extension can be build in the `spectr/` subdirectory with the following command.

    f2py3 -c --quiet -llapack --f90flags="-Wall -ffree-line-length-none" --opt="-O3" fortran_tools.f90

#### Matplotlib dependencies

Under ubuntu, installing the package `qt5-default` seem to be enough to get matplotlib to draw windowed figures.

### Building and installing in a Python virtual environment

The following shell commands should get things going in a linux environment if the external dependencies are met.

-   Get external dependencies.  On Ubuntu: `sudo apt update && sudo apt install gfortran python3-dev liblapack-dev qt5-default`
-   Clone spectr: `git clone --depth=1 https://github.com/aheays/spectr.git`
-   Create a virtual environment using python: `python -m venv env`
-   Activate the virtual environment: `source env/bin/activate`
-   Build and install: `pip install ./spectr`

### Testing the installation

Test by importing spectr and trying to plot something

    source env/bin/activate
    echo "from spectr.env import *" | python
    qplot absorption/data/2021_11_30_bcgr.0


Examples of using `spectr` are provided [`spectr_examples`](https://github.com/aheays/spectr_examples)

## Usage
    
### Importing `spectr`

A command to import all submodules and many common functions directly into the working namespace is `from spectr.env import *`.  Otherwise only the needed submodules can be imported, e.g., `import spectr.spectrum`



### Optimising things

The `optimiser.Optimiser` class is used to conveniently construct model objects with parameters that can be fit to experimental data. The real-number input arguments of most methods of objects base-classed on `Optimiser` can be marked for optimisation by replacing their values with a optimiser.Parameter object.  This has the abbreviated definition:

    P(value=float,
      vary=True|False,
      step=float,
      uncertainty=float,
      bounds=(float,float))

Only the first argument is required. For example, `x=P(2,True,1e-5,bounds=(0,100))` defines a parameter `x` that will be varied from an initial value of 2 but constrained to the range 0 to 100.  When computing the finite-difference approximation to the linear dependence of model error on `x` a step size of \num{e-5} will be used.  The fitting uncertainty `unc` will be set automatically after optimisation.
Multiple `Optimiser` objects can be combined in a hierarchy, so that multiple spectra can be fit at once to optimise a common parameter, for example a temperature-dependence coefficient fit to spectra at multiple temperatures.



### Encoding linear molecule quantum numbers

TBD



### `qplot`

This is a command line programming for making line plots, e.g., `qplot datafile`, or `qplot -h` for a list of options.



## Examples

Some examples scripts are provided in the repository <https://github.com/aheays/spectr_examples>



## Submodules



### `env.py`

Conveniently import all submodules.



### `dataset.py`

Storage, manipulation, and plotting of tabular data. Allows for the
recursive calculation of derived quantities



### `tools.py`

Functions for performing common mathematical and scripting tasks.



### `plotting.py`

Functions for plotting built on matplotlib.



### `convert.py`

Unit conversion, species name conversion, and various conversion formulae.



### `optimise.py`

General class for conveniently and hierarchically building numerical
models with optimisable parameters.



### `atmosphere.py`

Classes for analysing atmospheric photochemistry.



### `lines.py`

Dataset subclasses for storing atomic and molecular line data.



### `levels.py`

Dataset subclasses for storing atomic and molecular level data.



### `bruker.py`

Interact with output files of Bruker OPUS spectroscopic acquisition
and analysis software. 



### `database.py`

Interface to internal spectroscopic and chemistry database.  



### `electronic_states.py`

Calculation of diatomic level energies from potential-energy curves.



### `exceptions.py`

Exception used to internally communicate failure conditions.



### `hitran.py`

Access HITRAN spectroscopic data with hapy.



### `lineshapes.py`

Simulate individual and groups of spectra lines of various shapes.



### `quantum_numbers.py`

Functions for manipulating atomic and molecular quantum numbers.



### `spectrum.py`

Classes for manipulating and modelling of experimental spectroscopic datea.



### `thermochemistry.py`

Functions for computing thermochemical equilibrium with ggchem.



### `viblevel.py`

Classes for simulating diatomic levels and lines defined by effective Hamiltonians.



### `fortran_tools.f90`

Various fortran functions and subroutines.



## Bugs / improvements



### optimise.py


#### inhibit `add_input_function` in `input_function_method`?



### viblevel.py


#### Implement general Λ-doubling formula of brown1979

Currently the o/p/q Λ-doubling is handled with effective
(S,Λ)-dependent forumulae.  Instead implement the last three terms of
Eq. 18 of brown1979 into \_get<sub>linear</sub><sub>H</sub>()
.


#### Phase error in ⟨³Π|LS|¹Δ⟩

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
