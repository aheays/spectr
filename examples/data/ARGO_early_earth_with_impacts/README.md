# ARGO
Lagrangian High Energy Chemistry/Diffusion Code for Planet Atmospheres

## How to run the code.

Follow these steps:

1. [Clone the repository](https://help.github.com/en/articles/cloning-a-repository)
1. Go to the folder, and make the executable: Type `make`
1. Type `mkdir out` 
1. Type `cd out`
1. Type `mkdir Figures Plot Reactions`
1. Type `cd ..`
1. Run RTdepthrun.py: type `python RTdepthrun.py` **(Warning! This is a Python2 script and has not been updated to Python3!)**

This will read in the input files for modern day Earth, set by RTheader.py, solve the chemistry using the 'Stand2019May' network, and produce output files in the '.\out\' folder.

## Python and F77 Files

The list of Fortran 77 files is as follows:

* argo.f -- This is a 0D chemistry model code.
* dlsode.f -- This is the differential equation solver.

The list of python files is as follows:

* RTdepthrun.py -- This is the main python file that runs the entire model
* RTheader.py -- This is a header file that provides the names for the input files, input parameters, and, for the temperature profile, identifies which columns are which.
* RTlifetime.py -- This calculates lifetimes.
* RTmakedepth.py -- This constructs the ./out/depth.dat file.
* RTplot.py -- This constructs the **Figures** files.
* RTrenormalize.py -- Checks that all the mixing ratios, summed, add to one. If not, this adds them altogether, and then divides all mixing ratios by that number, and produces a warning that mixing ratios do not add to one.
* RTsavedata.py -- Run this to save the output data to a new folder.
* RTsetdown.py -- Keeps track of the banked species.
* RTtest.py -- Tests global convergence.
* radcalc.py -- Calculates the UV transport.

In addition, the **Makefile** is used to build the executible **argo** from **argo.f** and **dlsode.f**.

## Input files

The input files are as follows:

* blocked.dat -- The actinic flux read into argo, appropriate for the atmospheric height.
* input.dat -- The input chemical composition of the atmosphere, appropriate for the height of the parcel.
* input_parameter.dat -- The physical parameters (T,p,n, etc.) appropriate for the atmospheric height.
* modern-solar.dat -- The flux from the sun at the top of the atmosphere of the modern Earth.
* outgas.dat -- The outgassing rates for degassed species.
* photo-sigma.dat -- The photochemical cross-sections as a function of wavelength.
* random.dat -- File ussed for exploring chemical rate sensitivities, not yet implemented in this model.
* reservoir.dat -- The reservoirs and timescales for degassed species.
* resource.dat -- The source file for the reservoirs, used with calculated timescales (from **RTlifetime.dat**) to generate **reservoir.dat**
* source.dat -- The source file used by **RTdepthrun.dat** to generate **input_parameter.dat**
* timeres.dat -- Prescribed time-steps that have to be reached, depending on the dynamic timescales.

## Output files

The output files are as follows:

* Kout.dat -- The rates for each reaction at the last timestep for the parcel at a given atmospheric height.
* out/Reactions/Kup.dat and out/Reactions/Kdown.dat -- The last-timestep rates for each reaction for each atmospheric height, for upward and downward moving parcels.
* output.dat -- The output chemistry before the parcel moves.
* out/depth-up.dat out/depth-down.dat -- The chemical profiles tracked from upward and downward moving parcels.
* out/depth.dat -- The combined **final chemical atmospheric profiles.**
* out/old-depth-up.dat out/old-depth-down.dat out/old-depth.dat -- Output files from the last global calculation.


## Citations
