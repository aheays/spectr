# to do

- Fix up prototypes in lines_prototypes and level_prototypes

- Change _u/_l to ′″ and d_ to σ

- Improve Dataset indexing to cases:
  - get_value / get_uncertainty -- get numpy arrays
  - copy(keys,index) -- get copy of self with these limits
  - [key] -- return entire numpy array of key or scalar value
  - [int] -- return copy of self with all scalar data at int
  - [slice] -- return copy of self reduced to slice

- optim: introduce P and PD objects
- optim: Add decorator for adding construct functions
- optim: add new output of parameters based on
  index/optimiser/function/arg_parameter
