## 13C18O a'(5)-A(0) SOLEIL absorption spectrum to a band model, begin with a band model results and individually adjust some upper levels.
from spectr import *

## load band model data -- a' (ap) upper level only
line = dataset.load('data/04_fit_13C18O_ap-X_5-0_band_model.output/transition.line/dataset.h5',name='line')
line.limit_to_match(label_u='ap')

## adjust temperature and column density
line.set_parameter(key='Teq',value=P(274.1471053, True,0.1,2.3e-07),)
line.set_parameter(key='Nself',value=P(1.61200479151e+20, True,1e+18,9.9e-09),)

## move some upper levels
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 0.0 , 'ef_u': -1 , 'J_u': 1.0}  , key='E_u' , value=P(61989.4411493 , True , 0.001 , 2.1e-07) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 0.0 , 'ef_u': -1 , 'J_u': 2.0}  , key='E_u' , value=P(61996.1033115 , True , 0.001 , 8.8e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 0.0 , 'ef_u': -1 , 'J_u': 3.0}  , key='E_u' , value=P(62005.0931996 , True , 0.001 , 8.5e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 0.0 , 'ef_u': -1 , 'J_u': 4.0}  , key='E_u' , value=P(62016.4302467 , True , 0.001 , 4.5e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 0.0 , 'ef_u': -1 , 'J_u': 5.0}  , key='E_u' , value=P(62030.0257516 , True , 0.001 , 3.9e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 0.0 , 'ef_u': -1 , 'J_u': 6.0}  , key='E_u' , value=P(62045.9329822 , True , 0.001 , 4e-08)   , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 0.0 , 'ef_u': -1 , 'J_u': 7.0}  , key='E_u' , value=P(62064.110027  , True , 0.001 , 3.5e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 0.0 , 'ef_u': -1 , 'J_u': 8.0}  , key='E_u' , value=P(62084.5429665 , True , 0.001 , 3.5e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 0.0 , 'ef_u': -1 , 'J_u': 9.0}  , key='E_u' , value=P(62107.2719756 , True , 0.001 , 3.7e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 0.0 , 'ef_u': -1 , 'J_u': 10.0} , key='E_u' , value=P(62132.2543865 , True , 0.001 , 3.9e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 0.0 , 'ef_u': -1 , 'J_u': 11.0} , key='E_u' , value=P(62159.5419146 , True , 0.001 , 4.4e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 0.0 , 'ef_u': -1 , 'J_u': 12.0} , key='E_u' , value=P(62189.0724357 , True , 0.001 , 5e-08)   , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 0.0 , 'ef_u': -1 , 'J_u': 13.0} , key='E_u' , value=P(62220.886332  , True , 0.001 , 5.9e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 0.0 , 'ef_u': -1 , 'J_u': 14.0} , key='E_u' , value=P(62254.9346953 , True , 0.001 , 7.1e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 0.0 , 'ef_u': -1 , 'J_u': 15.0} , key='E_u' , value=P(62291.2534314 , True , 0.001 , 8.8e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 0.0 , 'ef_u': -1 , 'J_u': 16.0} , key='E_u' , value=P(62329.930391  , True , 0.001 , 1.1e-07) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 0.0 , 'ef_u': -1 , 'J_u': 17.0} , key='E_u' , value=P(62370.7638129 , True , 0.001 , 1.5e-07) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': -1 , 'J_u': 1.0}  , key='E_u' , value=P(61981.48139   , True , 0.001 , 5.5e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': -1 , 'J_u': 2.0}  , key='E_u' , value=P(61983.9751602 , True , 0.001 , 4e-08)   , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': -1 , 'J_u': 3.0}  , key='E_u' , value=P(61988.6364531 , True , 0.001 , 3.3e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': -1 , 'J_u': 4.0}  , key='E_u' , value=P(61995.5085162 , True , 0.001 , 3e-08)   , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': -1 , 'J_u': 5.0}  , key='E_u' , value=P(62004.6513387 , True , 0.001 , 2.8e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': -1 , 'J_u': 6.0}  , key='E_u' , value=P(62016.0484057 , True , 0.001 , 2.8e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': -1 , 'J_u': 7.0}  , key='E_u' , value=P(62029.7215998 , True , 0.001 , 2.8e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': -1 , 'J_u': 8.0}  , key='E_u' , value=P(62045.6589774 , True , 0.001 , 2.9e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': -1 , 'J_u': 9.0}  , key='E_u' , value=P(62063.8873354 , True , 0.001 , 3.1e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': -1 , 'J_u': 10.0} , key='E_u' , value=P(62084.3646688 , True , 0.001 , 3.5e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': -1 , 'J_u': 11.0} , key='E_u' , value=P(62107.1400458 , True , 0.001 , 4e-08)   , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': -1 , 'J_u': 12.0} , key='E_u' , value=P(62132.2071528 , True , 0.001 , 4.6e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': -1 , 'J_u': 13.0} , key='E_u' , value=P(62160.3805109 , True , 0.001 , 5.5e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': -1 , 'J_u': 14.0} , key='E_u' , value=P(62188.8437187 , True , 0.001 , 6.8e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': -1 , 'J_u': 15.0} , key='E_u' , value=P(62220.7145469 , True , 0.001 , 8.5e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': -1 , 'J_u': 16.0} , key='E_u' , value=P(62254.836401  , True , 0.001 , 1.1e-07) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': -1 , 'J_u': 17.0} , key='E_u' , value=P(62291.2688321 , True , 0.001 , 1.4e-07) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': 1  , 'J_u': 1.0}  , key='E_u' , value=P(61983.0903727 , True , 0.001 , 6.4e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': 1  , 'J_u': 2.0}  , key='E_u' , value=P(61987.6742122 , True , 0.001 , 5.5e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': 1  , 'J_u': 3.0}  , key='E_u' , value=P(61994.4863161 , True , 0.001 , 3.2e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': 1  , 'J_u': 4.0}  , key='E_u' , value=P(62003.5935778 , True , 0.001 , 2.5e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': 1  , 'J_u': 5.0}  , key='E_u' , value=P(62014.9668633 , True , 0.001 , 2.3e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': 1  , 'J_u': 6.0}  , key='E_u' , value=P(62028.6207497 , True , 0.001 , 2.1e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': 1  , 'J_u': 7.0}  , key='E_u' , value=P(62044.5476792 , True , 0.001 , 2.1e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': 1  , 'J_u': 8.0}  , key='E_u' , value=P(62062.7498207 , True , 0.001 , 2.2e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': 1  , 'J_u': 9.0}  , key='E_u' , value=P(62083.2188591 , True , 0.001 , 2.3e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': 1  , 'J_u': 10.0} , key='E_u' , value=P(62105.964056  , True , 0.001 , 2.5e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': 1  , 'J_u': 11.0} , key='E_u' , value=P(62130.9957016 , True , 0.001 , 2.8e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': 1  , 'J_u': 12.0} , key='E_u' , value=P(62158.2722221 , True , 0.001 , 3.2e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': 1  , 'J_u': 13.0} , key='E_u' , value=P(62187.9017829 , True , 0.001 , 3.7e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': 1  , 'J_u': 14.0} , key='E_u' , value=P(62219.8010235 , True , 0.001 , 4.5e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': 1  , 'J_u': 15.0} , key='E_u' , value=P(62254.1937644 , True , 0.001 , 5.5e-08) , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': 1  , 'J_u': 16.0} , key='E_u' , value=P(62287.5829237 , True , 0.001 , 7e-08)   , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': 1  , 'J_u': 17.0} , key='E_u' , value=P(62327.8877936 , True , 0.001 , 9e-08)   , )
line.set_parameter(match={'species_u': '[13C][18O]' , 'label_u': 'ap' , 'v_u': 5 , 'Σ_u': 1.0 , 'ef_u': 1  , 'J_u': 18.0} , key='E_u' , value=P(62369.5353707 , True , 0.001 , 1.2e-07) , )

## load experimental data
experiment = spectrum.Experiment('experiment',)
experiment.set_spectrum_from_soleil_file(filename='data/13C18O_spectrum.h5')

## construct a model
model = spectrum.Model('model',experiment)
model.add_intensity_spline(knots=[(60600, P(0.0772604178868,False,0.0001,6.1e-05)), (61100, P(0.115131675836,False,0.0001,6.1e-05))],)
model.add_absorption_lines(lines=line,lineshape='gaussian')
model.convolve_with_soleil_instrument_function()

## optimise everything
model.optimise(monitor_parameters=False)
# model.save_to_directory('td')

## plot spectrum
qfig(1)
model.plot(plot_labels=True)

## plot reduced upper energy levels of the a' state
qfig(2)
line.plot('J_u',('E_reduced_u',))


show()
