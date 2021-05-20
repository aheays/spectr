from spectr import *

## lower state
lower = viblevel.VibLevel(name='lower',species='[32S][16O]')
lower.add_level(name='X.3Σ+(v=0)', Tv=0, Bv=1, λv=1, γv=0.01)

## upper state
upper = viblevel.VibLevel(name='upper',species='[32S][16O]')
upper.add_level(name='B.3Π(v=0)', Tv=50000, Bv=1.1, Av=10) 

## transition
transition = viblevel.VibLine('transition',upper,lower,J_l=range(31),)
transition.add_transition_moment('B.3Π(v=0)','X.3Σ+(v=0)',μv=1)
transition.construct()

## save output
transition.line['ν']
transition.line.save('t0.psv')

## plot
qfig(1)
lower.plot()
title('Lower level energies')

qfig(2)
upper.plot()
title('Upper level energies')

qfig(3)
transition.plot()
title('Linestrengths by branch')

show()
