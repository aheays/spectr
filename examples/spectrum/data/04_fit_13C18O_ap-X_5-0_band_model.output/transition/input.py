from spectr import *

lower = viblevel.VibLevel(name='lower',species='[13C][18O]',Eref=0.0)
lower.add_level(
    name='X.1Σ+(v=0)',
    Tv=1031.055619,
    Bv=1.746408199,
    Dv=5.0488146e-06,
    Hv=4.35471e-12,
)

upper = viblevel.VibLevel(name='upper',species='[13C][18O]',Eref=0.0)
upper.add_level(
    name='A.1Π(v=0)',
    Tv=65793.805619,
    Bv=1.4574165,
    Dv=6.083e-06,
)
upper.add_level(
    name='ap.3Σ+(v=5)',
    Tv=P(61981.6505291,True,0.001,0.0029),
    Bv=P(1.13803219739,True,1e-05,8.6e-06),
    Dv=P(4.00454879896e-06,True,1e-09,1.2e-07),
    λv=P(-1.22024568535,True,0.001,0.0011),
    γv=P(0.00132309649171,True,1e-05,0.0001),
)
upper.add_LS_coupling(
    ηv=1,
    name1='A.1Π(v=0)',
    name2='ap.3Σ+(v=5)',
)

transition = viblevel.VibLine(name='transition',level_u=upper,level_l=lower,J_l=range(0, 31))
transition.add_transition_moment(
    μv=1,
    name_u='A.1Π(v=0)',
    name_l='X.1Σ+(v=0)',
)