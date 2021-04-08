from spectr import *

lower = viblevel.VibLevel(name='lower',species='[13C][18O]',Eref=0.0)
lower.add_level(
    name='X.1Σ+(v=0)',
    Tv=1031.055619,
    Bv=1.746408199,
    Dv=5.0488146e-06,
    Hv=4.35471e-12,
)