from spectr import *



results = {}
for i in range(300):
    print( i)

    ## define fucnction
    x = np.linspace(0,10,1000)
    y = (
        # 1 + 2*x
        + 0.2*np.sin(2*x)
        + lineshapes.gaussian(x,4,1,0.5)
        + 0.02*tools.randn(x.shape)
        # + 0.5*x**2
    )

    ## optimiser to model it
    o = Optimiser(
        # m=P(0, True,1e-3),
        # c=P(0, True,1e-3),
        x0=P(3, True,1e-3),
        S=P(4, True,1e-3),
        w=P(2, True,1e-3),
        # a=P(0.1, True,1e-3),
        # b=P(2.1, True,1e-3),
        verbose=False)
    o.add_construct_function(
        lambda: (
            y - (
                # float(o['c'])+float(o['m'])*x
                + lineshapes.gaussian(x,float(o['x0']),float(o['S']),float(o['w']))
                # +float(o['a'])*np.sin(float(o['b'])*x)
            )))

    ## collect statistics
    o.optimise(
        monitor_frequency='never',
        verbose=False,
        # rms_noise=0.2 
    )
    for key in o:
        if key not in results:
            results[key] = []
            results['d'+key] = []
        results[key].append(o[key].value)
        results['d'+key].append(o[key].uncertainty)

## plot statistics
qfig(1)
subplot()
plot(x,y)
plot(x,o.residual)
for key in o:
    subplot();plot_hist_with_fitted_gaussian(results[key]);title(key)
    subplot();plot_hist_with_fitted_gaussian(results['d'+key]);title('d'+key)
