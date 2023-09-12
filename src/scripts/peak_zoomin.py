import arviz as az
import intensity_models as im
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import pandas as pd
import paths
import seaborn as sns

if __name__ == '__main__':
    sns.set_palette('colorblind')
    trace = az.from_netcdf(op.join(paths.data, 'trace.nc'))
    p = trace.posterior

    ms = np.linspace(20, 60, 1024)
    dNs = np.zeros((len(p.chain), len(p.draw), len(ms)))

    for c in p.chain:
        for d in p.draw:
            args = map(float, [p.a[c,d], p.b[c,d], p.c[c,d], p.mpisn[c,d], p.mbhmax[c,d], p.sigma[c,d], p.fpl[c,d], p.beta[c,d], p.lam[c,d], p.kappa[c,d], p.zp[c,d]])
            log_dN = im.LogDNDMDQDV(*args)
            dN = float(p.R[c,d]) * np.exp(log_dN(ms, 1.0, 0.0))
            dNs[c,d,:] = dN

    plt.plot(ms, ms * np.median(dNs, axis=(0,1)), label=None, color=sns.color_palette()[0])

    print(dNs[0,0,:])

    for _ in range(100):
        plt.plot(ms, ms * dNs[np.random.choice(p.chain), np.random.choice(p.draw), :], label=None, color=sns.color_palette()[0], alpha=0.1)

    plt.xlabel(r'$m_1 / M_\odot$')
    plt.ylabel(r'$\left. m_1 \mathrm{d} N / \mathrm{d} m_1 \mathrm{d} q \mathrm{d} V \mathrm{d} t \right|_{q=1, z=0} / \mathrm{Gpc}^{-3} \, \mathrm{yr}^{-1}$')

    plt.xscale('log')
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig(op.join(paths.figures, 'peak-zoomin.pdf'))