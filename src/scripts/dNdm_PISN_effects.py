import intensity_models
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import paths
import seaborn as sns
import weighting

a, b, mpisn, mbhmax, sigma = weighting.a, weighting.b, weighting.mpisn, weighting.mbhmax, weighting.sigma

with sns.color_palette('husl', n_colors=5):
    m = np.linspace(5, 45, 1024)
    def plot_mass_model(dndm, label=None):
        dndm = dndm / np.trapz(dndm, m)
        plt.plot(m, dndm, label=label)

    plot_mass_model(np.exp(intensity_models.LogDNDMPISN(a, b, mpisn, mbhmax, sigma)(m).eval()), label='Default')
    plot_mass_model(np.exp(intensity_models.LogDNDMPISN(a, b, mpisn*1.1, mbhmax*1.1, sigma)(m).eval()), label='Mass + 10%')
    plot_mass_model(np.exp(intensity_models.LogDNDMPISN(a, b, mpisn, mbhmax, sigma-1)(m).eval()), label=r'$\sigma - 1$')
    plot_mass_model(np.exp(intensity_models.LogDNDMPISN(a, b, mpisn*1.1, mbhmax, sigma)(m).eval()), label=r'$m_\mathrm{PISN} + 10\%$')
    plot_mass_model(np.exp(intensity_models.LogDNDMPISN(a, b, mpisn, mbhmax*1.1, sigma)(m).eval()), label=r'$m_\mathrm{BH,max} + 10\%$')

    plt.legend()
    plt.xlabel(r'$m / M_\odot$')
    plt.ylabel(r'$p(m)$')

    plt.tight_layout()
    plt.savefig(op.join(paths.figures, 'dNdm_PISN_effects.pdf'))