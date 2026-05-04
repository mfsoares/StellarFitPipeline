from windows import plot_windows
import os
import pandas as pd
import numpy as np
from ferramentas import chi2_scaled

def carregar_modelos_e_chi2(
    valores_usados,
    pasta_destino,
    OBS,
    scale,
    windows,
    lam_min,
    lam_max,
    fwhm
):
    CHI2 = []
    FEXP =[]
# ───────────────────────────────────────────────
# ETAPA - USO DOS MODELOS
# ───────────────────────────────────────────────
    for pars in valores_usados:

        nome = f"teff_{pars['teff']:.0f}_logg_{pars['logg']:.2f}_vsini_{pars['vsini']:.0f}_vturb_{pars['vturb']:.0f}_vmac_{pars['vmac']:.0f}_C_{pars['c']:.2f}_N_{pars['n']:.2f}_O_{pars['o']:.2f}_Si_{pars['si']:.2f}_lim_{lam_min:.0f}_{lam_max:.0f}_fwhm_{fwhm:.3f}.parquet"

        caminho = os.path.join(pasta_destino, nome)
    
        #print("Tentando abrir:", caminho)

        if not os.path.exists(caminho):
            print("Arquivo não existe!")
            continue

        df = pd.read_parquet(caminho)

        wave = df['WaveLength'].values
        flux = df['NormFluxo'].values

        fexp = np.interp(OBS['WaveLength'], wave, scale*flux)

        FEXP.append(pd.DataFrame({
        'WaveLength': OBS['WaveLength'],
        'NormFluxo': fexp }))
        chi2_scaled
        
    
#----------------------------------------------------------------------------------------------------------------
# ETAPA - CHI2   
    for i in range(len(FEXP)):
       Chi2 = chi2_scaled(windows, OBS, FEXP[i])
       CHI2.append(Chi2)
    i_min = np.argmin(CHI2)
    MinChi2 = FEXP[i_min]
    MinChi2_values = valores_usados[i_min]
    
    return FEXP, CHI2, MinChi2, MinChi2_values
