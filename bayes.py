from .ferramentas import bayesian_inference_grid, janeladata,rebin_spectrum
import numpy as np
import pandas as pd

def executar_bayes(
    OBS,
    Dir_OBS2,
    FEXP,
    valores_usados,
    windows,
    dl,
    ordem_parametros,
    passo,
    priors,
    variar
):


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
# ETAPA - REBIN PARA A BAYESIAN
    # janela observacional
    OBS_janela = janeladata(windows, OBS)

    # janelas de TODOS os modelos
    modelos_jan = [janeladata(windows, df) for df in FEXP]    
    
    OBS_stat = rebin_spectrum(OBS_janela, dl)
    modelos_stat = [rebin_spectrum(m, dl) for m in modelos_jan]
    
    #print("Modelos carregados:", len(modelos_stat))
    #print("Modelos esperados:", len(valores_usados))
#----------------------------------------------------------------------------------------------------------------------
## ETAPA - RUN BAYESIAN INFERENCE
#--------------------------------------------------------------------------------------------------------------------- 
    if not Dir_OBS2:
        output = bayesian_inference_grid(OBS_stat,modelos_stat,valores_usados,ordem_parametros,
        passo=passo,priors=priors,dl=dl,npar=len(variar))
    else:
        output = bayesian_inference_grid(OBS_stat,modelos_stat,valores_usados,ordem_parametros,
        passo=passo,priors=priors,dl=dl,npar=len(variar))
        
    posterior = output["posterior"]
    CHI2 = output["chi2"]
    resultados  = output["results"]
    
    i_min2 = np.argmin(CHI2)
    
    MinChi2Model = FEXP[i_min2]
    MinChi2Model_values = valores_usados[i_min2]

# ETAPA - Maximum A Posteriori ----------------------------------------------------------------------------------
    i_map = np.argmax(posterior)

    Best_model_MAP_stat = modelos_stat[i_map]   # rebinado
    Best_model_MAP      = FEXP[i_map]           # alta resolução
    Best_values_MAP     = valores_usados[i_map]
    
    return {
    "posterior": posterior,
    "results": resultados,
    "MinChi2Model": MinChi2Model,
    "MinChi2Model_values": MinChi2Model_values,
    "Best_model_MAP": Best_model_MAP,
    "Best_values_MAP": Best_values_MAP,
    "i_min2":i_min2
}
