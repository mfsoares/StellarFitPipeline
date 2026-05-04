from .io import ler_ascii_obs
from .io import carregar_obs
from .modeling import gerar_modelos
from .statistics import carregar_modelos_e_chi2
from .bayes import executar_bayes
from .plot import spectra_plot
import pandas as pd
import numpy as np
import os

def execute_SFP(
    variar=None,              # 'logg' OU 'vsini' OU 'teff' OU lista
    priors=None,              # Reference values from literature or other method. 
    par_central=None,         # 
    passo=None,               # float ou dict
    number=None,              # int
    star=None,
    model_lim=None,
    Dir_OBS=None,
    Dir_OBS2=None, 
    windows=None,
    Compared=None,
    xlim=None,
    xlim2=None,
    ylim=None,
    ylim2=None,
    scale=1,
    scale_obs2=1,
    deltax=0,
    deltax2=0,
    deltaobs=0,
    deltaobs2=0,
    figsize=(18,15),
    model_spectra=False,
    plotlegend=False,
    only_bestfit=False,
    bbox=None,
    savefigure=False,
    ylimchi2=None,
    fwhm=0.25,
    leg_painel=None,
    legloc=None,
    chi2_loc=None,
    plot_2=True,
    ws=None,
    corte=None,
    line_high=None,
    image_grid=None,
    alpha=0.6,
    hl=10,
    lwbf=2,
    d=1.5,
    dl=0.1,
    path=None
):
    # Este código permite variar vários parâmetros 
    ## É uma funcão que roda o Synplot variando até nove parâmetros um de cada vez de forma automática.
    # os parâmetros estão na lista abaixo ordem_parametros.
    
    # Como exemplo, para usar é só rodar no jupyter notebook o seguinte comando: 
    # Em variar vc pode colordem_parametros = ['teff', 'logg', 'vsini','vturb','vmac','c','n','o','si']ocar todos os parametros.
    
    wstart = model_lim[0]
    wend=model_lim[1]
    ordem_parametros = ['teff', 'logg', 'vsini','vturb','vmac','c','n','o','si']
    
    # =====================================================
    # 1) LEITURA OBSERVACIONAL
    # =====================================================

    OBS, arq_out = carregar_obs(
        Dir_OBS,
        Dir_OBS2,
        corte
    )

    # =====================================================
    # 2) GERAÇÃO DOS MODELOS
    # =====================================================

    valores_usados, pasta_destino, par_vals = gerar_modelos(
        par_central=par_central,
        variar=variar,
        path=path,
        star=star,
        passo=passo,
        number=number,
        ordem_parametros=ordem_parametros,
        model_lim=model_lim,
        fwhm=fwhm,
        Dir_OBS=Dir_OBS,
        model_spectra=model_spectra,
        wstart=wstart,
        wend=wend
    )
    # =====================================================
    # 3) CARREGA MODELOS + INTERPOLAÇÃO + CHI2
    # =====================================================

    lam_min, lam_max = model_lim

    FEXP, CHI2, MinChi2, MinChi2_values = carregar_modelos_e_chi2(
        valores_usados=valores_usados,
        pasta_destino=pasta_destino,
        OBS=OBS,
        scale=scale,
        windows=windows,
        lam_min=lam_min,
        lam_max=lam_max,
        fwhm=fwhm
    )

    # =====================================================
    # 4) BAYESIAN
    # =====================================================

    bayes_output = executar_bayes(
        OBS=OBS,
        Dir_OBS2=Dir_OBS2,
        FEXP=FEXP,
        valores_usados=valores_usados,
        windows=windows,
        dl=dl,
        ordem_parametros=ordem_parametros,
        passo=passo,
        priors=priors,
        variar=variar
    )
    
    posterior = bayes_output["posterior"]
    resultados = bayes_output["results"]

    MinChi2Model = bayes_output["MinChi2Model"]
    MinChi2Model_values = bayes_output["MinChi2Model_values"]
    
    i_map = np.argmax(posterior)

    Best_model_MAP = bayes_output["Best_model_MAP"]
    Best_values_MAP = bayes_output["Best_values_MAP"]
    
    i_min2 = bayes_output["i_min2"]

    # =====================================================
    # 5) PRINT FINAL
    # ============================FEXP=========================  
    
    print("Os parâmetros do modelo de mínimo Chi2 são:")

    print(
        "Teff {} | Logg {:.2f} | Vsini {} | Vturb {} | Vmac {} | C {} | N {} | O {} | Si {}".format(
            MinChi2Model_values['teff'],
            MinChi2Model_values['logg'],
            MinChi2Model_values['vsini'],
            MinChi2Model_values['vturb'],
            MinChi2Model_values['vmac'],
            MinChi2Model_values['c'],
            MinChi2Model_values['n'],
            MinChi2Model_values['o'],
            MinChi2Model_values['si']
        )
    )
    
    # =====================================================
    # 6) PLOT
    # =====================================================
    spectra_plot(plot_2=plot_2,
                only_bestfit=only_bestfit,
                windows=windows,
                Compared=Compared,
                xlim=xlim,
                xlim2=xlim2,
                ylim=ylim,
                ylim2=ylim2,
                deltax=deltax,
                deltax2=deltax2,
                deltaobs=deltaobs,
                deltaobs2=deltaobs2,
                scale=scale,
                scale_obs2=scale_obs2,
                OBS=OBS,
                MinChi2Model=MinChi2Model,
                figsize=figsize,
                savefigure=savefigure,
                par_vals=par_vals,
                CHI2= CHI2,
                FEXP=FEXP,
                Best_model_MAP=Best_model_MAP,
                line_high=line_high,
                chi2_loc=chi2_loc,
                leg_painel=leg_painel,
                plotlegend=plotlegend,
                bbox=bbox,
                ylimchi2=ylimchi2,
                ws=ws,
                hl=hl,
                image_grid=image_grid,
                i_min2=i_min2,
                variar=variar,
                star=star,
                d=d,
                resultados=resultados,
                valores_usados=valores_usados,
                legloc=legloc,
                fwhm=fwhm,
                lwbf=lwbf,
                i_map=i_map)
                
    res = {'teff':resultados['teff']['median'],'logg':resultados['logg']['median'],'vsini':resultados['vsini']['median'],
           'vturb':resultados['vturb']['median'],'vmac':resultados['vmac']['median'],'c':resultados['c']['median'],
           'n':resultados['n']['median'],'o':resultados['o']['median'],'si':resultados['si']['median']}            
    
    # =====================================================
    # 7) RETORNO FINAL
    # =====================================================
    
    return Best_values_MAP, Best_values_MAP, FEXP, par_vals, valores_usados,CHI2,ordem_parametros, resultados, passo, variar, res,posterior, i_map


