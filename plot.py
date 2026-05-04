import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.legend_handler import HandlerBase
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator) 
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from collections import OrderedDict
import matplotlib.patches as mpatches
from windows import plot_windows
from ferramentas import minor_major_Ticks

class HandlerColormap(HandlerBase):
    def __init__(self, cmap, n=8, **kwargs):
        self.cmap = cmap
        self.n = n
        super().__init__(**kwargs)

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height,
                       fontsize, trans):
        artists = []
        for i in range(self.n):
            fc = self.cmap(i / (self.n - 1))
            r = mpatches.Rectangle(
                (xdescent + i * width / self.n, ydescent),
                width / self.n,
                height,
                transform=trans,
                facecolor=fc,
                edgecolor='none'
            )
            artists.append(r)
        return artists


def inferno_cut(x):
    return cm.inferno(0.08 + 0.80 * x)
    

        
def plottextlines(ax, line_high, d):
    """
    ax    : eixo matplotlib
    line_high : float
    d     : distância mínima em x entre rótulos
    """
    lines = {# H and He
    '4101.7':[4101.7,line_high,'HI'], '6562.8':[6562.8,line_high,'HI'],   '4120.9':[4120.9,line_high,'HeI'],
    #Si
    '4116.1':[4116.1,line_high,'SiIV'],'4128.05':[4128.0,line_high,'SiII'],'4130.8':[4130.8,line_high,'SiII'],'4552.4':[4552.4,line_high,'SiIII'],'4567.5':[4567.5,line_high,'SiIII'],'4574.5':[4574.5,line_high,'SiIII'],
    '6347.1':[6347.11,line_high,'SiII'],'6371.37':[6371.37,line_high,'SiII'],'4716.5':[4716.5,line_high,'SiIII'],
    # O
    '4132.8':[4132.8,line_high,'OII'],'4602.0': [4601.7,line_high,'OII'],'4609.3': [4609.3,line_high,'OII'],'4610.1': [4610.1,line_high,'OII'], '4641.8':[4641.6,line_high,'OII'],
    '4638.8 ':[4638.8 ,line_high,'OII'],'4649.1':[4649.1,line_high,'OII'],'4650.8':[4650.83,line_high,'OII'],'4590.9':[4590.974,line_high,'OII'],'4596.1':[ 4596.0,line_high,'OII'],
    '4661.6':[4661.6,line_high,'OII'],'4673.7':[4673.7,line_high,'OII'],'4676.2':[4676.2,line_high,'OII'],'4699.0':[4699.0,line_high,'OII'],'4701.2':[4701.2,line_high,'OII'],'4703.1':[4703.1,line_high,'OII'],'4705.3 ':[4705.3 ,line_high,'OII'],'4710.0 ':[4710.0 ,line_high,'OII'],
    # N
    '4601.5':[4601.2,line_high,'NII'],'4607.16':[4607.16,line_high,'NII'],'4613.9':[4613.9,line_high,'NII'],'4621.4':[4621.4,line_high,'NII'],'4630.54':[4630.54,line_high,'NII'], '4643.1':[4642.8,line_high,'NII'],
    #C
    '4638.9':[4638.9,line_high,'CII'],'4647.4':[4647.4,line_high,'CIII'],'4651.5':[4651.5,line_high,'CIII'],'4618.6':[4618.5,line_high,'CII'],'4619.3':[4619.3,line_high,'CII'],'6578.1':[6578.1,line_high,'CII'],'6582.9':[6582.9,line_high,'CII']}  

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # 1) filtra apenas linhas que estão dentro do eixo
    visible = []
    for key, (x, y, texto) in lines.items():
        if xmin <= x <= xmax and ymin <= y <= ymax:
            visible.append((key, x, y, texto))

    # 2) ordena por x
    visible.sort(key=lambda t: t[1])

    # 3) desenha apenas se estiver suficientemente longe da anterior
    last_x = None

    for key, x, y, texto in visible:
        if last_x is None or abs(x - last_x) >= d:
            ax.text(x, y,f"{texto} {key}",rotation="vertical",fontsize=10.5,ha="center",va="bottom",
            clip_on=True)
            last_x = x
            ax.vlines(last_x, y - 0.01, y - 0.005,colors='k', linestyles='-.')    
    


def spectra_plot(plot_2=None,
                only_bestfit=None,
                windows=None,
                xlim=None,
                xlim2=None,
                ylim=None,
                ylim2=None,
                deltax=None,
                deltax2=None,
                deltaobs=None,
                deltaobs2=None,
                scale=None,
                scale_obs2=None,
                Compared=None,
                lwbf=None,
                figsize=None,
                line_high=None,
                alpha=None,
                OBS=None,
                valores_usados=None,
                par_vals=None,
                CHI2=None,
                FEXP=None,
                MinChi2Model=None,
                Best_model_MAP=None,
                savefigure=None,
                chi2_loc=None,
                leg_painel=None,
                plotlegend=None,
                bbox=None,
                ylimchi2=None,
                ws=None,
                hl=None,
                image_grid=None,
                i_min2=None,
                variar=None,
                star=None,
                d=None,
                resultados=None,
                legloc=None,
                fwhm=None,
                i_map=None):
                

    
    if not isinstance(legloc,str):
        legloc = 'upper left'
#---------------------------------------   
# PLOT DE ESPECTROS OBS E SINTÉTICOS
#--------------------------------------
    if not plot_2:
        fig, ax = plt.subplots(figsize=figsize)
        
    
        cores = cm.inferno(np.linspace(0.08, 0.88, len(valores_usados))) # você pode mudar à vontade
        i_min = np.argmin(CHI2)
#-----------------------------------------------------------------------------------------------------------------------------------------------   
        formatos = {'teff': '{:.0f}', 'logg': '{:.2f}', 'vsini': '{:.0f}','vturb': '{:.0f}', 'vmac': '{:.0f}', 'c': '{:.2f}', 'n': '{:.2f}', 'o': '{:.2f}', 'si': '{:.2f}'}
        unidades = {'teff': ' K',     'logg': '',       'vsini': ' km/s','vturb': ' km/s','vmac': ' km/s','c': '','n': '','o': '','si': ''}

        parametros = list(par_vals.keys())   # parâmetro que está variando
        
        
        for i,df in enumerate(FEXP):
            
            if not only_bestfit:
                ax.plot(df['WaveLength'] + deltax,df['NormFluxo'],color = cores[i],linewidth = 2,zorder=1,alpha=alpha,rasterized=True)
                
                ax.plot(Best_model_MAP['WaveLength'] + deltax,Best_model_MAP['NormFluxo'],color='dimgray',linestyle='-.',linewidth =lwbf,label='Maximum A Posteriori ',zorder=2)
                
                ax.plot(MinChi2Model['WaveLength'] + deltax,MinChi2Model['NormFluxo'],color='royalblue',linestyle=':',linewidth=lwbf,label=r'$\mathrm{Minimum \ \  \chi^{2} \ \ model}$',zorder=3)
               
                ax.plot(OBS['WaveLength']+deltaobs,OBS['NormFluxo'],'k',linewidth =2,label=Compared,zorder=0)
                
            elif only_bestfit == 'two':
                ax.plot(OBS['WaveLength']+deltaobs,OBS['NormFluxo'],'k',linewidth =2,label=Compared,zorder=0)
        
                ax.plot(Best_model_MAP['WaveLength'] + deltax,Best_model_MAP['NormFluxo'],color='dimgray',linestyle='-.',linewidth
                =lwbf,label='Maximum A Posteriori',zorder=1)
            
                ax.plot(MinChi2Model['WaveLength'] + deltax,MinChi2Model['NormFluxo'],color='royalblue',linestyle=':',linewidth=lwbf,label=r'$\mathrm{Minimum \ \  \chi^{2} \ \ model}$',zorder=2)
            elif only_bestfit == 'MAP':
                ax.plot(OBS['WaveLength']+deltaobs,OBS['NormFluxo'],'k',linewidth =2,label=Compared,zorder=0)
        
                ax.plot(Best_model_MAP['WaveLength'] + deltax,Best_model_MAP['NormFluxo'],color='dimgray',linestyle='-.',linewidth
                =lwbf,label='Maximum A Posteriori',zorder=1)
            
            elif only_bestfit == 'MinChi2':
                ax.plot(OBS['WaveLength']+deltaobs,OBS['NormFluxo'],'k',linewidth =2,label=Compared,zorder=0)
                ax.plot(MinChi2Model['WaveLength'] + deltax,MinChi2Model['NormFluxo'],color='royalblue',linestyle=':',linewidth=lwbf,label=r'$\mathrm{Minimum \ \  \chi^{2} \ \ model}$',zorder=1)
            elif only_bestfit == 'No_bf':
                ax.plot(OBS['WaveLength']+deltaobs,OBS['NormFluxo'],'k',linewidth =2,label=Compared,zorder=0)
                ax.plot(df['WaveLength'] + deltax,df['NormFluxo'],color = cores[i],linewidth = 2,zorder=1,alpha=alpha)
            else:    
                ax.plot(OBS['WaveLength']+deltaobs,OBS['NormFluxo'],'k',linewidth =2,zorder=0)
                break
            

        plot_windows(windows,line_high-0.015)  ## veja no módulo windows na pasta plottools
        minor_major_Ticks(ax,labelright=False)
#-----------------------------------------------------------------------------------------------------------------------------------------------   
#PLOT INTERNO DO CHI2 -----------------------------------------------------------------------------------------------------------
        labels_latex = {
        'teff': r'$T_{\mathrm{eff}}\ \mathrm{(K)}$',
        'logg': r'$\log\,g\ \mathrm{(cm\,s^{-2})}$',
        'vsini': r'$v\sin i\ \mathrm{(km\,s^{-1})}$',
        'vturb': r'$v_{\mathrm{turb}}\ \mathrm{(km\,s^{-1})}$',
        'vmac': r'$v_{\mathrm{mac}}\ \mathrm{(km\,s^{-1})}$',
        'c': r'$\mathrm{[C/H]}$',
        'n': r'$\mathrm{[N/H]}$',
        'o': r'$\mathrm{[O/H]}$',
        'si': r'$\mathrm{[Si/H]}$'}
        
        
        if len(variar) ==1:
            ax_interno = fig.add_axes(chi2_loc)
    
            ax_interno.plot(par_vals[variar[0]],CHI2,'ko:',label =variar[0])
           
            ax_interno.plot(par_vals[variar[0]][i_min2],CHI2[i_min2],marker='x',color='gray',markersize=12,label=r'min $\chi^2$')                        

            ax_interno.axvline(par_vals[variar[0]][i_map],color='grey',linestyle='-.',linewidth=4,label='MAP')
            
            ax_interno.axvline(par_vals[variar[0]][i_min2], color='royalblue',linewidth=4, linestyle=':')
            
            for i,k in enumerate(par_vals[variar[0]]):
                ax_interno.axvline(par_vals[variar[0]][i], color=cores[i], linestyle='-',alpha=alpha)
                
            minor_major_Ticks(ax_interno,labelright=False)

            ax_interno.set_ylabel(r'$ \chi^2 \ \ $',fontsize=22,color='k')
            ax_interno.set_xlabel(labels_latex.get(variar[0], variar[0]),fontsize=20,color='k')

            ax_interno.set_ylim(ylimchi2[0], ylimchi2[1]*max(CHI2))
 
#--------------------------------------------------------------------------------

            #colocando as legendas para diferentes espectros
            #ax.text(xmintext,1.05,StarModel,fontsize=13)

        # Colocando nomes dos eixos
        ax.set_ylabel(r'$ Normalized  \ \ Flux \ \ $',fontsize=22,color='k')
        ax.set_xlabel(r'$ WaveLength \ \ (\AA) $',fontsize=22,color='k')

    #------------------------------
        ax.set_xlim(xlim)                                  # ajuste dos limites do gráfico em x e y
        ax.set_ylim(ylim)
        if plot_2:
            ax.set_ylim(ylim2)
    #------------------------------
        plottextlines(ax, line_high,d)
      
#######################################################################################################################

# Parte para fazer o plot de dois paineis com diferentes regiões com os mesmos modelos 
    else:
        fig = plt.figure(figsize=figsize)
        g1=();g2=()
        if image_grid == 'vertical':
           g1 = (2,1,1)           
           g2 = (2,1,2)       
        else:
           g1 = (1,2,1) 
           g2 = (1,2,2)      
        
               
        ax1 = fig.add_subplot(g1[0],g1[1],g1[2])
        plot_windows(windows,line_high-0.015)  ## veja no módulo windows na pasta plottools
        minor_major_Ticks(ax1,labelright=False,labelsize=10)
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        
        ax2 = fig.add_subplot(g2[0],g2[1],g2[2])
        plot_windows(windows,line_high-0.015)
        if image_grid == 'vertical':
            minor_major_Ticks(ax2,labelright=False)
        else:
            minor_major_Ticks(ax2,labelright=False,labelleft=False,labelsize=10)
        
        fig.subplots_adjust(wspace=ws)
        if xlim2:
            ax2.set_xlim(xlim2)  
            ax2.set_ylim(ylim2)

 # ----------------------------------------------------------------------   
        formatos = {'teff': '{:.0f}', 'logg': '{:.2f}', 'vsini': '{:.0f}','vturb': '{:.0f}', 'vmac': '{:.0f}', 'c': '{:.2f}', 'n': '{:.2f}', 'o': '{:.2f}', 'si': '{:.2f}'}
        unidades = {'teff': ' K', 'logg': '',       'vsini': ' km/s','vturb': ' km/s','vmac': ' km/s','c': '','n': '','o': '','si': ''}

        parametros = list(par_vals.keys())   # parâmetro que está variando
     
        for i, df in enumerate(FEXP):
            cores = cm.inferno(np.linspace(0.08, 0.88, len(valores_usados))) # você pode mudar à vontade
            if not only_bestfit:
                ax1.plot(df['WaveLength'] + deltax,df['NormFluxo'],color = cores[i],linewidth = 2,zorder=1,alpha=alpha)
                ax1.plot(OBS['WaveLength']+deltaobs,OBS['NormFluxo'],'k',linewidth =2,label=Compared,zorder=0)
                ax1.plot(MinChi2Model['WaveLength'] + deltax,MinChi2Model['NormFluxo'],color='royalblue',linestyle=':',linewidth =lwbf,label=r'$\mathrm{Minimum \ \  \chi^{2} \ \ model}$',zorder=3)                
                
                ax1.plot(Best_model_MAP ['WaveLength'] + deltax,Best_model_MAP['NormFluxo'],color='dimgrey',linestyle='-.',label='Maximum A Posteriori',linewidth =lwbf,zorder=2)
                
                ax2.plot(df['WaveLength'] + deltax2,df['NormFluxo'],color = cores[i],linewidth = 2,zorder=1,alpha=alpha)
                ax2.plot(OBS['WaveLength']+deltaobs2,OBS['NormFluxo']*scale_obs2,'k',linewidth =2,label=Compared,zorder=0)
                ax2.plot(MinChi2Model['WaveLength'] + deltax2,MinChi2Model['NormFluxo'],color='royalblue',linestyle=':',linewidth =lwbf,label=r'$\mathrm{Minimum \ \  \chi^{2} \ \ model}$',zorder=3)                
               
                ax2.plot(Best_model_MAP['WaveLength'] + deltax2,Best_model_MAP['NormFluxo'],color='dimgrey',linestyle='-.',label=r'$\mathrm{Minimum \ \  \chi^{2} \ \ model}$',linewidth =lwbf,zorder=2)
            elif only_bestfit == 'two':
                ax1.plot(OBS['WaveLength']+deltaobs,OBS['NormFluxo'],'k',linewidth =2,label=Compared,zorder=0)
                ax1.plot(MinChi2Model['WaveLength'] + deltax,MinChi2Model['NormFluxo'],color='royalblue',linestyle=':',linewidth =lwbf,label=r'$\mathrm{Minimum \ \  \chi^{2} \ \ model}$',zorder=2)                
                
                ax1.plot(Best_model_MAP ['WaveLength'] + deltax,Best_model_MAP['NormFluxo'],color='dimgrey',linestyle='-.',label='Maximum A Posteriori',linewidth =lwbf,zorder=1)
                
                ax2.plot(OBS['WaveLength']+deltaobs2,OBS['NormFluxo']*scale_obs2,'k',linewidth =2,label=Compared,zorder=0)
                ax2.plot(MinChi2Model['WaveLength'] + deltax2,MinChi2Model['NormFluxo'],color='royalblue',linestyle=':',linewidth =lwbf,label=r'$\mathrm{Minimum \ \  \chi^{2} \ \ model}$',zorder=2)                
                
                ax2.plot(Best_model_MAP['WaveLength'] + deltax2,Best_model_MAP['NormFluxo'],color='dimgrey',linestyle='-.',label=r'$\mathrm{Minimum \ \  \chi^{2} \ \ model}$',linewidth =lwbf,zorder=1)
                
            elif only_bestfit == 'MAP':
                ax1.plot(OBS['WaveLength']+deltaobs,OBS['NormFluxo'],'k',linewidth =2,label=Compared,zorder=0)
                ax1.plot(Best_model_MAP['WaveLength'] + deltax,Best_model_MAP['NormFluxo'],color='dimgray',linestyle='-.',linewidth
                =lwbf,label='Maximum A Posteriori',zorder=2)
                ax2.plot(OBS['WaveLength']+deltaobs2,OBS['NormFluxo']*scale_obs2,'k',linewidth =2,label=Compared,zorder=0)
                ax2.plot(Best_model_MAP['WaveLength'] + deltax2,Best_model_MAP['NormFluxo'],color='dimgray',linestyle='-.',linewidth
                =lwbf,label='Maximum A Posteriori',zorder=2)
           
            elif only_bestfit == 'MinChi2':
                ax1.plot(OBS['WaveLength']+deltaobs,OBS['NormFluxo'],'k',linewidth =2,label=Compared,zorder=0)
                ax1.plot(MinChi2Model['WaveLength'] + deltax,MinChi2Model['NormFluxo'],color='royalblue',linestyle=':',linewidth=lwbf,label=r'$\mathrm{Minimum \ \  \chi^{2} \ \ model}$',zorder=2)
                ax2.plot(OBS['WaveLength']+deltaobs2,OBS['NormFluxo']*scale_obs2,'k',linewidth =2,label=Compared,zorder=0)
                ax2.plot(MinChi2Model['WaveLength'] + deltax2,MinChi2Model['NormFluxo'],color='royalblue',linestyle=':',linewidth=lwbf,label=r'$\mathrm{Minimum \ \  \chi^{2} \ \ model}$',zorder=2)
    
            elif only_bestfit == 'No_bf':
                ax1.plot(OBS['WaveLength']+deltaobs,OBS['NormFluxo'],'k',linewidth =2,label=Compared,zorder=0)
                ax1.plot(df['WaveLength'] + deltax,df['NormFluxo'],color = cores[i],linewidth = 2,zorder=1,alpha=alpha)
                
                ax2.plot(OBS['WaveLength']+deltaobs2,OBS['NormFluxo']*scale_obs2,'k',linewidth =2,zorder=0)
                ax2.plot(df['WaveLength'] + deltax2,df['NormFluxo'],color = cores[i],linewidth = 2,zorder=1,alpha=alpha)
            else:    
                ax1.plot(OBS['WaveLength']+deltaobs,OBS['NormFluxo'],'k',linewidth =2,zorder=0)
                ax2.plot(OBS['WaveLength']+deltaobs2,OBS['NormFluxo']*scale_obs2,'k',linewidth =2,zorder=0)
                break
                
        
#-----------------------------------------------------------------------------------------------------------------------------------------------   
#PLOT INTERNO DO CHI2 -----------------------------------------------------------------------------------------------------------
        labels_latex = {
        'teff': r'$T_{\mathrm{eff}}\ \mathrm{(K)}$',
        'logg': r'$\log\,g\ \mathrm{(cm\,s^{-2})}$',
        'vsini': r'$v\sin i\ \mathrm{(km\,s^{-1})}$',
        'vturb': r'$v_{\mathrm{turb}}\ \mathrm{(km\,s^{-1})}$',
        'vmac': r'$v_{\mathrm{mac}}\ \mathrm{(km\,s^{-1})}$',
        'c': r'$\mathrm{[C/H]}$',
        'n': r'$\mathrm{[N/H]}$',
        'o': r'$\mathrm{[O/H]}$',
        'si': r'$\mathrm{[Si/H]}$'}
        
        if len(variar) ==1:
            ax_interno = fig.add_axes(chi2_loc)
    
            ax_interno.plot(par_vals[variar[0]],CHI2,'ko:',label =variar[0])
    
            ax_interno.plot(par_vals[variar[0]][i_min2],CHI2[i_min2],marker='x',color='gray',markersize=12,label=r'min $\chi^2$')   
        
            ax_interno.axvline(par_vals[variar[0]][i_map],color='grey',linestyle='-.',linewidth=4,label='MAP')
            
            ax_interno.axvline(par_vals[variar[0]][i_min2], color='royalblue',linewidth=4, linestyle=':')
            
            for i,k in enumerate(par_vals[variar[0]]):
                ax_interno.axvline(par_vals[variar[0]][i], color=cores[i], linestyle='-',alpha=alpha)
                
    
            minor_major_Ticks(ax_interno,labelright=False)

            ax_interno.set_ylabel(r'$ \chi^2 \ \ $',fontsize=22,color='k')
            ax_interno.set_xlabel(labels_latex.get(variar[0], variar[0]),fontsize=20,color='k')

            ax_interno.set_ylim(ylimchi2[0], ylimchi2[1]*max(CHI2))
 
#--------------------------------------------------------------------------------        
# Colocando Labels dos eixos 
            
        ax1.set_ylabel(r'$ Normalized \ \ Flux \ \ $',fontsize=22,color='k')
        if image_grid == 'vertical':
            ax2.set_ylabel(r'$ Normalized \ \ Flux \ \ $',fontsize=22,color='k')    
        if image_grid == 'horizontal':
            ax1.set_xlabel(r'$ WaveLength \ \ (\AA) $',fontsize=22,color='k')
        ax2.set_xlabel(r'$ WaveLength \ \ (\AA) $',fontsize=22,color='k')
        plottextlines(ax1, line_high,d)
        plottextlines(ax2, line_high,d)

#--------------------------------------------------------------------------------------------------------------------
# LEGENDAS
#--------------------------------------------------------------------------------------------------------------------
    if plotlegend:

        modelos_handle = 'modelos'

        if not plot_2:
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            
            # acrescenta a entrada de modelos (gradiente) se a flag permitir.
            if not only_bestfit:                
                by_label['Synthetic Model Variations'] = modelos_handle 

            ax.legend(by_label.values(), by_label.keys(),handler_map={modelos_handle: HandlerColormap(inferno_cut)}, loc=legloc,bbox_to_anchor=bbox,prop={'weight': 'normal'},handlelength=hl,handleheight=1.5,labelspacing=0.6,fontsize=24, frameon=False)
        
        else:
            if not leg_painel:
                handles, labels = ax1.get_legend_handles_labels()
                by_label = OrderedDict(zip(labels, handles))
                
                if not only_bestfit:  
                    by_label['Synthetic Model Variations'] = modelos_handle

                ax1.legend( by_label.values(),by_label.keys(),handler_map={modelos_handle: HandlerColormap(inferno_cut)},loc=legloc, bbox_to_anchor=bbox,prop={'weight':            'normal'},handlelength=hl,handleheight=1.5,labelspacing=0.6,fontsize=24, frameon=False)
           
            else:
                handles, labels = ax1.get_legend_handles_labels()
                by_label = OrderedDict(zip(labels, handles))

                by_label['Synthetic Model Variations'] = modelos_handle

                ax2.legend( by_label.values(),by_label.keys(),handler_map={modelos_handle: HandlerColormap(inferno_cut)},loc=legloc, bbox_to_anchor=bbox,prop={'weight':            'normal'},handlelength=hl,handleheight=1.5,labelspacing=0.6,fontsize=24, frameon=False)
            
      
############################################################################################################################################################################
############################################################################################################################################################################ 
########################################################################################################################################################################   
    res = {'teff':resultados['teff']['median'],'logg':resultados['logg']['median'],'vsini':resultados['vsini']['median'],
           'vturb':resultados['vturb']['median'],'vmac':resultados['vmac']['median'],'c':resultados['c']['median'],
           'n':resultados['n']['median'],'o':resultados['o']['median'],'si':resultados['si']['median']}            
       
    if not plot_2:
        lam_min, lam_max = xlim
    else:
        lam_min = xlim[0] ; lam_max = xlim2[1]
    nome = ( f"teff_{res['teff']:.0f}_"
    f"logg_{res['logg']:.2f}_"
    f"vsini_{res['vsini']:.0f}_"
    f"vturb_{res['vturb']:.0f}_"
    f"vmac_{res['vmac']:.0f}_"
    f"C_{res['c']:.2f}_"
    f"N_{res['n']:.2f}_"
    f"O_{res['o']:.2f}_"
    f"Si_{res['si']:.2f}_lim_{lam_min:.0f}_{lam_max:.0f}_fwhm_{fwhm:.2f}")
    ######################################################################################################################### ########################################################################################################################################################################         
    print('Os parâmetros variados e mais prováveis pela inferência Bayesiana são:')
    
    for par in variar:
        stats = resultados[par]
        if par == 'logg':
            print(
            rf"{par.upper():<6} "
            rf"{stats['median']:.2f} ± {stats['plus']:.2f} | "
            rf"p16 {stats['p16']:.2f}  p84 {stats['p84']:.2f}")
        elif par == 'teff':
            print(
            rf"{par.upper():<6} "
            rf"{stats['median']:.0f} ± {stats['plus']:.0f} | "
            rf"p16 {stats['p16']:.0f}  p84 {stats['p84']:.0f}")
        else:
            print(
            rf"{par.upper():<6} "
            rf"{stats['median']:.4f} ± {stats['plus']:.4f} | "
            rf"p16 {stats['p16']:.4f}  p84 {stats['p84']:.4f}")
            
    if savefigure:
        plt.savefig("/home/felipe/Documentos/MESTRADO/PROJETO/sintese/Figures/star_{}_{} .jpg".format(str(star),nome),format="jpg",dpi=300,bbox_inches='tight',pad_inches=0)
   
   
