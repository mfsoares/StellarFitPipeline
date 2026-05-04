# -*- coding: utf-8 -*-
"""
@author: mfss
"""
#Algumas bibliotecas e módulos que ajudam no plot.
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import cv2
from scipy.stats import norm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator) 
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u 
from astroquery.gaia import Gaia
import time
import tempfile
import re 

def minor_major_Ticks(ax,labelright=None,labelleft=None,labelbottom=None,labelsize=None):
    # Esta função ajusta os Eixos dos gráficos para terem separações menores e maiores e o gráfico ficar mais legível.
    # ax = plt.subplots()
    # label
    if not isinstance(labelright,bool):
        labelright = True
    if not isinstance(labelleft,bool):
        labelleft = True
    if not isinstance(labelbottom,bool):
        labelbottom = True
    ## Ticks minor
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(axis="both", direction="in", which='minor', length=5, width=0.3, color="k",right=True,top=True,labelsize=labelsize)

    ##Tick major
    ax.tick_params(axis="both", direction="in", which='major', length=12, width=0.3, color="k",labelright=labelright,labelleft=labelleft,labelbottom=labelbottom,right=True,top=True,labelsize=labelsize)  

def janeladata(windows,df):
  # Esta função retorna um intervalo desejado dentro de uma série de dados de dados.
  Dfs = []
  for i in range(int(len(windows)/2)):
    Df = df[(df.WaveLength >= windows[2*i]) & (df.WaveLength <= windows[2*i+1])]  
    Dfs.append(Df)
    janela = pd.DataFrame(pd.concat(Dfs))
  return janela

def best_scale(obs_flux, mod_flux):
    return np.sum(obs_flux * mod_flux) / np.sum(mod_flux**2)
  
  
def chi2_scaled(windows, dfobs, dfexp):
    obs = janeladata(windows, dfobs)
    mod = janeladata(windows, dfexp)

    # grid comum
    wave = obs['WaveLength'].values
    obs_flux = obs['NormFluxo'].values

    mod_flux_interp = np.interp(
        wave,
        mod['WaveLength'].values,
        mod['NormFluxo'].values
    )

    mask = np.isfinite(obs_flux) & np.isfinite(mod_flux_interp)
    obs_flux = obs_flux[mask]
    mod_flux = mod_flux_interp[mask]

    a = np.sum(obs_flux * mod_flux) / np.sum(mod_flux**2)
    mod_scaled = a * mod_flux

    chi2 = np.sum((obs_flux - mod_scaled)**2)
    return chi2
    
def chi2(windows,dfobs,dfexp):
  #Esta função calcula o chisquare no intervalo desejado e retorna este mesmo.
  #windows: intervalo desejado 
  #dfobs: dataframe dos dados observacionais
    obs = janeladata(windows,dfobs) 
    exp = janeladata(windows,dfexp)
    Chi = ((((obs - exp))**2)).dropna() 
    Chi2 = round(Chi.sum(),5)
    return Chi2.iloc[1] 

 
def minimo_Chi2(Chi2):
  #esta função retorna o mínimo ponto Chi2 em uma serie de dados Chi2
    Chi2min = min(i for j in Chi2 for i in j)
    
    return Chi2min

def best_value(lista,Chi2):
  #Esta função retorna o melhor valor da lista baseado no menor Chi2 e retorna o índice do menor Chi2
    Chi2 = np.asarray(Chi2)
    Best_idx = np.argmin(Chi2)
    Best_value = lista[Best_idx]
    return Best_value,Best_idx
  
def hmstodegree(coluna_ra,coluna_dec):
    # Transforma colunas RA e DEC que estão escritas neste formato 11343713	-6135132 para este 173.654708 -61.587000
    # INPUT dataframe RA DEC hms 
    # OUTPUT dataframe RA DEC Degree
    List1 = []; List2 = []; 
    for i in range(len(coluna_ra)):
        D1 = 15*int(str(coluna_ra[i])[:2]); 
        D2 = 15*int(str(coluna_ra[i])[2:4])/60; 
        D3 = 15*int(str(coluna_ra[i])[4:6])/3600; 
        D4 = 15*int(str(coluna_ra[i])[6:8])/3600/100; 
        Degree1 = D1 + D2 + D3 + D4
        List1.append(Degree1)
        
    # para melhorar ainda preciso ver o if do sinal  
        D5 = int(str(coluna_dec[i])[1:3]); 
        D6 = int(str(coluna_dec[i])[3:5])/60; 
        D7 = int(str(coluna_dec[i])[5:8])/10/3600; 
        Degree2 = D5 + D6 + D7
        
        sinal = str(coluna_dec[i])[0]
        if sinal == '-':
            Degree2 = Degree2*(-1)
                
        List2.append(Degree2)
        
    
    dataframe = pd.DataFrame(list(zip(List1, List2)),
               columns =['RA', 'DEC'])      
    return dataframe
    
    
def AjusteLinear(Eixox,Eixoy):

  x = Eixox  
  y = Eixoy
 
  n=len(x)  #Número de elementos
  mq = np.sum(x**2)/n
  mx = x.mean() # Média x
  my = y.mean() # Média y
  vx = x.var()  # Variânça x
  vy = y.var()  # Variânça y
  dx = np.sqrt(vx)      # Desvio padrão x
  dy = np.sqrt(vy)      # Desvio padrão y
  cxy= Eixox.cov(Eixoy)   # covariânça
  a = cxy/(dx**2)         # SLOPE 
  b = my -a*mx           # b
  s= n/(n-2)          
  ey = np.sqrt(s*(dy**2-(cxy**2)/(dx**2)))        # incerteza em eixo y
  ex= ey/a              #  incerteza em eixo x
  r=a*dx/dy            # pearson
  ea = ey/(dx*np.sqrt(n))    # incerteza do SLOPE
  eb = ea*np.sqrt(mq)        # incerteza do b

  return  a,b,r,ex,ey,ea,eb
  
  
def loadtxt(filename,rows):

  data= np.loadtxt(filename,dtype=float,skiprows=rows)

  fig, ax=plt.subplots(figsize=(10,5))
  ax.plot(data[:,0],data[:,1])


def printphoto(Path_Figuras,Figura_name,high,large,rows,cols,Title):
    Path_Figura = Path_Figuras + Figura_name 
    # create figure
    fig = plt.figure(figsize=(high, large))
    # setting values to rows and column variables
    rows = 1; cols = 1
    # reading images
    Image1 = cv2.imread(Path_Figura)
    # Adds a subplot at the 1st position
    ax1 = fig.add_subplot(rows, cols, 1)
    # showing image
    ax1.imshow(Image1) ; ax1.axis('off'); ax1.set_title(Title)
    #plt.legend(fontsize=17,frameon=False,handlelength=0,handletextpad=0)
    plt.tight_layout(w_pad=-25)
    plt.show()

   
def loop_errorbar(ax, yourResult, literature,index,yourResult_erro,literature_erro,marker,color,label,authors,markersize=None):
    # plot points of a selected reference that you defined
    # ax = axis
    # index is the index of yourResult
    # label is the label you want to select
    # authors is a column with more than one author, 
    # but in the case of only one author , you just put authors = None and put the label = 'reference'  
    
    global handle
    try:
        for i , Ref in zip(index,authors.dropna()):
    	    if Ref == label:
    	      handle = ax.errorbar(yourResult[i],literature[i],yerr=literature_erro[i],xerr=yourResult_erro[i],
    	      fmt=marker,color=color,label=label,ecolor='k',markeredgecolor='k',markeredgewidth=0.3,
    	      elinewidth=0.3,capsize=5,capthick=0.3,markersize=markersize)
    	    else:
    	    	 pass
    except AttributeError: 
        for i in index:   # In the case that authors is None and is only one reference.
    	      handle = ax.errorbar(yourResult[i],literature[i],yerr=literature_erro[i],xerr=yourResult_erro[i],
    	      fmt=marker,color=color,label=label,ecolor='k',markeredgecolor='k', markeredgewidth=0.3,elinewidth=0.3,capsize=5,capthick=0.3,markersize=markersize)	
    return handle
            
##################################################################################

def mean_distance_literature(ax,yourResult,literature,index,label,authors,a=None,b=None,xi=None,xf=None,num=None):
    # but in the case of only one author , you just put authors = None and put the label = 'reference'    
    dist = [];Result_slice=[]; 
    try:
        for i, Ref in zip(index,authors):
            Dist_Result = literature[i] - yourResult[i]
            if Ref == label and np.isnan(Dist_Result) == False:        
              dist.append(Dist_Result)
              Result_slice.append(yourResult[i])
              Mean_dist = np.mean(dist)                                  # Mean
              Desvio_medio = np.sum((literature-yourResult))/len(yourResult)
    except TypeError:
        for i in (index):	            
            Dist_Result = literature[i] - yourResult[i]
            if np.isnan(Dist_Result) == False:
              dist.append(Dist_Result)
              Result_slice.append(yourResult[i])            
              Mean_dist = np.mean(dist)
              Desvio_medio = np.sum((literature-yourResult))/len(yourResult)
    except KeyError:
        for i in (index):	            
            Dist_Result = literature[i] - yourResult[i]
            if np.isnan(Dist_Result) == False:
              dist.append(Dist_Result)
              Result_slice.append(yourResult[i])            
              Mean_dist = np.mean(dist)
              Desvio_medio = np.sum((literature-yourResult))/len(yourResult)                

    # Função  f(t) = at + b 
    a=a;b=b; xi=xi; xf=xf; t = np.linspace(xi,xf,num); f = a*t+b; b1=Mean_dist; f1 = a*t+b1
    
 
    df = pd.DataFrame({'a':[a],'b':[b],'xi':[xi],'xf':[xf],'t':[t],'f':[f],'f1':[f1],'dist':[dist],'Mean_dist':[Mean_dist],
    'Result_slice':[Result_slice]})
    
    return df
######################################################################################################

def plot_straight_curve(ax,t,f,label,ls=None,color=None):
    ax.plot(t, f,linewidth=1,linestyle=ls,color=color,zorder=0,label=label)
    

    
def hist_2(ax,data1,data2,c1,c2,edge,bin_num,width,label_S,decimals,orientation=None,l1=None,l2=None):

    # Set the bin edges to ensure same width bins
    bin_edges = np.linspace(min(min(data1), min(data2)), max(max(data1), max(data2)), bin_num)

    # Plot histogram for data1
    ax.hist(data1, bins=bin_edges, color=c1, alpha=1, label=l1,edgecolor=edge,orientation=orientation)

    # Plot histogram for data2
    ax.hist(data2, bins=bin_edges, color=c2, alpha=1, label=l2,edgecolor=edge,orientation=orientation)

    
    # Set edgecolor to align with ticks
    #plt.gca().spines['bottom'].set_color('black')

    # tick params
        
    ax.tick_params(axis="both",direction="in", width=width, which='major', length=10, color="black",labelsize=label_S,right=True,top=True,bottom=True,)
    

    # Set x-axis tick labels to be rounded to 1 decimal place
    #plt.xticks(np.round(bin_edges, decimals=1))
    
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.'+decimals+'f'))

def hist_3(ax,data1,data2,data3,c1,c2,c3,edge,bin_num,width,label_S,decimals,orientation=None,l1=None,l2=None):

    # Set the bin edges to ensure same width bins
    bin_edges = np.linspace(min(min(data1), min(data2),min(data3)), max(max(data1), max(data2),max(data3)), bin_num)

    # Plot histogram for data1
    ax.hist(data1, bins=bin_edges, color=c1, alpha=1, label=l1,edgecolor=edge,orientation=orientation)

    # Plot histogram for data3
    ax.hist(data3, bins=bin_edges, color=c3, alpha=1, label=l2,edgecolor=edge,orientation=orientation)
    
    # Plot histogram for data2
    ax.hist(data2, bins=bin_edges, color=c2, alpha=1, label=l2,edgecolor=edge,orientation=orientation)
    
    
    
    # Set edgecolor to align with ticks
    #plt.gca().spines['bottom'].set_color('black')

    # tick params
        
    ax.tick_params(axis="both",direction="in", width=width, which='major', length=10,
    color="black",labelsize=label_S,right=True,top=True,bottom=True,)
    

    # Set x-axis tick labels to be rounded to 1 decimal place
    #plt.xticks(np.round(bin_edges, decimals=1))
    
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.'+decimals+'f'))
    
    
  
def gauss_hist(ax,data,bins,lim,n,lw,color_curva,color_hist,alpha,orientation=None):
    if not isinstance(orientation,str):
        orientation = 'vertical'
    data=data
    # Criar o histograma das ocorrências
    counts, bins, ignored = ax.hist(data,bins=bins,alpha=alpha, color=color_hist, edgecolor='black', density=False,orientation=orientation)

    # Ajustar uma curva gaussiana aos dados
    mu, std = norm.fit(data)   
    if orientation == 'vertical':
    	# Calcular a curva gaussiana 
    	x = np.linspace(lim[0],lim[1], n)
    	p = norm.pdf(x, mu, std) * len(data) * (bins[1] - bins[0])

    	# Plotar a curva gaussiana ajustada
    	ax.plot(x, p, c=color_curva, linewidth=lw,zorder=2)
    	ax.vlines(mu,0,counts.max(),colors='k',linestyle="dashed",lw=lw,alpha=alpha)
    else:
    	# Calcular a curva gaussiana
    	y = np.linspace(lim[0],lim[1], n)
    	p = norm.pdf(y, mu, std) * len(data) * (bins[1] - bins[0])

    	# Plotar a curva gaussiana ajustada
    	ax.plot(p, y, c=color_curva, linewidth=lw,zorder=2)    # Consequentemente no plot da gaussiana a pdf p fica no eixo x e o data no y
    	ax.hlines(mu,0,counts.max(),colors='k',linestyle="dashed",lw=lw,alpha=alpha)
    
    return mu, std, counts
    
def plotmembership(CLUSTER,Name,Path_Figures,distance,plx,pmra,pmde,xlim,ylim,radius,mag1,mag2,Mag1,Mag2,Cor1,Cor2,u,
                   band,cor,bins,label_S,fontsize_l,fontsize_L,Thick,figsize,whspace,linhas,colunas,Sphigh,Splow):
    #plx=[plx0,plx1,plx2,dplxl,dplxr]
    #pmra=[pmra1,pmra2,pmra3]
    #pmde=[pmde1,pmde2,pmde3]
    #xlim =[xlim1,xlim2,xlim3]
    #bins=[plx,pm]
    #figsize=[fisize1,figsize2]
    
    
    dplxl=plx[3];dplxr=plx[4]; radius = radius  ; u = u
    mag1=mag1 ;mag2 = mag2;  Cor1 = Cor1 ; Cor2 = Cor2

    # 4755
    xlim1 = xlim[0] ;ylim1 = ylim[0]
    xlim2 = xlim[1] ;ylim2 = ylim[1]
    xlim3 = xlim[2] ;ylim3= ylim[2]

    plx1 =  plx[0]  #:   0.0260     |2021A&A...651A.104P|
    plx2 =  plx[1]  #:   0.046      |2021MNRAS.504..356D|
    plx3 =  plx[2]   #:   0.044      |2020A&A...640A...1C|

    pmra1 = pmra[0];
    #'[    0.118]|'
    pmde1 = pmde[0]    
    #[    0.125]|  #|2021A&A...651A.104P|   
    pmra2 = pmra[1];
    #'[    0.136]|'
    pmde2 = pmde[1]    
    #[    0.137]|    |2021MNRAS.504..356D|
    pmra3 = pmra[2];
    #'[    0.132]|'
    pmde3 = pmde[2]    
#[    0.134]|ICRS|2020A&A...640A...1C|
    
    
    #3293
    mediaPMRA =np.mean([pmra1,pmra2,pmra3])   ### Média 3 PMRA SIMBAD
    mediaPMDE =np.mean([pmde1,pmde2,pmde3])   ### Média 3 PMDE SIMBAD
    mediaPLX = np.mean([plx1,plx2,plx3])      ### Média 3 PLx SIMBAD 
     
    CLUSTER = CLUSTER

    CLUSTER['Vmag'] = CLUSTER['Gmag'] -(-0.02704 + 0.01424*CLUSTER['BP-RP'] -0.2156*CLUSTER['BP-RP']**2 +0.01426*CLUSTER['BP-RP']**3) 
    CLUSTER['Bmag'] = CLUSTER['Gmag'] -(0.01448 -0.6874*CLUSTER['BP-RP'] -0.3604*CLUSTER['BP-RP']**2 +0.06718*CLUSTER['BP-RP']**3-0.006061*CLUSTER['BP-RP']**4) 
    CLUSTER['Imag'] = CLUSTER['Gmag'] -(-0.03298 +1.259*CLUSTER['BP-RP'] -0.1279*CLUSTER['BP-RP']**2 +0.01631*CLUSTER['BP-RP']**3) 
    CLUSTER['B-V'] = CLUSTER['Bmag']-CLUSTER['Vmag']
    CLUSTER['B-I'] = CLUSTER['Bmag']-CLUSTER['Imag']
    CLUSTER['Mv'] = CLUSTER['Vmag'] + 5 - 5 * np.log10(distance) 

    
    
    # CutCor
    CLUSTER_cutcor = CLUSTER[(CLUSTER[cor]>Cor1)& (CLUSTER[cor]<Cor2)]

    # CutMag
    CLUSTER_cutgmag = CLUSTER_cutcor[(CLUSTER_cutcor[band]>mag1)& (CLUSTER_cutcor[band]<mag2)]

    #Corte Erro CLUSTER_cutgmag
    CLUSTER_cutgmag = CLUSTER_cutgmag[CLUSTER_cutgmag['e_Gmag']<=u]
    CLUSTER_cutgmag = CLUSTER_cutgmag[CLUSTER_cutgmag['e_BPmag']<=u]
    CLUSTER_cutgmag = CLUSTER_cutgmag[CLUSTER_cutgmag['e_RPmag']<=u]



    Data = CLUSTER               #
    Data_ = CLUSTER_cutgmag  ; 


    #Corte PM  Teorema de pitágoras
    Data['distance_radius'] = np.sqrt((Data['pmRA'] - mediaPMRA)**2 + (Data['pmDE'] - mediaPMDE)**2)
    Data = Data[Data['distance_radius'] <= radius]

    Data_['distance_radius'] = np.sqrt((Data_['pmRA'] - mediaPMRA)**2 + (Data_['pmDE'] - mediaPMDE)**2)
    Data_= Data_[Data_['distance_radius'] <= radius]


    # CORTE PLX
    Data = Data[(Data['Plx']> mediaPLX-dplxl) & (Data['Plx'] < mediaPLX+dplxr)]     
    Data_= Data_[(Data_['Plx']> mediaPLX-dplxl) & (Data_['Plx'] < mediaPLX+dplxr)]     

    Md = 5 - 5 * np.log10(4000) 
    EBV = 0
    BPRP_d = (1.08337-0.63439)*(3.1*EBV)
    # Correções mag Stars
    BP_RP = Data_['BP-RP']- BPRP_d
    G = Data_['Gmag'] -(0.83627*(3.1*EBV))
    
    Condition = Data[(Data['Mv']>Mag1) & (Data['Mv']<Mag2)]
             
    mag = np.mean(Condition[band])


    mediaplx =Data_['Plx'].mean()
    mediapmra = Data_['pmRA'].mean()
    mediapmde = Data_['pmDE'].mean()



    bins_pm=bins[0];bins_plx=bins[1];
    label_S = label_S; fontsize_l = fontsize_l; fontsize_L = fontsize_L; Thick = Thick; 
    fig = plt.figure(figsize=(figsize[0],figsize[1]))
    fig.subplots_adjust(wspace=whspace[0],hspace=whspace[1]) 

    ####################  Set ax1
    ax1 = fig.add_subplot(linhas,colunas,1)   #

    ax1.scatter(Data['pmRA'],Data['pmDE'],marker='.',s=20,color='dimgrey',alpha=1)
    ax1.scatter(Data_['pmRA'],Data_['pmDE'],marker='.',s=20,color='r',alpha=1)
    ########################################################
    # PROPER MOTION:
    #######################################################################################################
    # create new axes on the right and on the top of the current axes
    divider = make_axes_locatable(ax1)
    # below height and pad are in inches##   Tamanho dos gráficos superiores
    ax1_histx = divider.append_axes("top", 1.5, pad=0.2, sharex=ax1)
    ax1_histy = divider.append_axes("right", 1.5, pad=0.4, sharey=ax1)

    # make some labels invisible
    ax1_histx.xaxis.set_tick_params(labelbottom=False)
    ax1_histy.yaxis.set_tick_params(labelleft=False)

    #HISTOGRAMA and lines

    mediara, stdra, countsra = gauss_hist(ax1_histx,Data_['pmRA'],bins=bins_pm,lim=xlim1,n=100,lw=2,color_curva='k',color_hist='r',alpha=0.4)
    mediade, stdde, countsde = gauss_hist(ax1_histy,Data_['pmDE'],bins=bins_pm,lim=ylim1,n=100,lw=2,color_curva='k',color_hist='r',alpha=0.4,orientation='horizontal')

    ax1.scatter(mediapmra,mediapmde,marker=r'$\odot$',s=10,color='k')

    #ax1.scatter(mediaRA,mediaDE,marker='$\odot$',s=200,color='orange')


    # Circle
    circ1 = plt.Circle((mediapmra, mediapmde), radius, lw=2,color='b', fill=False)
    #circ2 = plt.Circle((mediaPMRA, mediaPMDE), 1, color='r', fill=False)
    #ax1.add_patch(circ1)
    #ax1.add_patch(circ2)

    # Linha vertical Ref hist

    ax1_histx.vlines(mediaPMRA,0,countsra.max(),colors='r',linestyle="dashed",lw=4)
    ax1_histy.hlines(mediaPMDE,0,countsde.max(),colors='r',linestyle="dashed",lw=4)
    


    #######################################################################################################
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    # ax1 Parametros dos Major ticks
    ax1.tick_params(axis="both", direction="in", length=10, width=Thick, color="k",right=True,top=True,labelsize=label_S)
    ax1_histx.tick_params(axis="both", direction="in", length=10, width=Thick, color="k",right=True,top=True,labelsize=label_S)
    ax1_histy.tick_params(axis="both", direction="in", length=10, width=Thick, color="k",right=True,top=True,labelsize=label_S)
    ax1.set_xlim(xlim1)
    ax1.set_ylim(ylim1)
    ax1.set_xlabel(r'$\mu_{\alpha}\cos \delta \ \ [m \ as \ yr^{-1}]$',fontsize=fontsize_L,color='k')
    ax1.set_ylabel(r'$\mu_{\delta}  \ \ [m \ as \ yr^{-1}]$',fontsize=fontsize_L,color='k')

    ##########################################################
    ########################################################## Set ax2
    # GMAG PARALLAX 
    ax2 = fig.add_subplot(linhas,colunas,2)

    ax2.scatter(Data['Plx'],Data[band],marker='.',s=30,color='dimgrey',alpha=1)
    ax2.scatter(Data_['Plx'],Data_[band],marker='.',s=30,color='r',alpha=1)

    # create new axes on the right and on the top of the current axes
    divider2 = make_axes_locatable(ax2)
    # below height and pad are in inches##   Tamanho dos gráficos superiores
    ax2_histx = divider2.append_axes("top", 1.5, pad=0.2, sharex=ax2)
    # make some labels invisible
    ax2_histx.xaxis.set_tick_params(labelbottom=False)

    #HISTOGRAMA

    mediaplx, stdplx, counts = gauss_hist(ax2_histx,Data_['Plx'],bins=bins_plx,lim=xlim2,n=100,lw=2,color_curva='k',color_hist='r',alpha=0.4)

    #ax2.vlines(min(Data_['Plx']),0,75,colors='b',linestyle="dashed",lw=3)
    #ax2.vlines(max(Data_['Plx']),0,75,colors='b',linestyle="dashed",lw=3)

    # Linha vertical Ref hist

    ax2_histx.vlines(mediaPLX,0,counts.max(),colors='r',linestyle="dashed",lw=4)

    #################################################################################################
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.tick_params(axis="both", direction="in",width=Thick, which='major', length=10,color="k",right=True,top=True,labelsize=label_S)
    ax2_histx.tick_params(axis="both", direction="in", length=10, width=Thick, color="k",right=True,top=True,labelsize=label_S)
    ax2.set_xlim(xlim2)
    ax2.set_ylim(ylim2)
    ax2.set_xlabel(r'$\varpi \ \ [m \ as]$',fontsize=fontsize_L,color='k')
    ax2.set_ylabel(r'$ {} [mag] $'.format(band[0]),fontsize=fontsize_L,color='k')

    ax2.invert_yaxis() 
    ############## Set ax3
    #CMD
    ax3 = fig.add_subplot(linhas,colunas,3)

    ax3.scatter(Data[cor],Data[band],marker='.',s=30,color='dimgrey',alpha=1)
    ax3.scatter(Data_[cor],Data_[band],marker='.',s=30,color='r',alpha=1)


    #Label  Coloquei # dia 18 de dezembro de 2025 pq estava dando problema, veja se resolve depois, é bobeira
    #ax3.set_ylabel(r'$'+band[0]+' \ [mag] $',fontsize=fontsize_L,color='k')
    #ax3.set_xlabel(r'$'+cor+' \ \ [mag]$',fontsize=fontsize_L,color='k')


    ax3.set_xlim(xlim3) 
    ax3.set_ylim(ylim3) 

    ax3.invert_yaxis() 

    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    ## Ticks Major
    ax3.tick_params(axis="both", direction="in", width=Thick, which='major', length=10, color="k",right=True,top=True,labelsize=label_S)

    ax3.hlines(mag1,-1,3,colors='k',linestyle="dashed",lw=2,label=Sphigh)
    ax3.hlines(mag2,-1,3,colors='k',linestyle="-",lw=2,label=Splow)

    ax3.legend()
    plt.savefig(Path_Figures + '/'+Name+'.jpg',dpi=500,format='jpg',bbox_inches='tight',pad_inches=0)
    
    
    print('Média Plx:',mediaplx)
    print("Média PLX SIMBAD : ",mediaPLX)
    print('Média PMRA SIMBAD:',mediaPMRA)
    print('Média PMDE SIMBAD:',mediaPMDE)
    print('Média pmra :',mediapmra)
    print('Média pmde :',mediapmde)
    print('Magnitude média da classe spectral escolhida:',mag)
    
   
    return Data, Data_
    plt.show()
    
   

def get_xy_for_z_values(df, z_values):
    """
    Retorna uma lista de pares (x, y) para os valores fornecidos de z.
    
    Se z não existir diretamente, calcula a média de x e y interpolando as linhas vizinhas.

    Parameters:
        df (pd.DataFrame): DataFrame contendo as colunas 'x', 'y' e 'z'.
        z_values (list): Lista de valores de z desejados.

    Returns:
        list: Lista de pares (x, y) correspondentes.
    """
    if not all(col in df.columns for col in ['x', 'y', 'z']):
        raise ValueError("O DataFrame deve conter as colunas 'x', 'y' e 'z'.")

    df = df.sort_values(by='z').reset_index(drop=True)  # Garantir que os dados estão ordenados por z
    results = []

    for z in z_values:
        if z in df['z'].values:
            # Se z está diretamente na coluna, pega os valores correspondentes de x e y
            row = df[df['z'] == z].iloc[0]
            results.append((row['x'], row['y']))
        else:
            # Se z não está diretamente presente, faz interpolação
            lower = df[df['z'] < z].iloc[-1] if not df[df['z'] < z].empty else None
            upper = df[df['z'] > z].iloc[0] if not df[df['z'] > z].empty else None

            if lower is not None and upper is not None:
                # Interpolação linear para x e y
                x_interp = np.mean([lower['x'], upper['x']])
                y_interp = np.mean([lower['y'], upper['y']])
                results.append((x_interp, y_interp))
            else:
                # Caso z esteja fora do intervalo, retorna None
                results.append((None, None))
                

    return results

def plot_points_with_markers(dataframe,ax):
    """
    Plota pontos de um DataFrame com marcadores diferentes baseados no índice da coluna 'x'.

    Parâmetros:
        dataframe (pd.DataFrame): Deve conter as colunas 'x' e 'y'.
    """
    # Lista de marcadores disponíveis
    markers = ['<', 'D', 'p', 'o', 's']
       
    for idx, (x, y) in enumerate(zip(dataframe['[Fe/H]'], dataframe['[O/Fe]'])):
        marker = markers[idx % len(markers)]  # Escolhe marcador com base no índice
        ax.scatter(x, y,color='w',s = 100,edgecolor='k',zorder=5,label='',marker=marker)
        
def calcula_rgc(df,l,b,Distance,R_Sol=8.33):
    """
    Calcula o raio galactocêntrico (Rgc) para um conjunto de objetos.
    
    Parâmetros:
    df : DataFrame contendo as colunas 'Distances' (pc), 'l' (graus) e 'b' (graus).
    R_Sol : Distância do Sol ao centro da Galáxia em kpc (padrão: 8.2 kpc).
    
    Retorna:
    Uma Série pandas com os valores de Rgc em kpc.
    """
    DSun_OC = df[Distance]/ 1000  # Converter para kpc
    
    l_rad = np.radians(df[l])
    b_rad = np.radians(df[b])
    
    RG_OC = np.sqrt(R_Sol**2 + (DSun_OC * np.cos(b_rad))**2 
                     - 2 * R_Sol * DSun_OC * np.cos(l_rad) * np.cos(b_rad))
    
    return RG_OC
    
# -------------------------------------------------------------
#  Estimando o sigma 
#-------------------------------------------------------

def estimate_sigma(flux_obs, model_best, npar, mask=None):
    """
    Estima σ do espectro observado a partir dos resíduos
    """

    flux_obs = np.asarray(flux_obs)
    model_best = np.asarray(model_best)

    if mask is None:
        resid = flux_obs - model_best
        nu = resid.size - npar
    else:
        mask = np.asarray(mask, dtype=bool)
        resid = flux_obs[mask] - model_best[mask]
        nu = mask.sum() - npar

    if nu <= 0:
        raise ValueError("Graus de liberdade <= 0")

    sigma2 = np.sum(resid**2) / nu
    return np.sqrt(sigma2) 

# -------------------------------------------------------------
#  χ² ponderado (físico)
# -------------------------------------------------------------

def chi2_weighted(obs, mod, sigma, mask=None):
    if mask is None:
        return np.sum((((obs - mod).to_numpy()) / sigma)**2)
    return np.sum(((obs[mask] - mod[mask]) / sigma)**2)
      
# -------------------------------------------------------------
#  χ² ponderado para todos os modelos
#--------------------------------------------------------------

def compute_chi2_models(flux_obs,models_df,sigma):
    Chi2w = []
    for model in models_df:
        chi2w = chi2_weighted(flux_obs, model, sigma, mask=None)
        Chi2w.append(chi2w)
    return Chi2w

# -------------------------------------------------------------
# Incertesa via deltaχ² 
#------------------------------------------------------------

def parameter_uncertainty_list(Modelos,windows, param, delta_chi2,DirOBS,npar):

#----------------------------------------------------------------------------------------------------------------------------------
#   Calculando Chi2 com sigma
#----------------------------------------------------------------------------------------------------------------------------------
    models_df=Modelos[2]; models=Modelos[4]
    
    
    
    Dir_OBS = os.path.abspath(DirOBS)
    OBS = pd.read_csv(Dir_OBS, sep=r'\s+', header=0, names=['WaveLength', 'NormFluxo'])
    
    Best_Model_janela = janeladata(windows,Modelos[1])
    OBS_janela = janeladata(windows,OBS)

    modelos = []
    for i,k in enumerate(models_df):
        fxp = janeladata(windows,models_df[i])
        modelos.apchi2pend(fxp)

    sigma = estimate_sigma(OBS_janela,Best_Model_janela,npar)
    
    chi2w = compute_chi2_models(flux_obs=OBS_janela, models_df=modelos,sigma=sigma)
#----------------------------------------------------------------------------------------------------------------------------------    
#----------------------------------------------------------------------------------------------------------------------------------
    

    chi2 = np.asarray(chi2w)

    if len(models) != len(chi2):
        raise ValueError("models e chi2 devem ter o mesmo tamanho")

    values = np.array([m[param] for m in models])

    chi2_min = chi2.min()
    delta = chi2 - chi2_min

    mask = delta <= delta_chi2
    if not np.any(mask):
        raise RuntimeError(f"Nenhum modelo satisfaz Δχ² para {param}")

    idx_best = np.argmin(chi2)
    p_best = values[idx_best]

    p_low = values[mask].min()
    p_high = values[mask].max()

    return {
        'best': p_best,
        'minus': p_best - p_low,
        'plus': p_high - p_best
    }

#### Forma de usar a parameter_uncertainty_list
#results = {}
#for p in params:
#    results[p] = parameter_uncertainty_list(Modelos = Best_Teff_CNO,param=p,delta_chi2=250,DirOBS='../NGC3766/NGC_1_UVL.txt',npar=3)
        
#results

def rebin_spectrum(df, dl=0.05):
    bins = np.arange(df['WaveLength'].min(),
                      df['WaveLength'].max() + dl,
                      dl)

    idx = np.digitize(df['WaveLength'], bins)

    dfb = (
        df.assign(bin=idx)
          .groupby('bin', as_index=False)
          .mean()
    )

    return dfb[['WaveLength', 'NormFluxo']]


# -------------------------------------------------------------
#  Incerteza via inferência bayesiana
#--------------------------------------------------------------

def bayesian_inference_grid(
    OBS_stat,
    modelos_stat,
    valores_usados,
    ordem_parametros,
    passo,
    priors=None,
    dl=1,
    npar=1):

    params = ordem_parametros

    # -------------------------------------------------------
    # Sigma_model — definido a partir do OBS (NÃO do modelo)
    # -------------------------------------------------------
    #sigma0 = np.std(OBS_stat["NormFluxo"])
    flux = OBS_stat["NormFluxo"].values
    sigma0 = 1.4826 * np.median(np.abs(flux - np.median(flux)))

    sigma_model_grid = np.logspace(np.log10(0.3 * sigma0), np.log10(5.0 * sigma0),15)

    prior_sigma = 1.0 / sigma_model_grid
    prior_sigma /= prior_sigma.sum()

    # -------------------------------------------------------
    # Likelihood marginalizado em sigma_model
    # -------------------------------------------------------
    likelihood = np.zeros(len(modelos_stat))
    chi2_eff   = np.zeros(len(modelos_stat))

    for k, sigma_m in enumerate(sigma_model_grid):

        chi2_k = compute_chi2_models( flux_obs=OBS_stat, models_df=modelos_stat, sigma=sigma_m)

        chi2_k = np.asarray(chi2_k)

        chi2_eff += chi2_k * prior_sigma[k]

        logL_k = -0.5 * chi2_k
        logL_k -= logL_k.max()   # estabilidade numérica

        likelihood += np.exp(logL_k) * prior_sigma[k]

    # -------------------------------------------------------
    # Priors nos parâmetros físicos
    # -------------------------------------------------------
    prior_total = np.ones_like(likelihood)

    if priors is not None:
        for p in params:
            values = np.array([m[p] for m in valores_usados])
            if p in priors:
                prior_total *= priors[p]["func"](values)

    # -------------------------------------------------------
    # Volume da célula
    # -------------------------------------------------------
    cell_volume = 1.0
    for p in params:
        cell_volume *= passo[p]

    posterior = likelihood * prior_total * cell_volume

    if posterior.sum() == 0 or not np.isfinite(posterior.sum()):
        raise RuntimeError("Posterior inválida — likelihood zerada.")

    posterior /= posterior.sum()

    # -------------------------------------------------------
    # Marginalizações
    # -------------------------------------------------------
    results = {}

    for p in params:
        values = np.array([m[p] for m in valores_usados])
        unique_vals = np.unique(values)

        post_1d = np.zeros_like(unique_vals, dtype=float)

        for i, v in enumerate(unique_vals):
            post_1d[i] = posterior[values == v].sum()

        post_1d /= post_1d.sum()
        results[p] = summarize_posterior(unique_vals, post_1d)

    return {"posterior": posterior,"results": results,"chi2": chi2_eff }

    
    
##########################################################################################################################   Esta parte se refere aos priors, ainda não está funcionando bem esta metodologia, acho que deixarei apenas como priors = None , pois não vi diferença em usar. 
def gaussian_prior(x,mu, sigma):
        return np.exp(-0.5*((x-mu)/sigma)**2)
        
def gaussian_prior_from_best(best, step, nsigma=1):
    sigma = nsigma * step

    def prior(x):
        return np.exp(-0.5*((x-best)/sigma)**2)

    return {
        "type": "gaussian",
        "mu": best,
        "sigma": sigma,
        "func": prior
    }        
def flat_prior_from_best(best, step, nsigma=1):
    lo = best - nsigma*step
    hi = best + nsigma*step

    def prior(x):
        return np.where((x >= lo) & (x <= hi), 1.0, 0.0)

    return {
        "type": "flat",
        "lo": lo,
        "hi": hi,
        "func": prior
    }
    
def New_priors(Type,pre_fit,nsigma):
    priors = {}
    passo = pre_fit[8]
    params = pre_fit[9]
    best = pre_fit[10]

    if Type == 'gaussian':
        for p in params:              
            priors[p] = gaussian_prior_from_best(best[p], passo[p], nsigma)    
    else:
        for p in params:              
            priors[p] = flat_prior_from_best(best[p], passo[p], nsigma)    
        
    return priors
##########################################################################################################################
def summarize_posterior(values, posterior):
    """
    Retorna mediana e intervalo 68% (16–84%)
    """
    cdf = np.cumsum(posterior)

    p16 = np.interp(0.16, cdf, values)
    p50 = np.interp(0.50, cdf, values)
    p84 = np.interp(0.84, cdf, values)

    return {'median': p50,'minus': p50 - p16,'plus':  p84 - p50,'p16': p16,'p84': p84}



def dividir_ascii_em_partes(
    arquivo_ascii,
    pasta_saida=".",
    n_partes=5,
    intervalo=None
):

    wave = []
    flux = []

    # =========================
    # LEITURA
    # =========================
    with open(arquivo_ascii, 'r') as f:
        for line in f:

            if line.strip() == "" or line.startswith("#"):
                continue

            line = line.replace(",", " ")
            parts = line.split()

            if len(parts) < 2:
                continue

            try:
                wave.append(float(parts[0]))
                flux.append(float(parts[1]))
            except:
                continue

    if len(wave) == 0:
        raise ValueError("Nenhum dado válido")

    wave = np.array(wave)
    flux = np.array(flux)

    # ordena
    idx = np.argsort(wave)
    wave = wave[idx]
    flux = flux[idx]

    os.makedirs(pasta_saida, exist_ok=True)
    nome_base = os.path.splitext(os.path.basename(arquivo_ascii))[0]

    # =========================
    # MODO CORTE ÚNICO
    # =========================
    if intervalo is not None:
        w1, w2 = intervalo

        mask = (wave >= w1) & (wave <= w2)

        wave_cut = wave[mask]
        flux_cut = flux[mask]

        if len(wave_cut) == 0:
            raise ValueError("Intervalo sem dados")

        nome_saida = f"{nome_base}_cut_{int(w1)}_{int(w2)}.txt"
        caminho = os.path.join(pasta_saida, nome_saida)

        np.savetxt(
            caminho,
            np.column_stack([wave_cut, flux_cut]),
            fmt="%.6f"
        )

        print(f"✂ Corte único: {w1}-{w2} Å | {len(wave_cut)} pontos")

        return [caminho]

    # =========================
    # MODO DIVISÃO NORMAL
    # =========================
    w_min = wave.min()
    w_max = wave.max()

    cortes = np.linspace(w_min, w_max, n_partes + 1)

    arquivos_saida = []

    for i in range(n_partes):

        w1 = cortes[i]
        w2 = cortes[i+1]

        mask = (wave >= w1) & (wave <= w2)

        wave_cut = wave[mask]
        flux_cut = flux[mask]

        if len(wave_cut) == 0:
            continue

        nome_saida = f"{nome_base}_part{i+1}_{int(w1)}_{int(w2)}.txt"
        caminho = os.path.join(pasta_saida, nome_saida)

        np.savetxt(
            caminho,
            np.column_stack([wave_cut, flux_cut]),
            fmt="%.6f"
        )

        print(f"✔ Parte {i+1}: {w1:.1f}-{w2:.1f} Å | {len(wave_cut)} pontos")

        arquivos_saida.append(caminho)

    return arquivos_saida
    
def buscar_estrelas_simbad(
    df,
    coluna_ids,
    prefixo_id="HIP",
    colunas=None,
    criteria=None
):

    if colunas is None:
        colunas = ['ra', 'dec']

    # garantir mesdistance
    if 'mesdistance' not in colunas:
        colunas = colunas + ['mesdistance']

    df = df.copy()

    # =========================
    # LIMPEZA ID
    # =========================
    def extrair_id_base(x):
        if pd.isna(x):
            return None
        return str(x).strip().split(",")[0]

    df["id_base"] = df[coluna_ids].apply(extrair_id_base)
    ids_unicos = df["id_base"].dropna().unique()

    # =========================
    # QUERY PRINCIPAL (com distância)
    # =========================
    simbad = Simbad()
    simbad.reset_votable_fields()

    for c in colunas:
        simbad.add_votable_fields(c)

    dfs = []

    for id_base in ids_unicos:
        nome = f"{prefixo_id} {id_base}"

        try:
            res = simbad.query_object(nome, criteria=criteria)
        except Exception:
            continue

        if res is None:
            continue

        df_res = res.to_pandas()
        df_res["id_base"] = id_base
        df_res["query_id"] = nome

        dfs.append(df_res)

    if not dfs:
        return None

    df_all = pd.concat(dfs, ignore_index=True)

    # =========================
    # BASE LIMPA (SEM DUPLICAÇÃO)
    # =========================
    cols_base = ["id_base", "ra", "dec", "main_id", "query_id"]

    df_all_unique = (
        df_all[cols_base]
        .drop_duplicates(subset='id_base')
        .copy()
    )

    df_before = df[['id_base']].drop_duplicates().copy()

    df_before['query_id'] = df_before['id_base'].apply(lambda x: f"{prefixo_id} {x}")

    df_before = df_before.merge(df_all_unique, on='id_base', how='left')

    df_before['query_id'] = df_before['query_id_x'].combine_first(df_before['query_id_y'])

    df_before = df_before.drop(columns=['query_id_x', 'query_id_y'])

    # =========================
    #  REQUERY APENAS FALTANTES
    # =========================
    faltantes = df_before[df_before['ra'].isna()]['id_base'].tolist()

    if faltantes:

        simbad_clean = Simbad()
        simbad_clean.reset_votable_fields()
        simbad_clean.add_votable_fields('ra', 'dec', 'main_id')

        dfs_missing = []

        for id_base in df_before[df_before['ra'].isna()]['id_base']:
            nome = f"{prefixo_id} {id_base}"

            try:
                res = simbad_clean.query_object(nome)
            except Exception:
                continue

            if res is None:
                continue

            df_res = res.to_pandas()
            df_res["id_base"] = id_base

            dfs_missing.append(df_res)

        if dfs_missing:
            df_missing = pd.concat(dfs_missing, ignore_index=True)

            df_missing = df_missing[['id_base', 'ra', 'dec', 'main_id']].drop_duplicates()

            # Merge sem apagar dados existentes
            df_before = df_before.merge(df_missing,on='id_base',how='left',suffixes=('', '_new'))

            df_before['ra'] = df_before['ra'].fillna(df_before['ra_new'])
            df_before['dec'] = df_before['dec'].fillna(df_before['dec_new'])
            df_before['main_id'] = df_before['main_id'].fillna(df_before['main_id_new'])
            

            df_before = df_before.drop(columns=['ra_new', 'dec_new', 'main_id_new'])

    # =========================
    # DF FINAL (melhor distância)
    # =========================
    df_final = df_all.copy()

    if 'mesdistance.method' in df_final.columns:

        prioridade = {
            'paral': 0,
            'CaIIHK': 1,
            'ST-L':2
        
        }

        df_final['priority'] = df_final['mesdistance.method'].map(prioridade).fillna(99)

        df_final = (
            df_final
            .sort_values(['id_base', 'priority'])
            .drop_duplicates(subset='id_base', keep='first')
        )

    else:
        df_final = df_final.drop_duplicates(subset='id_base', keep='first')

    # =========================
    # COORDENADAS
    # =========================
    for dframe in [df_before, df_final]:

        if "ra" in dframe.columns and "dec" in dframe.columns:

            coords = SkyCoord(
                ra=dframe["ra"].values * u.degree,
                dec=dframe["dec"].values * u.degree,
                frame='icrs'
            )

            dframe["RA_HMS"] = coords.ra.to_string(unit=u.hour, sep=':')
            dframe["DEC_DMS"] = coords.dec.to_string(unit=u.degree, sep=':')

    # =========================
    # ORGANIZAÇÃO FINAL
    # =========================
    def organizar(df_in):

        df_in = df_in.copy()
        
        colunas_fixas = ["main_id", "query_id", "id_base", "ra", "dec", "RA_HMS", "DEC_DMS"]

        #  mantém só as que EXISTEM
        colunas_existentes = [c for c in colunas_fixas if c in df_in.columns]

        colunas_restantes = [c for c in df_in.columns if c not in colunas_existentes]

        df_in = df_in[colunas_existentes + colunas_restantes]

        df_in.loc[:, 'id_base_num'] = pd.to_numeric(df_in['id_base'], errors='coerce')

        return (
            df_in
            .sort_values('id_base_num')
            .drop(columns='id_base_num')
            .reset_index(drop=True)
        )

    df_before = organizar(df_before)
    df_final = organizar(df_final)

    return df_before, df_final    


def ler_aglomerados_de_arquivo(caminho_arquivo):
    """
    Lê um arquivo de texto contendo nomes de aglomerados e retorna uma lista.
    
    Parâmetros:
    - caminho_arquivo (str): Caminho do arquivo.
    
    Retorna:
    - list: Lista com os nomes dos aglomerados.
    """
    try:
        with open(caminho_arquivo, 'r') as file:
            aglomerados = []
            for linha in file.readlines():
                partes = linha.split()
                if len(partes) >= 1:
                    aglomerados.append(partes[0])
        return aglomerados
    except FileNotFoundError:
        print(f"O arquivo {caminho_arquivo} não foi encontrado.")
        return []



def parse_ra_dec(coord_str):
    """
    Converte coordenadas no formato string ('HH:MM:SS.SS' ou '±DD:MM:SS.SS') para graus.
    
    - RA (horas:minutos:segundos → graus)
    - DEC (graus:minutos:segundos → graus, mantendo o sinal)
    """
    match = re.match(r"(-?\d+):(\d+):([\d\.]+)", coord_str)
    if not match:
        raise ValueError(f"Formato inválido: {coord_str}")
    
    d, m, s = map(float, match.groups())
    
    # Se for RA (hms), converter para graus
    if ':' in coord_str and coord_str[0] != '-':  # RA
        return 15 * (d + m / 60 + s / 3600)
    
    # Se for DEC (dms), manter o sinal
    sign = -1 if d < 0 else 1
    return sign * (abs(d) + m / 60 + s / 3600)

def equatorial_to_galactic(ra, dec):
    """
    Converte RA/DEC (formato string 'HH:MM:SS.SS' e '±DD:MM:SS.SS' ou graus) para galácticas (l, b).
    
    Aceita:
    - RA: '07:18:41.04' ou 109.67 (graus)
    - DEC: '-24:57:14.4' ou -24.95 (graus)
    
    Retorna:
    - l: Longitude galáctica (graus)
    - b: Latitude galáctica (graus)
    """
    # Converter para graus se for string
    if isinstance(ra, str):
        ra = parse_ra_dec(ra)
    if isinstance(dec, str):
        dec = parse_ra_dec(dec)
    
    # Converter para radianos
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)

    # Passo 1: Coordenadas cartesianas no sistema equatorial (ICRS)
    x_eq = np.cos(dec_rad) * np.cos(ra_rad)
    y_eq = np.cos(dec_rad) * np.sin(ra_rad)
    z_eq = np.sin(dec_rad)

    # Passo 2: Matriz de transformação
    trans_matrix = np.array([
    [-0.0548755604, -0.8734370902, -0.4838350155],
    [ 0.4941094279, -0.4448296300,  0.7469822445],
    [-0.8676661490, -0.1980763734,  0.4559837762]])

    # Multiplicação matricial
    x_gal, y_gal, z_gal = np.dot(trans_matrix, [x_eq, y_eq, z_eq])

    # Passo 3: Converter para l e b
    l_rad = np.arctan2(y_gal, x_gal)  # Longitude galáctica
    b_rad = np.arcsin(z_gal)          # Latitude galáctica

    # Converter para graus
    l = np.degrees(l_rad) % 360
    b = np.degrees(b_rad)

    return l, b

def add_galactic_coordinates(df,ra,dec):
    """
    Adiciona colunas 'l' e 'b' a um DataFrame com colunas 'ra' e 'dec' no formato 'HH:MM:SS.SS' e '±DD:MM:SS.SS'.
    """
    df[['l', 'b']] = df.apply(lambda row: equatorial_to_galactic(row[ra], row[dec]), axis=1, result_type='expand')
    return df

def crossmatch_gaia_bacht(
    df,
    radius_arcsec=1.0,
    chunk_size=500
):

    df = df.copy()

    resultados_totais = []

    # =========================
    # separar em chunks
    # =========================
    n_chunks = int(np.ceil(len(df) / chunk_size))

    for k in range(n_chunks):

        print(f"Chunk {k+1}/{n_chunks}")

        chunk = df.iloc[k*chunk_size:(k+1)*chunk_size].copy()

        # remover sem coord
        chunk_valid = chunk.dropna(subset=['ra', 'dec'])

        if len(chunk_valid) == 0:
            continue

        # =========================
        # upload table
        # =========================
        upload_table = Table.from_pandas(
            chunk_valid[['query_id', 'ra', 'dec']]
        )

        # =========================
        # query Gaia
        # =========================
        query = f"""
        SELECT
            input.query_id,
            g.source_id,
            g.ra AS gaia_ra,
            g.dec AS gaia_dec,
            g.parallax,
            g.ruwe,
            g.phot_g_mean_mag,
            g.radial_velocity,
            DISTANCE(
                POINT('ICRS', input.ra, input.dec),
                POINT('ICRS', g.ra, g.dec)
            ) AS separation
        FROM tap_upload.mytable AS input
        JOIN gaiadr3.gaia_source AS g
        ON 1 = CONTAINS(
            POINT('ICRS', g.ra, g.dec),
            CIRCLE('ICRS', input.ra, input.dec, {radius_arcsec}/3600.)
        )
        """

        try:

            job = Gaia.launch_job_async(
                query=query,
                upload_resource=upload_table,
                upload_table_name="mytable"
            )

            r = job.get_results().to_pandas()

            # =========================
            # manter match mais próximo
            # =========================
            r = (
                r.sort_values(['query_id', 'separation'])
                .drop_duplicates('query_id', keep='first')
            )

            resultados_totais.append(r)

            time.sleep(2)

        except Exception as e:
            print(f"Erro no chunk {k+1}: {e}")

    # =========================
    # concatenar resultados
    # =========================
    if len(resultados_totais) == 0:
        return df

    df_gaia = pd.concat(resultados_totais, ignore_index=True)

    # =========================
    # merge final
    # =========================
    df_final = df.merge(df_gaia, on='query_id', how='left')

    return df_final

def crossmatch_gaia_loop(df, radius_arcsec=1.0, max_retries=3):

    resultados = []

    for i, row in df.iterrows():

        ra = row['ra']
        dec = row['dec']

        if pd.isna(ra) or pd.isna(dec):
            resultados.append({})
            continue

        query = f"""
        SELECT TOP 1
            source_id,
            ra, dec,
            ruwe,
            phot_g_mean_mag
        FROM gaiadr3.gaia_source
        WHERE CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {radius_arcsec}/3600.)
        ) = 1
        """

        sucesso = False

        for tentativa in range(max_retries):

            try:
                job = Gaia.launch_job(query)
                r = job.get_results()

                if len(r) == 0:
                    resultados.append({})
                else:
                    resultados.append(dict(r[0]))

                sucesso = True
                break

            except Exception as e:
                print(f"Erro no índice {i} (tentativa {tentativa+1}): {e}")

                erro_str = str(e)

                # 🔥 ERROS CRÍTICOS DO GAIA → esperar mais
                if (
                    "transaction is aborted" in erro_str
                    or "PooledConnection" in erro_str
                    or "Error 500" in erro_str
                ):
                    print("⚠️ Servidor instável, aguardando...")
                    time.sleep(5)

                else:
                    # erro comum → espera menor
                    time.sleep(2)

        if not sucesso:
            resultados.append({})

        # 🔥 controle de taxa (ESSENCIAL)
        time.sleep(0.5)

    df_gaia = pd.DataFrame(resultados)

    df_gaia = df_gaia.rename(columns={'ra': 'gaia_ra','dec': 'gaia_dec'})

    return pd.concat([df.reset_index(drop=True), df_gaia], axis=1)


def flag_binarias(df):
    """
    Cria flags de binariedade usando SIMBAD + Gaia
    """

    # -------------------------
    # SIMBAD
    # -------------------------
    if 'otype' in df.columns:
        df['bin_simbad'] = df['otype'].str.contains(r'\*\*|SB\*|EB\*|WU\*|bL\*|Al\*',na=False)
    else:
        df['bin_simbad'] = False

    # -------------------------
    # GAIA (RUWE)
    # -------------------------
    if 'ruwe' in df.columns:
        df['bin_gaia'] = df['ruwe'] > 1.4
    else:
        df['bin_gaia'] = False

    # -------------------------
    # FLAG FINAL
    # -------------------------
    df['is_binary'] = df['bin_simbad'] | df['bin_gaia']

    return df

def pipeline_binarias(mode,df_simbad):
    """
    Pipeline completo:
    SIMBAD → Gaia → flags binárias
    """
    if mode == 'loop':
        print("Fazendo crossmatch com Gaia...")
        df = crossmatch_gaia_loop(df_simbad)
    
    else:
        print("Fazendo crossmatch com Gaia...")
        df = crossmatch_gaia_bacht(df_simbad)
    print("Calculando flags de binariedade...")
    df = flag_binarias(df)

    return df


