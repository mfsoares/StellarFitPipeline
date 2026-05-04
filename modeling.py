import os
import pandas as pd
import subprocess
import shutil
import numpy as np
from itertools import product

def gera_par_vals(par_central, passo, number):
    indices = np.linspace(-(number-1)/2, (number-1)/2, number)
    return par_central + indices * passo


def gerar_modelos(
    par_central,
    variar,
    path,
    star,
    passo,
    number,
    ordem_parametros,
    model_lim,
    fwhm,
    Dir_OBS,
    model_spectra,
    wstart,
    wend
    ):
#------------------------------------------------------------------------------------------------------------------------------------------------

    elements = ['c','n','o','si']
    parametros = ['teff','logg','vsini','vturb','vmac']
    arquivo_original = "/home/felipe/Documentos/MESTRADO/PROJETO/synplot/synplot/fort.11"
    
    TEFF = par_central['teff']
    LOGG = par_central['logg']
    VSINI = par_central['vsini']
    VTURB = par_central['vturb']
    VMAC = par_central['vmac']
    C = par_central['c']
    N = par_central['n']
    O = par_central['o']
    Si = par_central['si']
 
    
#------------------------------------------------------------------------------------------------------------------------#---------------------------------------------------------------------------------------------------------------------   
    
    for i in variar:
        nome_parametros = "_".join(variar)
        if i in elements and len(variar)>1:
            pasta_destino    = path +str(star)+ '/Abundancia'+'/'+'MultiplosElementos'
        elif i in elements and len(variar)==1:
            pasta_destino    = path +str(star)+ '/Abundancia'+'/'+i
        elif i in parametros and len(variar) >1:
            pasta_destino    = path +str(star)+ '/'+'MultiplosParametros'
        else:
            pasta_destino    = path +str(star)+'/'+i

   # ───────────────────────────────────────────────
   # 1) GARANTE QUE A PASTA DE DESTINO EXISTE
   # ───────────────────────────────────────────────
    os.makedirs(pasta_destino, exist_ok=True)

   # ───────────────────────────────────────────────
   # 2) NORMALIZA "variar" PARA LISTA
   # ───────────────────────────────────────────────
    if isinstance(variar, str):
        variar = [variar]

    variar = [p for p in ordem_parametros if p in variar]

    # ───────────────────────────────────────────────
    # 3) GERA VALORES DOS PARÂMETROS
    # ───────────────────────────────────────────────
    par_vals = {}
    for par in variar:
        pc = par_central[par] if isinstance(par_central, dict) else par_central
        ps = passo[par] if isinstance(passo, dict) else passo
        par_vals[par] = gera_par_vals(pc, ps, number)

    # ───────────────────────────────────────────────
    # 4) LOOP SOBRE COMBINAÇÕES
    # ───────────────────────────────────────────────
    valores_usados = []

    for combo in product(*(par_vals[p] for p in variar)):

        teff, logg, vsini, vturb, vmac, c, n, o, si = TEFF, LOGG, VSINI, VTURB, VMAC, C, N, O, Si

        for p, v in zip(variar, combo):
            if p == 'teff':  teff = v
            if p == 'logg':  logg = v
            if p == 'vsini': vsini = v
            if p == 'vturb': vturb = v
            if p == 'vmac':  vmac = v
            if p == 'c':     c = v
            if p == 'n':     n = v
            if p == 'o':     o = v
            if p == 'si':    si = v

        valores_usados.append({'teff': teff,'logg': logg,'vsini': vsini,
        'vturb': vturb,'vmac': vmac,'c': c,'n': n,'o': o,'si': si  })

        lam_min, lam_max = model_lim

        nome_base = f"teff_{teff:.0f}_logg_{logg:.2f}_vsini_{vsini:.0f}_vturb_{vturb:.0f}_vmac_{vmac:.0f}_C_{c:.2f}_N_{n:.2f}_O_{o:.2f}_Si_{si:.2f}_lim_{lam_min:.0f}_{lam_max:.0f}_fwhm_{fwhm:.3f}"

        destino_parquet = os.path.join(pasta_destino, nome_base + ".parquet")
        destino_txt = os.path.join(pasta_destino, nome_base + ".txt")

        # se já existe parquet, pula
        if os.path.exists(destino_parquet):
            continue

        if model_spectra:
            comando_gdl = """synplot, wstart={0},wend={1},ystyle=1,xstyle=1,/rel,xr=[4458,4500],yr=[0.7,1.1], $
                     teff={2},logg={3:.2f},vrot={4},OBS='{5}',vturb={6},vmac_rt={7},rv=-19,scale=1.007, $ 
                     abund=[2,2,10.91,6,6,{8:.2f},7,7,{9:.2f},8,8,{10:.2f},12,12,7.45,14,14,{11:.2f},26,26,7.49,28,28,6.21],fwhm={12}""".format(wstart,wend,teff,logg,vsini,Dir_OBS,vturb,vmac,c,n,o,si,fwhm)
      
            cmd = ["gdl", "-e", comando_gdl]

            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            proc.wait()

        # copia txt gerado
            shutil.copy2(arquivo_original, destino_txt)

        # ───────────────────────────────────────────────
        # CONVERTE TXT → PARQUET
        # ───────────────────────────────────────────────
            df = pd.read_csv(destino_txt, sep=r'\s+', header=0,
                         names=['WaveLength','NormFluxo'])

            df['WaveLength'] = df['WaveLength'].astype('float32')
            df['NormFluxo']  = df['NormFluxo'].astype('float32')

            df.to_parquet(destino_parquet)

            os.remove(destino_txt)
    return valores_usados, pasta_destino, par_vals
