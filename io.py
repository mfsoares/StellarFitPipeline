import pandas as pd
import numpy as np
import os

# This code consernes to the read the file.     


def ler_ascii_obs(caminho):
    # This code consernes to the read the file ,      
    # tentativa 1: pandas padrão
    try:
        df = pd.read_csv(caminho, sep=r'\s+', comment='#')
        if df.shape[1] >= 2:
            df = df.iloc[:, :2]
        df.columns = ['WaveLength', 'NormFluxo']
        return df
    except:
        pass


    # tentativa 2: delimitador vírgula
    try:
        df = pd.read_csv(caminho, sep=',', comment='#')
        if df.shape[1] >= 2:
            df = df.iloc[:, :2]
            df.columns = ['WaveLength', 'NormFluxo']
        return df
    except:
        pass

    # tentativa 3: numpy (mais robusto)
    try:
        data = np.loadtxt(caminho)
        if data.shape[1] >= 2:
            df = pd.DataFrame(data[:, :2], columns=['WaveLength', 'NormFluxo'])
        return df
    except:
        pass

    # tentativa 4: leitura manual (último recurso)
    wave = []
    flux = []

    with open(caminho, 'r') as f:
        for line in f:

            if line.strip() == "" or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            try:
                wave.append(float(parts[0]))
                flux.append(float(parts[1]))
            except:
                continue
    if len(wave) == 0:
        raise ValueError(f"Não foi possível interpretar o arquivo ASCII: {caminho}")
    df = pd.DataFrame({'WaveLength': wave,'NormFluxo': flux})
    return df
        
def carregar_obs(dir_obs, dir_obs2=None, corte=None):

    arq1 = os.path.abspath(dir_obs)

    if dir_obs2 is None:

        obs = ler_ascii_obs(arq1)

        nome1 = os.path.splitext(os.path.basename(arq1))[0]

        arq_out = os.path.join(
            os.path.dirname(arq1),
            f"{nome1}.txt"
        )

    else:

        arq2 = os.path.abspath(dir_obs2)

        obs1 = ler_ascii_obs(arq1)
        obs2 = ler_ascii_obs(arq2)

        if corte is not None:
            obs1 = obs1[obs1['WaveLength'] <= corte]

        obs = pd.concat([obs1, obs2], ignore_index=True)

        nome1 = os.path.splitext(os.path.basename(arq1))[0]
        nome2 = os.path.splitext(os.path.basename(arq2))[0]

        arq_out = os.path.join(
            os.path.dirname(arq1),
            f"{nome1}_{nome2}.txt"
        )

    obs = obs.sort_values('WaveLength').reset_index(drop=True)

    obs.to_csv(arq_out, sep=' ', index=False)

    return obs, arq_out
