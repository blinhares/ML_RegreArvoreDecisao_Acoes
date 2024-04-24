import requests
from bs4 import BeautifulSoup
from rich.console import Console
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import lista_acoes
from rich.progress import track
from rich.panel import Panel
from pathlib import Path
from time import sleep
from sklearn.tree import DecisionTreeRegressor

ROOT_DIR = Path(__file__).parent
FILE_DIR = ROOT_DIR / 'src'

con = Console()

def print(*args, **kwargs):
    con.print(*args, **kwargs)

def plog(*args, **kwargs):
    con.log(*args, **kwargs)
         

def coletar_lista_acoes_b3():
    #TODO implementar coleta de dados desta fonte https://www.infomoney.com.br/cotacoes/empresas-b3/
    print('Iniciando Coleta de Dados...')

    saida = set()

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:50.0'
    }

    url = 'https://www.genialinvestimentos.com.br/onde-investir/renda-variavel/acoes/listagem-de-acoes/page/'

    for i in range(1,100):
        try:
            r = requests.get(url + f'/{i}/',headers=headers ,timeout=10)
        except Exception as e:
            print('[red]Time-Out Atingido![/red]')
            continue

        if r.status_code == 200:
                    
            soup = BeautifulSoup(r.text,'html.parser')

            for linha in soup.find_all('a'):
                if not (link := linha.get('href')) == None and '/acoes/' in link:
                    link = link.split('/')[-3:-1] #['acoes', 'aheb6f']
                    if len(link) == 2 and link[0] == 'acoes':
                        saida.add(link[-1].upper()+ '.SA')
            print(f'Pagina {i} [green]carregada[/green]')
                    
    return saida

#####

def coletar_dados_papeis(lista_papeis:list[str], 
                             col=3, periodo = '5y'):
    coluna = ['Open', 'High', 'Low', 'Close', 'Volume',
              'Dividends', 'Stock Splits'][col]
    df_out = pd.DataFrame()
    for papel in lista_papeis:
        numero_tentativas = 3
        while numero_tentativas > 0:
            acao = yf.Ticker(papel)
            df = acao.history(period=periodo)
            if not df.size == 0:
                numero_tentativas = 0
            else:
                numero_tentativas -= 1
                plog(f'[red]ERRO[/red] ao obter os dados de [red]{papel}[/red]')
                plog(f'Uma [b]nova[/b] tentativa sera realizada em [b]3s[/b]')
                sleep(3)

        if df.size == 0:
            plog(f'[red]Não foi possivel obter os dados de {papel}[red]')
            continue
        df.rename(columns={coluna:papel}, inplace=True)
        df = df[papel]
        df_out = pd.concat((df_out,df), axis=1)

    return df_out

def ajustar_df_alvo(df:pd.DataFrame,dias_no_futuro=1):
    dias_futuro = dias_no_futuro
    data = [df[df.columns[0]].\
            loc[df.index[i]] for i in range(dias_futuro,len(df.index))]
    index = [df.index[i] for i in range(len(df.index)-dias_futuro) ]
    return pd.Series(data, index,name=df.columns[0])
    
def download_papel_to_file(papel:str):
    df = coletar_dados_papeis([papel])
    if df.size > 360:
        df.to_parquet(FILE_DIR / f'{papel}.parquet')
        plog(f"Salvo em: {FILE_DIR / f'{papel}.parquet'} - Tam.:{df.size} ")
        return
    plog('[red]Erro[/red] ao gravar arquivo. Tamanho inssuficiente!')

def load_from_files():
    df_out= pd.DataFrame()
    for file in Path.iterdir(FILE_DIR):
        df = pd.read_parquet(file)
        df_out = pd.concat((df_out, df),axis=1)
    #prrencher valores NA com 0
    df_out.fillna(0, inplace=True)
    return df_out
    
####

def main():
    papeis_preditores_considereveis = []
    
    papel_alvo = 'GRND3.SA'
    dias_futoro_prev = 7

    
    #coletar dados do alvo
    df_alvo = coletar_dados_papeis([papel_alvo])
    #recebendo o ultimo valor do alvo para comparar com o previsto
    ultimo_valor = float(df_alvo.iloc[-1].values[0])
    #ajustando o df do alvo
    df_alvo_ajustado = ajustar_df_alvo(df_alvo,dias_futoro_prev)
            
    #Coletar dados dos preditores
    # df_preditores = coletar_dados_papeis(papeis_preditores)
    
    df_preditores = load_from_files()
    if papel_alvo in df_preditores.columns:
        df_preditores.drop(papel_alvo,axis=1)

    #unir dados em um unico df
    df = pd.concat((df_preditores,df_alvo_ajustado),axis=1).\
            sort_index(ascending=False)

    #pegar os dados de hoje para prever informacao
    x_hoje = df.sort_index(ascending=False).head()[:1]
    x_hoje.drop(papel_alvo,axis=1, inplace=True)

    #limpar dados nulos
    df.dropna(inplace=True)

    # separando variaveis
    x = df.drop(papel_alvo, axis=1)
    y = df[papel_alvo]
    #dividinod entre treino e teste
    x_treino, x_teste , y_treino, y_teste = train_test_split(
        x,y,
        test_size=0.2,
        random_state=0)
    #iniciando modelo
    regressor = DecisionTreeRegressor()
    regressor.fit(x_treino,y_treino)
    #modelo treinado
    #fazenod previsao
    y_previsto = regressor.predict(x_teste)

    #incluindo coluna de previsao no df de teste
    x_teste['Previsao']= y_previsto
    x_teste.sort_index(inplace=True)

    #prevendo valor de amanha
    v_dia_seguinte = float(regressor.predict(x_hoje)[0])

    dif = (v_dia_seguinte-ultimo_valor)
    texto = f'''
    Erro Absoluto Médio: {
        round(metrics.mean_absolute_error(y_teste, y_previsto),4)}
    Erro Quadrático Médio: {
        round(metrics.mean_squared_error(y_teste, y_previsto), 4)}
    A Raiz do Erro Quadrático Médio (RMSE) : {
        round(np.sqrt(metrics.mean_squared_error(y_teste, y_previsto)),4)}
    *O valor previsto para o {dias_futoro_prev} apos hoje é: {'[green]' if dif > 0 else f'[red]'} {v_dia_seguinte :.2f}[/]    
    Uma diferenca % de: {'[green] +' if dif > 0 else '[red] '} {dif/ultimo_valor :.2f} %[/]
    O ultimo valor foi: {ultimo_valor :.2f}
'''
    for nome, importancia in zip(regressor.feature_names_in_,regressor.feature_importances_):
        if importancia > 0.2:
            papeis_preditores_considereveis.append(nome)
    con.clear()  
    print(Panel.fit(texto, title=f' Resumo {papel_alvo}'))      
    print(Panel.fit(str(papeis_preditores_considereveis), 
                title='Papeis Consideraveis',
                # subtitle=f'Total: {len(papeis_preditores_considereveis)} - {len(papeis_preditores_considereveis)/(i+conjunto)}'
                ))
    
    # Plot the results
    plt.figure()
    plt.scatter(y.index, y, s=20, edgecolor="black",
                c="darkorange", label="V. Reais")
    
    plt.plot(x_teste.index, x_teste['Previsao'],
            color="cornflowerblue",
            label=f"Previsão d+{dias_futoro_prev}", linewidth=2)
    plt.xlabel("Data")
    plt.ylabel("$Close")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()
        

if __name__ == '__main__':
    main()
    

