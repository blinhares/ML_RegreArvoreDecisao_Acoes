# ML. Arvore de Decisão Usando Para Constatar (ou Não) a correlação entre Ações usando `scikit-learn`.

Mais Buscado que a Pedra Filosofal, prever o preco de acoes do mecado financeiro é o desejo de todo esntusiasta da inteligencia artificial. Sabemos que prever esse tipo de coisa é praticamente impossivel mais certamente as ferramentas que temos nos dão um grande auxilio na hora de analizar informações.

Bem, o intuito deste projeto é verificar a correlacao entre os precos de fechamentos de acoes em um dia com o preco de fechamento de uma ação alvo em d+X, elaborar um modelo e valida-lo.

Para ficar mais claro, a ideia é ver se o fato das acoes da `Microsoft` fecharam em alta no dia anterior implica em uma queda nas acoes da `Apple` no dia seguinte por exemplo.

Como podem imagina, não é uma tarefa fácil,mas vamos tentar...

Para isso vamos o algoritimo de usar a biblioteca `DecisionTreeRegressor` do [Scikit-Learn](https://scikit-learn.org/s)

# Instalação 

Com python 3 Instalado...

```bash
pip3 install -U scikit-learn
```

Checando instalacao

```bash
python3 -m pip show scikit-learn  # to see which version and where scikit-learn is installed
python3 -m pip freeze  # to see all packages installed in the active virtualenv
python3 -c "import sklearn; sklearn.show_versions()"
```

No meu caso instalei outras bibliotecas que estaram listadas no arquivo `requeriments.txt`. Na lista esta o Yfinance utilizado para coletar os dados das acoes.

# Documentos

Não cabe aqui a explicação do modelo utilizado mas segue uma serie de links que pode auxiliar.

[Documentação Oficial Scikit-Learn aqui!](https://scikit-learn.org/stable/install.html)

[Exemplo Decision Tree Regression ](https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html#sphx-glr-auto-examples-tree-plot-tree-regression-py)

[Doc. Oficial Decision Trees](https://scikit-learn.org/stable/modules/tree.html#tree)


# Coletando Dados

Vamos Utilizar a biblioteca [yfinance](https://pypi.org/project/yfinance/) para coletar os dados. Para nosso exemplo, vamos utilizar o Preco de Fechamento das Acoes como Anteriormente comentado e o seu historico de 5 anos para treinar nosso modelo.

Criamos uma funcao que coleta todas as informacoes das acoes que julgamos influentes no preco da nossa acao alvo.

Neste caso, como sao muitas acoes eu fiz o downlaod delas e salvei em um arquivo localmente para reduzir o numero de solicitacoes aos servidores do yahoo.

Encontrar um modelo razuavel envolvera uma seria de tentarivas e erros e essas tentativas exigiram carregar dados varias vezes.

Depois coletamos as informacoes da nossa acao alvo com o detalhe de defazar em `X dias` seus resultados para que os dados de um dia das acoes preditoras (que influenciam) se refiram ao valor da acao alvo em X linhas seguintes. Pois nossa intencao é que os valores obtidos na data Y prevejam valores obtidos na data Y+X.

Feito isso é só correr para o abraco.
Fiz um teste com o papel `GRND3.SA` que compartilho com voces.

```bash
╭────────────────────────────────────────────────────────────────  Resumo GRND3.SA ─────────────────────────────────────────────────────────────────╮
│                                                                                                                                                   │
│     Erro Absoluto Médio: 0.1643                                                                                                                   │
│     Erro Quadrático Médio: 0.0504                                                                                                                 │
│     A Raiz do Erro Quadrático Médio (RMSE) : 0.2245                                                                                               │
│     *O valor previsto para o 7 apos hoje é:  6.41                                                                                                 │
│     Uma diferenca % de:  + 0.05 %                                                                                                                 │
│     O ultimo valor foi: 6.13                                                                                                                      │
│                                                                                                                                                   │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

Com um erro Quadratico de : 0.0504, talvez seja uma boa representacao...

Segue o grafico gerado.

![Alt text](https://github.com/blinhares/ML_RegreArvoreDecisao_Acoes/blob/main/resultado_grafico.png "Imagem")

Bem legal né!

Esse projeto tem apenas carater didatico, sinta-se livre pra editar compartilhar e criticar. Caso essa ideia te deixe rico, lembre de mim!