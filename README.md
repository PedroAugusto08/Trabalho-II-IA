<h1 align="center">Trabalho Prático IA (2025/2)</h1>

<div align="center">

![VS Code](https://img.shields.io/badge/visual%20studio%20code-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/python-blue?style=for-the-badge&logo=python&logoColor=white)

Inteligência Artificial <br>
Engenharia de Computação <br>
Prof. Tiago Alves de Oliveira <br>
CEFET-MG Campus V <br>
2025/2 


</div>

# Sobre o Repositório

Este repositório é referente ao **Trabalho II** da disciplina de **Inteligência Artificial**, desenvolvido em **Python**. O projeto tem como objetivo aplicar técnicas de Inteligência Artificial no desenvolvimento, treinamento e avaliação de um modelo de aprendizado de máquina, consolidando conceitos teóricos vistos em sala de aula por meio de aplicações práticas.

---

## Atividades Desenvolvidas

- Preparação e organização dos dados;
- Utilização de algoritmos clássicos de IA: SVN, KNN, CLONALG e PSO.
- Avaliação do desempenho dos modelos;
- Geração de gráficos;
- Reprodutibilidade via utilização de mesma geração (seed).

---

## Estrutura do Repositório

O repositório está organizado da seguinte forma:

```plaintext
.
├── ia-trabalho-2025-2
│   ├── data
│   │   └── diabetes_dataset.csv
│   └── src
│       ├── part1_tree_manual
│       │   ├── perguntas.json
│       │   ├── tree_manual.py
│       │   └── tree_diagram.png
│       ├── part2_ml
│       │   ├── train_knn.py
│       │   ├── train_svm.py
│       │   └── train_tree.py
│       ├── part3_ga
│       │   ├── feature_selection.py
│       │   └── ga.py
│       └── part4_swarm_immune
│           ├── clonalg.py
│           ├── fitness.py
│           ├── pso.py
│           └── run_meta.py
├── requirements.txt
└── README.md

```

### Parte 1
- `perguntas.json`: Contém a árvore de decisão com perguntas e resultados;
- `tree_diagram.py`: Contém a implementação que navega pela árvore através de perguntas feitas ao usuário.

### Parte 2
- `train_knn.py`: Busca o melhor valor de `k` via cross-validation sobre o conjunto de treino (apenas valores ímpares), treina o classificador KNN final e gera métricas, matriz de confusão e curva ROC.
- `train_svm.py`: Realiza padronização, aplica PCA (retendo 95% da variância), treina SVM (kernel linear), salva/recupera `svm.model` e avalia o desempenho com métricas e gráficos.
- `train_tree.py`: Treina uma árvore de decisão (com `max_depth` configurável), plota a árvore, gera matriz de confusão, curva ROC e executa avaliação por cross-validation.

### Parte 3
- `feature_selection.py`: Rotinas para seleção de features que serão usadas pelo `ga.py` para avaliação do fitness das soluções.
- `ga.py`: Implementação do Algoritmo Genético para seleção/otimização de features ou hiperparâmetros.

### Parte 4
- `clonalg.py`: Implementação do algoritmo CLONALG (sistema imune artificial) para otimização.
- `pso.py`: Implementação do Particle Swarm Optimization usada em experimentos de otimização.
- `fitness.py`: Funções de avaliação (fitness) utilizadas por PSO/CLONALG/GA.
- `run_meta.py`: Script de integração para executar os experimentos meta-heurísticos e avaliar resultados.

## Execução e Pré-processamento

### Parte 1 — Árvore de Decisão Manual

#### Objetivo
Implementar manualmente uma árvore de decisão simples, sem o uso de bibliotecas
de aprendizado de máquina, com o objetivo de compreender o funcionamento interno
do algoritmo, incluindo critérios de divisão e tomada de decisão.

#### Execução

```powershell
python ia-trabalho-2025-2/src/part1_tree_manual/tree_manual.py
```

### Parte 2 — Aprendizado de Máquina Supervisionado (KNN / SVM / Árvore)

#### Dataset
- Arquivo: data/diabetes_dataset.csv
- Origem: Kaggle
- Tarefa: classificação binária
- Variável alvo (target): diagnosed_diabetes

#### Pré-processamento
- Seleção explícita de 16 features principais diretamente no código
- Não é realizada imputação explícita de valores ausentes
- Padronização com StandardScaler():
  - Aplicada em KNN, SVM e pipelines
- No modelo SVM:
  - Aplicação adicional de PCA com n_components=0.95
  - Retenção de 95% da variância explicada

#### Validação e Avaliação
- Divisão hold-out estratificada 80/20:
  train_test_split(..., stratify=y, random_state=42)
- Validação cruzada estratificada:
  StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- A validação cruzada é utilizada para:
  - Seleção de hiperparâmetros (KNN)
  - Avaliação dos modelos (SVM e Árvore)

#### Reprodutibilidade
A semente aleatória random_state=42 é utilizada de forma consistente para:
- Divisão treino/teste
- Validação cruzada
- Inicialização dos modelos

#### Execução

```powershell
python ia-trabalho-2025-2/src/part2_ml/train_knn.py
python ia-trabalho-2025-2/src/part2_ml/train_svm.py
python ia-trabalho-2025-2/src/part2_ml/train_tree.py
```

### Parte 3 — Algoritmo Genético (GA)

#### Objetivo
Demonstrar o uso de um Algoritmo Genético (GA) para seleção de atributos,
representando soluções como vetores binários.

#### Dataset
Utiliza o dataset embutido da biblioteca scikit-learn:
sklearn.datasets.load_breast_cancer(). Aqui são utilizadas 16 features.

#### Tarefa
- Otimização e seleção de atributos
- Cada indivíduo representa um vetor binário de tamanho 16
- A função de fitness é definida localmente como:
  - Soma dos bits ativos
  - Penalização implícita para soluções triviais

#### Pré-processamento
Não se aplica ao exemplo demonstrativo, pois não há uso de dados reais.
Em um cenário integrado, o GA poderia operar sobre os mesmos dados e rotinas de
pré-processamento definidos na Parte 2.

#### Validação
- O algoritmo genético utiliza exclusivamente a função de fitness definida em código
- Não é empregado hold-out ou validação cruzada neste exemplo

#### Reprodutibilidade
- Uso de seed = 42 para controle do processo evolutivo

#### Execução

```powershell
python ia-trabalho-2025-2/src/part3_ga/ga.py
python ia-trabalho-2025-2/src/part3_ga/feature_selection.py
```

### Parte 4 — PSO / CLONALG / Integração com Classificação

#### Objetivo
Aplicar algoritmos meta-heurísticos inspirados em enxames de partículas (PSO) e
sistemas imunes artificiais (CLONALG) para seleção de atributos, integrando
explicitamente a avaliação com um classificador supervisionado.

#### Dataset
Utiliza o dataset embutido da biblioteca scikit-learn:
sklearn.datasets.load_breast_cancer()

#### Pré-processamento
- Aplicação de StandardScaler() dentro de pipelines
- O pré-processamento ocorre antes da etapa de classificação, respeitando o fluxo
correto de validação

#### Avaliação e Função de Fitness
- A qualidade de cada subconjunto de atributos é avaliada por meio de:
  - LogisticRegression
  - Validação cruzada estratificada
- Configuração padrão:
  StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
- PSO e CLONALG realizam a seleção de atributos
- Validação cruzada é incorporada diretamente na função de fitness

O fluxo feito representa o cenário mais próximo de aplicações reais de otimização
em aprendizado de máquina.

#### Reprodutibilidade
- Uso consistente de seed = 42 para:
  - Algoritmos PSO e CLONALG
  - Validação cruzada
  - Classificador utilizado na função de fitness

#### Execução

```powershell
python ia-trabalho-2025-2/src/part4_swarm_immune/run_meta.py
python ia-trabalho-2025-2/src/part4_swarm_immune/pso.py
python ia-trabalho-2025-2/src/part4_swarm_immune/clonalg.py
```

## Requisitos e Instalação

**Requisitos**
- Python 3.8+ (recomendado 3.10+)
- Dependências listadas em [ia-trabalho-2025-2/requirements.txt](ia-trabalho-2025-2/requirements.txt).

**Instalação rápida**
- Criar e ativar um ambiente virtual (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ia-trabalho-2025-2/requirements.txt
```

## Base de Dados

O dataset principal está em [ia-trabalho-2025-2/data/diabetes_dataset.csv](ia-trabalho-2025-2/data/diabetes_dataset.csv). Os scripts de treino usam esse arquivo por padrão.

Essa base de dados é pública e está disponível no **Kaggle**, ela pode ser visualizada e baixada pelo link: https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset

O conjunto de dados reúne indicadores de saúde relacionados ao diagnóstico de diabetes, e possui grande número maior de variáveis. Para fins deste trabalho, nem todas as colunas foram utilizadas. Foi realizada uma seleção prévia de atributos, considerando-se apenas aquelas características julgadas mais relevantes para a tarefa de classificação proposta. Além disso, durante o processo de utilização do dataset, registros contendo valores ausentes são removidos. Essa decisão foi adotada com o objetivo de evitar a introdução de estimativas arbitrárias por meio de técnicas de imputação, técnicas essas que poderiam comprometer a interpretação dos padrões aprendidos pelos algoritmos e, consequentemente, a validade dos resultados obtidos.

## Autoria e Contato

<div align="center">

### 👤 Jader Oliveira Silva  
<i>Computer Engineering Student @ CEFET-MG</i>  

[![Gmail](https://img.shields.io/badge/Gmail-jaderoliveira28%40gmail.com-D14836?style=for-the-badge&logo=Gmail&logoColor=white)](mailto:jaderoliveira28@gmail.com)

### 👤 Pedro Augusto Gontijo Moura  
<i>Computer Engineering Student @ CEFET-MG</i>  

[![Gmail](https://img.shields.io/badge/Gmail-pedroaugustomoura70927%40gmail.com-D14836?style=for-the-badge&logo=Gmail&logoColor=white)](mailto:pedroaugustomoura70927@gmail.com)

</div>
