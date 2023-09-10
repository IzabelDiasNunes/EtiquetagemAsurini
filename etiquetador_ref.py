import nltk
import re
from nltk.tag import RegexpTagger
from nltk.tag import DefaultTagger


text = open('relatos_asurini_compilado.txt', 'r', encoding='utf-8')
text = text.read()
text_lower = text.lower()
text_clean = re.sub('[.;,-:!?—()]','',text_lower)
set_list = list(set(text_clean.split()))
set_list.sort()
set_list

# padrão desinências

patterns = [
            (r'.*pe$', 'None'),
            (r'.*imo$', 'None'),
            (r'.*i$', 'None'),
            (r".*y’ým.*", 'PRED'),
            (r".*e’ým.*", 'PRED'),
            (r".*y’ýw.*", 'PRED'),
            (r'.*a$', 'ARG'),
            (r'a.*','PRED'),
            (r'o.*','PRED'),
            (r'pe.*','PRED'),
            (r'ere.*','PRED'),
            (r'sa.*','PRED'),
            (r'oro.*','PRED'),
            (r'.*ihí$', 'PRED'),
            (r'.*rapo$', 'PRED'), 
            (r'.*reme$', 'PRED'),
            (r".*mo.*", 'PRED')
]


nom_tagger = nltk.RegexpTagger(patterns)
tags = nom_tagger.tag(set_list)
tags = dict(tags)

for key, value in tags.copy().items():
    if value is None:
        tags[key] = 'NONE'
        
        
print(tags)

tags

import pandas as pd
from sklearn.metrics import f1_score

# Importar o arquivo CSV para um DataFrame
df = pd.read_csv('etiquetas_1.csv')

# Criar um dicionário onde a chave é a palavra e o valor é a etiqueta
palavra_etiqueta_dict = dict(zip(df['Palavra'], df['Etiqueta']))

# Exemplo de outro dicionário para comparar (você pode ajustar isso)
dicionario_comparacao = tags

# Separar palavras e etiquetas do dicionário de comparação
palavras_comparacao = list(dicionario_comparacao.keys())
etiquetas_comparacao = [dicionario_comparacao[palavra] for palavra in palavras_comparacao]

# Criar listas para armazenar as etiquetas reais e previstas
etiquetas_reais = []
etiquetas_previstas = []

# Preencher as listas de etiquetas reais e previstas
for palavra in palavras_comparacao:
    if palavra in palavra_etiqueta_dict:
        etiquetas_reais.append(palavra_etiqueta_dict[palavra])
        etiquetas_previstas.append(dicionario_comparacao[palavra])

# Calcular o F-score usando a função f1_score do sklearn

f_score = f1_score(etiquetas_reais, etiquetas_previstas, average='weighted')
print("F-Score:", f_score)
