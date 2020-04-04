#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns


# In[3]:


df = pd.read_csv('Family Income and Expenditure.csv', sep = ',')


# In[4]:


#df.info()


# In[5]:


#type object: considerado soma de 2 ou mais tipos de variável
for item in list(df.columns.values):
    if df[item].dtype == 'O':
        df[item] = df[item].astype(str) #transformar type(object) para type(str) para nao gerar TypeError


# ## Milestone 1

# #### Amostragem 

# In[6]:


#AMOSTRA ALEATÓRIA SIMPLES (AAS)

index_sample = []
for i in range(100):
    index_sample.append(random.randint(0,(len(df)+1)))
    
df_aas = df.loc[index_sample,:]


# In[7]:


#AMOSTRA SISTEMÁTICA

df = df.sort_values(by = ['Total Household Income'],ascending = True).reset_index().drop(columns = ['index'])

k = len(df)//100

index_sample = []

i = 0 #adicionar valor minimo
for j in range(100):
    index_sample.append(i)
    i+=k
    
df_sist = df.loc[index_sample,:]


# #### Estatísticas Descritivas e Análise Visual

# In[8]:


quant_items = []
quali_items = []

for item in list(df.columns.values):
    if df[item].dtype == 'O':
        quali_items.append(item)
    else:
        quant_items.append(item)


# In[9]:


df_aas_quant = df_aas.drop(columns = quali_items).loc[:,quant_items[0:10]]
df_sist_quant = df_sist.drop(columns = quali_items).loc[:,quant_items[0:10]]
df_all_quant = df.drop(columns = quali_items).loc[:,quant_items[0:10]]

describe_aas = df_aas_quant.describe().drop('count',axis = 0)
describe_sist = df_sist_quant.describe().drop('count',axis = 0)
describe_all = df_all_quant.describe().drop('count',axis = 0)


# In[95]:


#Selecionando as 10 primeiras variáveis quantitativas do sistema, podemos plotar a estatística descritiva
#desses dados e avaliar o quanto as amostras estão próximas do todo 

variavel = list(df_aas_quant.columns.values)

x = [i+1 for i in range(len(describe_aas))]

for i in variavel:
    
    plt.figure(figsize=(6,6))

    y1 = list(describe_aas[i].values)
    y2 = list(describe_sist[i].values)
    y3 = list(describe_all[i].values)
    
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.plot(x,y3)
    
    plt.legend(['AAS','SISTEMÁTICA','ALL'])
    plt.title(i)
    plt.xticks(x, list(describe_aas.index.values))
        
    plt.show()


# #### Considerações

# Conforme observado nas estatísticas descritivas e nos gráficos, a amostragem aleatória simples e a amostragem sistemática trouxeram uma boa representação do todo. 
# 
# No caso do ponto máximo da amostra com todos os valores, podemos considerar o ponto máximo um outlier em praticamente em todos os gráficos, fugindo da distribuição em torno da média, principalmente considerando que o levantamento feito neste estudo é de distribuição de gastos e renda.
# 
# Para eliminar a aleatoriedade das inúmeras vezes que o programa rodar, o estudo será prosseguido com a distribuição sistemática.

# In[93]:


#Análise da distribuição dos dados

#O Gráfico de calor podera nos fornecer informaçoes importante sobre a correlaçao das variaveis!
#O grafico abaixo ira demontrar um grafico de calor montado com as variáveis que apresentam as 9 correlações 
#mais relevantes com nossa variável 'Total Household Income"

k = 10
corrmat = df_sist.corr()
cols = corrmat.nlargest(k, 'Total Household Income')['Total Household Income'].index
f, ax = plt.subplots(figsize=(8, 5))
cm = np.corrcoef(df_sist[cols].values.T)
sns.set(font_scale=0.75)
s = sns.heatmap(cm,  linewidths=0.18,cbar=True, annot=True, square = True, fmt='.2f', 
                annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
ax.set_ylim(0,10)
s.set_title("Variaveis mais correlacionadas com Total Household Income", size=12)
plt.show()


# In[101]:


#Mais uma maneira de avaliar a distribuição e a correlação dos dados!
#pairplot mostrando as relações das variáveis entre elas, e não apenas com o Total Household Income
main_vars = list(cols)

sns.set()
sns.pairplot(df_sist[main_vars], size = 3)
plt.show();


# In[110]:


#Abaixo pdemos analisar as distribuiçoes da variaveis mais correlacionadas com 'Total Household Income'
for k in main_vars:

    x=df_sist[k].values
    f, ax = plt.subplots(figsize=(10, 5))
    try:
        s = sns.distplot(df_sist[k])
        s.set(ylabel='Density')
    except:
        plt.hist(x, label = 'k')
        plt.ylabel('Values')
        plt.xlabel(k)
        
    plt.show()
    print("Skewness:", df_sist[k].skew())


# ### Considerações Finais

# O mapa de calor é uma ferramenta excelente para analisar o quanto uma variável principal está correlacionada com outras.
# 
# O pairplot do seaborn também é uma ótima ferramenta para avaliar distribuição de dados e a correlação entre eles (e não apenas com uma variável principal).
# 
# A partir dos histogramas, com as distribuições de densidade definidas ou, em casos os quais não foi possível plotar a curva de densidade mas apenas seus valores, conseguimos observar suas distribuições e suas assimetrias. 
# Os valores de "Skewness" confirmam a assimetria positiva. 

# ## Milestone 2

# Calcule com 95% de confiança a média de dinheiro gasto com saúde e educação (colunas Medical Care Expenditure e Education Expenditure)

# In[16]:


array_MCE = df_sist['Medical Care Expenditure'].values
array_EE = df_sist['Education Expenditure'].values


# In[17]:


import numpy as np                                                         
import scipy as sp                                                         
import scipy.stats
import statistics as st

def intervalo_conf(array):
    
    media = np.mean(array)
    desv_pad = np.std(array)
    
    conf_int = scipy.stats.norm.interval(0.95, loc=media, scale=desv_pad/np.sqrt(len(array)))
    
    return conf_int


# In[18]:


#Dinheiro gasto com saúde
intervalo_conf(array_MCE)


# In[19]:


#Dinheiro gasto com educação
intervalo_conf(array_EE)


# Calcule se há diferença de média de número total de membros da família pelo sexo do chefe de família (Household Head Sex e Total Number of Family members) com 95% de confiança.

# In[126]:


df_sist_male = df_sist[df_sist['Household Head Sex']=='Male']
df_sist_female = df_sist[df_sist['Household Head Sex']=='Female']

array_male = df_sist_male['Total Number of Family members'].values
array_female = df_sist_female['Total Number of Family members'].values


# In[127]:


from scipy.stats import ttest_ind
from scipy.stats import f

def compare_2_means(array1,array2,alpha):
    
    #TESTE F PARA SABER SE AS VARIÂNCIAS SÃO IGUAIS
    
    equal_vars = 1 #hipotese H0 de que as variancias são iguais
    F = (np.std(array1)**2)/(np.std(array2)**2)
    len_1 = len(array1)-1
    len_2 = len(array2)-1
    
    p_f = f.cdf(F,len_1,len_2) #p-value para o teste f
                                 
    if p_f > alpha:
        equal_vars = 0 #rejeita a hipotese de que as variancias são iguais
        
    #TESTE T PARA SABER SE AS MÉDIAS SÃO IGUAIS
    
    if equal_vars == 0: #rejeita a hipotese de que as variancias são iguais
        stat,p = ttest_ind(array1, array2,equal_var = False)
    else:
        stat,p = ttest_ind(array1, array2,equal_var = True)
        
    print('Statistics:', round(stat,2),',', 'P-value: ',round(p,2))


# In[128]:


compare_2_means(array_male, array_female,0.05)

#Como alpha < p-value, as médias podem ser consideradas iguais


# Calcule se há diferença de média total de renda da família(total household Income) de acordo com a fonte salarial (main source of income)

# In[129]:


# 'Total Household Income'
df_sist['Main Source of Income'].unique()


# In[130]:


df_sist['Main Source of Income'] = df_sist['Main Source of Income'].replace(['Other sources of Income', 
                                                                        'Enterpreneurial Activities'],'Others')
df_sist['Main Source of Income'].unique()


# In[131]:


df_sist_others = df_sist[df_sist['Main Source of Income']=='Others']
df_sist_ws = df_sist[df_sist['Main Source of Income']=='Wage/Salaries']

array_others = df_sist_others['Total Household Income'].values
array_ws = df_sist_ws['Total Household Income'].values


# In[133]:


compare_2_means(array_others, array_ws,0.05)

#Como alpha < p-valor, podemos aceitar H0! Isto é, as médias podem ser consideradas iguais.


# ## Milestone 3

# In[27]:


df['Household Head Job or Business Indicator'].unique() #2 classificações apenas (ufa!)


# In[28]:


df['Household Head Highest Grade Completed'].unique() #imprimir valores únicos e classificá-los manualmente

fundamental = ['Grade 4','Elementary Graduate','No Grade Completed','Grade 2', 'Grade 1', 'Grade 3', 'Grade 5',
               'Grade 6', 'Preschool']
medio = ['High School Graduate','Second Year High School', 'Third Year High School','First Year High School']
superior = ['Second Year College','Computing/Information Technology Programs',
            'Engineering and Engineering trades Programs','Business and Administration Programs',
            'Engineering and Engineering Trades Programs','First Year Post Secondary',
            'Other Programs in Education at the Third Level, First Stage, of the Type that Leads to an Award not Equivalent to a First University or Baccalaureate Degree',
            'Agriculture, Forestry, and Fishery Programs','Transport Services Programs', 'Humanities Programs',
            'Health Programs', 'Second Year Post Secondary','Health Programs', 'Second Year Post Secondary',
            'Security Services Programs', 'Environmental Protection Programs','Physical Sciences Programs', 
            'Basic Programs','Social Services Programs', 'Post Baccalaureate', 'Manufacturing and Processing Programs',
            'Life Sciences Programs','Arts Programs', 'Journalism and Information Programs','Veterinary Programs',
            'Mathematics and Statistics Programs', 'Law Programs','Architecture and Building Programs',
            'Other Programs of Education at the Third Level, First Stage, of the Type that Leads to a Baccalaureate or First University/Professional Degree (HIgher Education Level, First Stage, or Collegiate Education Level)',
            'Social and Behavioral Science Programs','Personal Services Programs', 'Third Year College',
            'First Year College', 'Fourth Year College','Teacher Training and Education Sciences Programs']

ensino = [fundamental, medio, superior]
class_ensino = ['fundamental','medio','superior']

for i in range(len(ensino)):    
    df_sist = df_sist.apply(lambda x: x.replace(ensino[i], class_ensino[i]), axis=0)


# #### Construção de Gráfico para Gestão Visual 

# In[29]:


#https://medium.com/horadecodar/gr%C3%A1ficos-de-barra-com-matplotlib-85628bfc4351

df_nojob = df_sist[df_sist['Household Head Job or Business Indicator']=='No Job/Business']
df_withjob = df_sist[df_sist['Household Head Job or Business Indicator']=='With Job/Business']

#porcentagens em cada nível (fundamental, medio e superior)
x = class_ensino
nojob = list(df_nojob['Household Head Highest Grade Completed'].value_counts().sort_index().values)
withjob = list(df_withjob['Household Head Highest Grade Completed'].value_counts().sort_index().values)

#gráfico
barWidth = 0.25

plt.figure(figsize = (12,6))

r1 = [0,1,2]
r2 = [i + barWidth for i in r1]

plt.bar(r1, nojob, color = '#6A5ACD', width = barWidth, label = 'No Job')
plt.bar(r2, withjob, color = '#00BFFF', width = barWidth, label = 'With Job')

plt.xlabel('Nível de Escolaridade')
plt.xticks([(r+(barWidth)/2) for r in r1],['Fundamental','Médio','Superior'])
plt.ylabel('Porcentagem')
plt.title('Porcentagem de Pessoas com Emprego vs. Nível de Escolaridade')

plt.legend()
plt.show()


# #### Cálculo de Independência 

# Para realizar o cálculo de independência, usaremos o teste do qui-quadrado.
# 
# HIPÓTESES:
# 
# H0: o nível de escolaridade não influencia no fato de ter ou não emprego
# 
# H1: existe diferença
# 
# Sendo assim, vamos considerar como cenário ideal as médias entre as populações.

# In[30]:


#tabela suposta
table = pd.DataFrame(data = {'No Job': nojob, 'With Job': withjob})
table['Soma_h'] = (table['No Job'].values + table['With Job'].values)

table = table.T.rename(columns = {0:'Fundamental',1:'Médio',2:'Superior'})
table_3 = table.drop(index = 'Soma_h')

table['Soma_v'] = table['Fundamental']+table['Médio']+table['Superior']

table


# In[31]:


table_3 #tabela observada


# In[40]:


from scipy.stats import chisquare,chi2_contingency

chi2_3, pval_3, dof_3, exp_3 = chi2_contingency(table_3)


# In[41]:


exp_3 #tabela esperada


# In[42]:


chi2_3, pval_3 #statistica e p-valor


# #### Considerações Finais

#  Conforme observado visualmente no gráfico de número de empregados e desempregados por nível de escolaridade, e confirmado pelo teste Qui-quadrado (statistic < p-value), o grau de escolaridade NÃO influencia na empregabilidade, sendo assim, devemos aceitar a hipótese H0.
