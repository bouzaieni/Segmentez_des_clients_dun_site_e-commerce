#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'pycodestyle_magic')
get_ipython().run_line_magic('pycodestyle_on', '')
# Pour vérifier que le code respecte la convention PEP8


# In[2]:


from jyquickhelper import add_notebook_menu
add_notebook_menu()


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns
import matplotlib as mpl
import sys

import sklearn
from datetime import datetime
import re
import missingno as msno
from termcolor import colored, cprint

from sklearn import preprocessing
from sklearn import decomposition
from sklearn.manifold import TSNE

from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans


import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# ## <span style='background:Thistle'>1. Importation des données et affichage de 3 lignes de chaque base</span>

# In[4]:


database_name = ['olist_customers_dataset', 'olist_geolocation_dataset',
                 'olist_order_items_dataset', 'olist_order_payments_dataset',
                 'olist_order_reviews_dataset', 'olist_orders_dataset',
                 'olist_products_dataset', 'olist_sellers_dataset',
                 'product_category_name_translation']
chemin = ''
extension = '.csv'
database = []
for i in range(len(database_name)):
    cprint(' ---------------------------------------------------          ----------------------------------------')
    cprint('Importation de la base ', colored(database_name[i],
                                              'red', attrs=['bold']), ':')
    data = pd.read_csv(chemin + database_name[i] + extension)
    data = pd.DataFrame(data)
    database.append(data)
    print(' Les 3 premèeres lignes de cette base : ')
    display(data.head(3))
df_customers = database[0]
df_geo = database[1]
df_items = database[2]
df_payments = database[3]
df_reviews = database[4]
df_orders = database[5]
df_products = database[6]
df_sellers = database[7]
df_translation = database[8]


# In[5]:


def types_variables(data):
    # Pie plot types de colonnes
    data.dtypes.value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('Types de variables')
    plt.ylabel('')
    plt.show()


# In[6]:


def taux_remplissage(data):
    # Pie plot taux de remplissage du jeu de données
    non_null = data.notna().mean().mean()
    est_null = data.isna().mean().mean()
    index = ['taux de remplissage', 'taux de valeurs manquantes']
    pdf = pd.DataFrame([non_null, est_null], index=index)
    pdf.plot.pie(autopct='%1.1f%%', subplots=True)
    plt.title('Taux de remplissage')
    plt.ylabel('')
    plt.legend(loc='center')
    plt.show()


# In[7]:


def description_base(data, data_name):
    desc_base = []
    print('Analyse de la base ',
          colored(data_name, 'red', attrs=['bold']), ':')
    print(100*'-')
    print('Quelques informations sur la base :\n')
    print(data.info())
    print(100*'-')
    print('Il y a ', data.shape[0], 'lignes et ',
          data.shape[1], 'colonnes dans la base ', data_name)
    desc_base.append(data.shape[1])
    desc_base.append(data.shape[0])
    # données manquantes par colonnes
    print(100*'-')
    print('Le nombre de données manquantes par colonnes : \n',
          data.isna().sum())
    # données manquantes dans toute la base
    print(100*'-')
    print('Le nombre total de données manquantes est : \n',
          data.isna().sum().sum())
    desc_base.append(data.isna().sum().sum())
    # pourcentage des données manquantes
    print(100*'-')
    print('Le pourcentage des données manquantes est : \n',
          round(data.isna().mean().mean()*100, 2), '%')
    desc_base.append(round(data.isna().mean().mean()*100, 2))
    # nombre de doublons sur toutes les colonnes
    if data.duplicated().all():
        print('Il y a ', data.duplicated().sum(), 'lignes dupliquées')
        desc_base.append(data.duplicated().sum())
    else:
        print('Il n\'y a pas de doublons dans cette base')
        desc_base.append(0)
    print(colored(100*'*', 'blue'))
    types_variables(data)
    if data.isna().sum().sum() != 0:
        print(colored(100*'*', 'blue'))
        taux_remplissage(data)
    return desc_base


# In[8]:


database = [df_customers, df_geo, df_items, df_payments,
            df_reviews, df_orders, df_products, df_sellers,
            df_translation]
description_jeu_donnees = []
for i in range(len(database)):
    desc_base = description_base(database[i], database_name[i])
    description_jeu_donnees.append(desc_base)
jeu_donnees = pd.DataFrame(description_jeu_donnees,
                           columns=['Colonnes', 'Lignes',
                                    'Nbre donnees manquantes',
                                    '% donnees manquantes', 'Nbre doublons'],
                           index=database_name)


# In[9]:


print('Tableau récapitulatif')
jeu_donnees


# ## <span style='background:Thistle'>2. Fusion des jeux de données</span>

# In[10]:


# Fusion des jeux de données : df_customers et df_orders
# Variable de jointure : customer_id
data1 = pd.merge(df_customers, df_orders, on='customer_id')
data1.shape


# In[11]:


# Fusion des jeux de données : data1 et  df_reviews
# Variable de jointure : order_id
data3 = pd.merge(data1, df_reviews, on='order_id', how='left')
data3.shape


# In[12]:


# Fusion des jeux de données : data3 et  df_items
# Variable de jointure : order_id
data4 = pd.merge(data3, df_items, on='order_id', how='left')
data4.shape


# In[13]:


# Fusion des jeux de données : data4 et  df_payments
# Variable de jointure : order_id
data5 = pd.merge(data4, df_payments, on='order_id', how='left')
data5.shape


# In[14]:


# Fusion des jeux de données : data5 et  df_products
# Variable de jointure : product_id
data6 = pd.merge(data5, df_products, on='product_id', how='left')
data6.shape


# In[15]:


# Fusion des jeux de données : data6 et  df_translation
# Variable de jointure : product_category_name
data7 = pd.merge(data6, df_translation, on='product_category_name', how='left')
data7.shape


# In[16]:


# Fusion des jeux de données : data7 et  df_sellers
# Variable de jointure : seller_id
data = pd.merge(data7, df_sellers, on='seller_id', how='left')
data.shape


# In[17]:


# liste des colonnes après jointure
liste_Colonnes = data.columns.tolist()
liste_Colonnes


# In[18]:


types_variables(data)


# In[19]:


taux_remplissage(data)


# In[20]:


# Calcul du taux de remplissage  par colonne
plt.figure(figsize=(15, 5))
G = gridspec.GridSpec(1, 1)
ax = plt.subplot(G[0, :])
taux_remplissage = 100-data.isna().mean()*100
ax = taux_remplissage.plot(kind='bar', color='red')
ax.set_title('Taux de remplissage par colonne')
ax.set_xlabel('Colonne')
ax.set_ylabel('Taux de remplissage')
ax.grid(True)
plt.show()


# ## <span style='background:Thistle'>3. Quelques statistiques</span>

# In[21]:


s = 'Nous avons {} différents clients, {} différentes commandes, ' +     '{} différents produits et {} différents vendeurs'
print(s.format(data['customer_unique_id'].nunique(),
      data['order_id'].nunique(),
      data['product_id'].nunique(),
      data['seller_id'].nunique()))


# In[22]:


print("Les commandes ont été passées entre ",
      min(data['order_purchase_timestamp']),
      " et ", max(data['order_purchase_timestamp']))


# In[23]:


# Répartition du nbre de clients par nbre de commandes
ax = plt.gca()
nbre_client = df_customers['customer_unique_id'].nunique()
nbre_commande_client = df_customers.groupby(
    'customer_unique_id').size().value_counts()
df_cmde = pd.DataFrame(
    {'Nbre_commandes': nbre_commande_client.index,
     'Nbre_clients': nbre_commande_client.values})
df_cmde['pourc'] = round(
    (df_cmde['Nbre_clients'])*100/nbre_client, 2)


sns.barplot(x='Nbre_commandes', y='Nbre_clients',
            data=df_cmde, ax=ax)
ax.set_xlabel('Nbre de commandes', fontsize=12)
ax.set_ylabel('Nbbre de clients', fontsize=12)
ax.set_title('Nbre de clients par commandes')
for i, p in enumerate(ax.patches):
    ax.text(
        p.get_width()/4+p.get_x(),
        p.get_height()+p.get_y(),
        df_cmde['pourc'][i]
        )
plt.show()


# Nous remarquons bien que environ 97% des clients ont passé une seule commande sur le site

# In[24]:


time = data.loc[:, ['order_id']]
time['order_purchase_year'] = pd.to_datetime(data[
    'order_purchase_timestamp']).dt.year
time['order_purchase_month'] = pd.to_datetime(data[
    'order_purchase_timestamp']).dt.month

time = time.groupby(
    ['order_purchase_month', 'order_purchase_year']).count().reset_index()
time["period"] = time["order_purchase_year"].astype(str) + "/" + time[
    "order_purchase_month"].astype(str)
time.columns = ["order_purchase_month", "order_purchase_year",
                "Number of order", "period"]
time = time.sort_values(by=['order_purchase_year', 'order_purchase_month'])

# Nbre de commandes en fonction du temps
(fig, ax) = plt.subplots(figsize=(13, 7))
plt.title("Nombre de commandes en fonction du temps")
ax = plt.bar(range(0, time["period"].nunique()),
             time["Number of order"].values)
plt.xticks(range(0, time["period"].nunique()),
           time["period"].unique(), rotation=90)
plt.xlabel("Periode")
plt.ylabel("Nombre de commandes")
plt.show()


# In[25]:


time_month = data.loc[:, ['order_id']]
time_month['order_purchase_month'] = pd.to_datetime(
    data['order_purchase_timestamp']).dt.month_name()
time_month = time_month.groupby('order_purchase_month').count().reset_index()
time_month.columns = ["order_purchase_month", "Number of order"]

# Nombre de commandes en fonction des mois
(fig, ax) = plt.subplots(figsize=(13, 7))
plt.title("Nombre de commandes en fonction des mois")
ax = plt.bar(range(0, time_month["order_purchase_month"].nunique()),
             time_month["Number of order"].values)
plt.xticks(range(0, time_month["order_purchase_month"].nunique()),
           time_month["order_purchase_month"].unique(), rotation=90)
plt.xlabel("Mois")
plt.ylabel("Nombre de commandes")
plt.show()


# In[ ]:





# In[26]:


time_day = data.loc[:, ['order_id']]
time_day['order_purchase_day'] = pd.to_datetime(
    data['order_purchase_timestamp']).dt.day_name()
time_day = time_day.groupby('order_purchase_day').count().reset_index()
time_day.columns = ["order_purchase_day", "Number of order"]

# Nombre de commandes en fonction des jours de la semaine
(fig, ax) = plt.subplots(figsize=(13, 7))
plt.title("Nombre de commandes en fonction des jours de la semaine")
ax = plt.bar(range(0, time_day["order_purchase_day"].nunique()),
             time_day["Number of order"].values)
plt.xticks(range(0, time_day["order_purchase_day"].nunique()),
           time_day["order_purchase_day"].unique(), rotation=90)
plt.xlabel("Jour")
plt.ylabel("Nombre de commandes")
plt.show()


# In[ ]:





# In[27]:


# Nombre de clients par état
clients_par_etat = data.groupby(
    ['customer_id', 'customer_state']).count().reset_index()
sns.countplot(x=clients_par_etat.customer_state, data=clients_par_etat,
              order=clients_par_etat.customer_state.value_counts().index)
plt.title('Nombre de clients par état')
plt.ylabel('Nombre', fontsize=14)
plt.xlabel('Etat', fontsize=14)

plt.show()


# In[28]:


# nombre de commandes par état
sns.countplot(x='geolocation_state', data=df_geo,
              order=df_geo['geolocation_state'].value_counts(
              ).sort_values(ascending=False).index)

plt.title('Nombre de commandes par état '.title(),
          fontsize=20)
plt.ylabel('Nombre'.title())
plt.xlabel('Etat'.title())


plt.show()


# In[29]:


# Répartition des différents mode de paiements
fig, ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(aspect='equal'))
explode = (0.1, 0, 0, 0)
legend = ['Carte crédit', 'Billet', 'Voucher',
          'Carte débit']

p = df_payments['payment_type'][df_payments['payment_type']
                                != 'not_defined'].value_counts()
p.plot(kind="pie", legend=False, labels=None, autopct='%1.0f%%',  ax=ax)
ax.legend(legend, loc='best', shadow=True,
          prop={'weight': 'bold'},
          bbox_to_anchor=(0.8, 0, 0.5, 1))
plt.title('Différents Types de paiement')
plt.ylabel("")
plt.show()


# In[30]:


# Répartition des différents mode de paiements
fig, ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(aspect='equal'))
explode = (0.1, 0, 0, 0)
legend = ['5', '4', '1', '3', '2']

p = df_reviews['review_score'][df_reviews['review_score']
                               != 'not_defined'].value_counts()
p.plot(kind="pie", legend=True, labels=None, autopct='%1.0f%%',  ax=ax)
ax.legend(legend, loc='best', shadow=True,
          prop={'weight': 'bold'},
          bbox_to_anchor=(0.8, 0, 0.5, 1))
plt.title('Différents notes des avis du client sur les commandes')
plt.ylabel("")
plt.show()


# In[31]:


# catégories des produits les plus commandés
plt.figure(figsize=(14, 8))

sns.countplot(data=df_products, x='product_category_name',
              order=df_products.product_category_name
              .value_counts().index)

plt.title('Catégories des produits les plus commandés')
plt.ylabel('Nombre')
plt.xlabel('Catégorie des produits')

plt.xticks(rotation=90)
plt.yticks()
plt.grid(False)
plt.show()


# ## <span style='background:Thistle'>4. Imputation des valeurs manquantes</span>

# In[32]:


def informations_valeurs_manqantes(df):
    print('Nombre de valeurs manquantes par colonne')
    msno.bar(df)
    print(100*'*')
    print('Matrice de chaleur des valeurs manquantes')
    msno.heatmap(df)
    print(100*'*')
    print('Dendogramme des valeurs manquantes')
    msno.dendrogram(df)
    print(100*'*')


# In[33]:


informations_valeurs_manqantes(data)


# In[34]:


data = data.drop(['review_comment_title', 'review_comment_message'], 1)
data = data.dropna()


# ## <span style='background:Thistle'>5. Analyse valeur aberrante</span>

# In[35]:


colonnes_numeriques = data.select_dtypes(include=[np.number]).columns.tolist()


# In[36]:


for colonne in colonnes_numeriques:
    sns.boxplot(x=colonne, data=data, whis=[10, 90])
    plt.title(('Boite a moustache de la colonne ' + colonne))
    plt.show()


# D'après ces boites à moustaches, nous observons qu'il ya des valeurs aberrantes pour la majorité des colonnes. Je vais traiter ces valeurs aberranes par la méthode des quartiles (0.1 comme quartile inf et 0.9 comme quartile sup pour ne pas perdre beaucoups d'observations). Le nombre de clients ayant fait plus d'une commande  est d'environ 3%.

# In[37]:


def traitement_valeurs_aberrantes(data, colonne):
    Q1 = data[colonne].quantile(0.1)
    Q3 = data[colonne].quantile(0.9)
    borneInf = Q1 - 1.5*(Q3 - Q1)
    borneSup = Q3 + 1.5*(Q3 - Q1)
    data.drop(data.loc[data[colonne] > borneSup].index, inplace=True)
    data.drop(data.loc[data[colonne] < borneInf].index, inplace=True)


# In[38]:


for colonne in colonnes_numeriques:
    traitement_valeurs_aberrantes(data, colonne)


# In[39]:


dataa = data.copy()


# ## <span style='background:Thistle'>6. Analyse univariée</span>

# In[40]:


colonnes_numeriques.remove('order_item_id')
colonnes_numeriques.remove('payment_sequential')
colonnes_numeriques.remove('seller_zip_code_prefix')
colonnes_numeriques.remove('customer_zip_code_prefix')


# In[41]:


def analyse_univariee(data, colonne, label):
    print(f'moyenne : {round(data[colonne].mean(), 2)}')
    print(f'mediane : {round(data[colonne].median(), 2)}')
    print(f'mode : {round(data[colonne].mode(), 2)}')
    print(f'variance : {round(data[colonne].var(), 2)}')
    print(f'skewness : {round(data[colonne].skew(), 2)}')
    print(f'kurtosis : {round(data[colonne].kurtosis(), 2)}')
    print(f'ecart type : {round(data[colonne].std(), 2)}')
    print(f'min : {round(data[colonne].min(), 2)}')
    print(f'25% : {round(data[colonne].quantile(0.25), 2)}')
    print(f'50% : {round(data[colonne].quantile(0.5), 2)}')
    print(f'75% : {round(data[colonne].quantile(0.75), 2)}')
    print(f'max : {round(data[colonne].max(), 2)}')
    print(colored('Interprétation', 'red', attrs=['bold']))
    if np.floor(data[colonne].skew()) == 0:
        print('la distribution de la colonne ' + colonne + ' est symétrique')
    elif round(data[colonne].skew(), 2) > 0:
        print('la distribution de la colonne ' + colonne +
              ' est étalée à droite')
    else:
        print('la distribution de la colonne ' + colonne +
              ' est étalée à gauche')
    if np.floor(data[colonne].kurtosis()) == 0:
        print('la distribution de la colonne ' + colonne +
              ' a le même aplatissement que la distribution normale')
    elif round(data[colonne].kurtosis(), 2) > 0:
        print('la distribution de la colonne ' + colonne +
              ' est moins aplatie que la distribution normale')
    else:
        print('la distribution de la colonne ' + colonne +
              ' est plus aplatie que la distribution normale')
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 2, 1)
    sns.boxplot(data[colonne], width=0.5, color='red')
    plt.title('Boite a moustache de la colonne ' + label, fontsize=15)
    plt.subplot(1, 2, 2)
    sns.histplot(data[colonne], kde=True, color='blue')
    plt.title('histogramme de la colonne  ' + label, fontsize=15)
    plt.show()
    plt.tight_layout()


# In[42]:


for colonne in colonnes_numeriques:
    print(colored(100*'*', 'blue', attrs=['bold']))
    print(colored('Analyse de la colonne ' + colonne, 'red', attrs=['bold']))
    analyse_univariee(data, colonne, str(colonne))
print(colored(150*'*', 'blue', attrs=['bold']))


# ## <span style='background:Thistle'>7. Analyse multivariée</span>

# In[43]:


def matrice_correlation(data, colonnes_a_analyser):
    plt.rcParams["figure.figsize"] = [15, 7]
    data = data[colonnes_a_analyser]
    mask = np.triu(np.ones_like(data.corr(), dtype=bool))
    sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True, fmt=".2f", mask=mask)
    plt.title('matrice de corrélation entre les colonnes ')
    plt.show()


# In[44]:


matrice_correlation(data, colonnes_numeriques)


# D'après la matrice de corrélation, nous remarquons une forte corrélation entre payment_value et price.
# 
# Il faut supprimer l'une de ces variables ou en créer des nouvelles

# ## <span style='background:Thistle'>8. Sélection des colonnes utiles</span>

# In[45]:


data = dataa[['customer_unique_id', 'customer_state',
              'order_id', 'order_purchase_timestamp',
              'order_delivered_customer_date', 'product_id',
              'price', 'payment_value', 'payment_type',
              'freight_value', 'payment_installments', 'product_length_cm',
              'product_height_cm', 'product_width_cm',
              'review_score', 'product_category_name_english']].copy()


# In[46]:


def val_colonne(colonne):
    print('nbre_valeurs : ', data[colonne].nunique())
    if data[colonne].dtypes != 'object':
        print('Liste_valeurs : ', sorted(data[colonne].unique().tolist()))
    else:
        print('Liste_valeurs : ', data[colonne].unique().tolist())


# In[47]:


val_colonne('product_category_name_english')


# In[48]:


# Le nombre de catégories est 71. C'est beaucoup.
# Il faut créer un dictionnaire pour regrouper les catégories
get_ipython().run_line_magic('pycodestyle_off', '')
dict_categories = {
            'la_cuisine':'home',
            'small_appliances_home_oven_and_coffee':'home',
            'home_comfort_2':'home',
            'home_appliances_2':'home',
            'furniture_mattress_and_upholstery':'home',
            'bed_bath_table':'home', 
            'kitchen_dining_laundry_garden_furniture':'home',            
            'furniture_living_room':'home',
            'furniture_bedroom':'home',
            'small_appliances':'home',           
            'home_appliances':'home',
            'home_confort':'home',
            'fashion_male_clothing':'fashion',
            'fashio_female_clothing':'fashion',
            'fashion_sport':'fashion',
            'fashion_childrens_clothes':'fashion',
            'fashion_shoes':'fashion',
            'fashion_underwear_beach':'fashion',
            'sports_leisure':'fashion',
            'fashion_bags_accessories':'fashion',
            'furniture_decor':'fashion',
            'luggage_accessories':'fashion',
            'security_and_services':'tools',
            'arts_and_craftmanship':'tools',
            'signaling_and_security':'tools',
            'construction_tools_safety':'tools',
            'industry_commerce_and_business':'tools',
            'home_construction':'tools',
            'costruction_tools_tools':'tools',
            'garden_tools':'tools',
            'air_conditioning':'tools',     
            'construction_tools_lights':'tools',           
            'construction_tools_construction':'tools',           
            'costruction_tools_garden':'tools',
            'auto':'tools',
            'computers_accessories':'multimedia',
            'dvds_blu_ray':'multimedia',
            'audio':'multimedia',
            'musical_instruments':'multimedia',
            'art':'multimedia',
            'consoles_games':'multimedia',
            'books_imported':'multimedia',
            'cds_dvds_musicals':'multimedia',
            'music':'multimedia',
            'computers':'multimedia',
            'books_general_interest':'multimedia',
            'books_technical':'multimedia',
            'office_furniture':'multimedia',           
            'fixed_telephony':'multimedia',
            'tablets_printing_image':'multimedia',
            'stationery':'multimedia',
            'telephony':'multimedia',
            'cine_photo':'multimedia',     
            'watches_gifts':'multimedia',
            'electronics':'multimedia',
            'cine_photo':'multimedia',
            'flowers':'consumption',
            'food':'consumption',
            'agro_industry_and_commerce':'consumption',
            'diapers_and_hygiene':'consumption',
            'health_beauty':'consumption',
            'perfumery':'consumption',
            'party_supplies':'consumption',
            'toys':'consumption',
            'drinks':'consumption',
            'christmas_supplies':'consumption',
            'pet_shop':'consumption',
            'housewares':'consumption',
            'baby':'consumption',
            'food_drink':'consumption',
            'market_place':'consumption',                
            'cool_stuff':'consumption',
            }


# In[49]:


get_ipython().run_line_magic('pycodestyle_on', '')
data['product_category_name'] = data[
    'product_category_name_english'].replace(dict_categories)


# In[50]:


val_colonne('product_category_name')


# In[51]:


val_colonne('customer_state')


# In[52]:


# Créer un dictionnaire pour regrouper les régions
dict_state = {
    "DF": "centre_ouest",
    "GO": "centre_ouest",
    "MS": "centre_ouest",
    "MT": "centre_ouest",
    "AL": "nord_est",
    "BA": "nord_est",
    "CE": "nord_est",
    "MA": "nord_est",
    "PE": "nord_est",
    "PB": "nord_est",
    "PI": "nord_est",
    "RN": "nord_est",
    "SE": "nord_est",
    "AC": "nord",
    "AM": "nord",
    "AP": "nord",
    "PA": "nord",
    "RO": "nord",
    "RR": "nord",
    "TO": "nord",
    "ES": "sud_est",
    "MG": "sud_est",
    "RJ": "sud_est",
    "SP": "sud_est",
    "PR": "sud",
    "RS": "sud",
    "SC": "sud",
    }


# In[53]:


data['customer_state'] = data['customer_state'].replace(dict_state)


# In[54]:


val_colonne('customer_state')


# ## <span style='background:Thistle'>9. Feature engineering</span>

# Nouvelle variable nombre de commande

# In[55]:


nbre_commande = pd.DataFrame(data.groupby(["customer_unique_id"])
                             ["order_id"].nunique())
nbre_commande.rename(columns={"order_id": "nbre_commande"}, inplace=True)
data = pd.merge(data, nbre_commande, on='customer_unique_id')


# Nouvelle variable nombre de produits

# In[56]:


nbre_produit = pd.DataFrame(data.groupby(["customer_unique_id"])
                            ["product_id"].count())
nbre_produit.rename(columns={"product_id": "nbre_produit"}, inplace=True)
data = pd.merge(data, nbre_produit, on='customer_unique_id')


# Nouvelle variable montant des achats

# In[57]:


# Montant total des achats par client
montant_achat = pd.DataFrame(data.groupby(['customer_unique_id'])
                             ['price'].sum())
montant_achat.rename(columns={"price": "montant_achat"},
                     inplace=True)
data = pd.merge(data, montant_achat, on='customer_unique_id')
montant_achat.head()


# Nouvelle variable note moyenne des reviews

# In[58]:


# Montant total des achats par client
score_moyen = pd.DataFrame(data.groupby(['customer_unique_id'])
                           ['review_score'].mean())
score_moyen.rename(columns={"review_score": "score_moyen"},
                   inplace=True)
data = pd.merge(data, score_moyen, on='customer_unique_id')
score_moyen.head()


# Nouvelle variable delai de livraison

# In[59]:


get_ipython().run_line_magic('pycodestyle_off', '')
var1 = 'order_purchase_timestamp'
var2 = 'order_delivered_customer_date'
var3 = 'delai_livraison'
data[var1] = pd.to_datetime(data[var1])
data[var2] = pd.to_datetime(data[var2])
data[var3] = (data[var2] - data[var1]).apply(lambda x:x.days)


# In[60]:


get_ipython().run_line_magic('pycodestyle_on', '')
data = data[['customer_unique_id', 'order_id', 'customer_state',
             'order_purchase_timestamp', 'payment_type', 'payment_value',
             'product_category_name',  'nbre_commande', 'nbre_produit',
             'montant_achat', 'score_moyen', 'delai_livraison']]


# ## <span style='background:Thistle'>10. Data Frame RFM</span>

# In[61]:


data.head()


# In[62]:


def clean(row):
    try:
        return pd.to_datetime(row['order_purchase_timestamp'],
                              format="%Y %m %d")
    except ValueError:
        data.drop(row.name, inplace=True, axis=0)


data['order_purchase_timestamp'] = data.apply(clean, axis=1)


# In[63]:


date_recence_max = data['order_purchase_timestamp'].max()
date_recence_max_str = date_recence_max.strftime('%d/%m/%Y %Hh%m')
print(f'La data à partir de la quelle on va calculer      la récence en nombre de jours:', {date_recence_max_str})


# In[64]:


# Récence = nombre de jours depuis le dernier achat
dict_rfm = {'order_purchase_timestamp': lambda x: (
    date_recence_max - x.max()).days}
dict_new_rfm = {'order_purchase_timestamp': 'recence'}


# In[65]:


# Fréquence = nombre d'achat sur toute la période
dict_rfm['order_id'] = 'count'
dict_new_rfm['order_id'] = 'frequence'


# In[66]:


# Montant = somme des paiements sur toute la période
dict_rfm['payment_value'] = 'sum'
dict_new_rfm['payment_value'] = 'montant'


# In[67]:


# Dataframe RFM
df_rfm = data.groupby('customer_unique_id').agg(dict_rfm)
df_rfm.rename(columns=dict_new_rfm, inplace=True)
df_rfm.reset_index(inplace=True)
data = data.drop(['order_id', 'payment_value'], 1)


# In[68]:


df_rfm.head()


# ## <span style='background:Thistle'>11. Analyse en composantes principales (ACP)</span>

# In[69]:


data_acp = pd.merge(data, df_rfm, on='customer_unique_id')


# In[70]:


colonnes_a_normaliser = data_acp.select_dtypes(include=[np.number])                        .columns.tolist()
data_a_normaliser = data_acp[colonnes_a_normaliser].copy()
scaler = preprocessing.StandardScaler()

data_a_normaliser = scaler.fit_transform(data_a_normaliser)
data_a_normaliser = pd.DataFrame(data_a_normaliser,
                                 columns=colonnes_a_normaliser,
                                 index=data_acp.index.to_list())
for colonne in colonnes_a_normaliser:
    data_acp[colonne] = data_a_normaliser[colonne]


# In[71]:


colonnes_categoriques = ['customer_state', 'payment_type',
                         'product_category_name']


# In[72]:


def encodage_categorielle(data, colonne):
    df_encodage = pd.get_dummies(data[colonne], prefix=colonne)
    return (df_encodage)


# In[73]:


data_encodage_categorielle = pd.DataFrame(index=data_acp.index)
for colonne in colonnes_categoriques:
    df_encodage = encodage_categorielle(data_acp, colonne)
    data_encodage_categorielle = pd.concat([data_encodage_categorielle,
                                            df_encodage], axis=1)
data_acp = data_acp.drop(colonnes_categoriques, 1)
data_acp = pd.concat([data_acp, data_encodage_categorielle], axis=1)


# In[74]:


data_acp = data_acp.drop('order_purchase_timestamp', 1)
data_acp = data_acp.set_index('customer_unique_id')
pca = decomposition.PCA(n_components=22)
pca.fit(data_acp)


# Pour avoir un affichage clair des ercles de corrélations, je vais renommer les colonnes avec des noms plus courts

# In[75]:


get_ipython().run_line_magic('pycodestyle_off', '')
data_acp.rename(columns={'nbre_commande':'nc', 'nbre_produit':'nb', 
                         'montant_achat':'ma', 'score_moyen':'S',
                         'delai_livraison':'L', 'recence':'R', 
                         'frequence':'F', 'montant':'M',
                         'customer_state_centre_ouest':'co', 
                         'customer_state_nord':'sn',
                         'customer_state_nord_est':'sne', 
                         'customer_state_sud':'ss',
                         'customer_state_sud_est':'sse', 
                         'payment_type_boleto':'pb',
                         'payment_type_credit_card':'pcc', 
                         'payment_type_debit_card':'pcd',
                         'payment_type_voucher':'pv', 
                         'product_category_name_consumption':'con',
                         'product_category_name_fashion':'fas', 
                         'product_category_name_home':'hom',
                         'product_category_name_multimedia':'mul', 
                         'product_category_name_tools':'too'}, 
                 inplace = True)


# In[76]:


get_ipython().run_line_magic('pycodestyle_on', '')
data_acp1 = pca.transform(data_acp)


# In[77]:


# diagramme d’éboulis des valeurs propres
cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.plot(np.arange(1, len(cumsum)+1), cumsum*100, 'o:', color='red')
plt.xlim(0, 22)
plt.ylim(0, 100)
plt.grid()
plt.xlabel('Rang de l\'axe d\'inertie')
plt.ylabel('Pourcentage d\'inertie %')
#  trouver le moment où on attient 95% d'informations
plt.plot(cumsum)
# argmax pour > 90 %
nb_composantes_arret = np.argmax(cumsum > 0.90)
plt.axhline(y=90, color='r')
plt.text(3, 92, '>90%', color='b', fontsize=10)
plt.axvline(x=nb_composantes_arret, color='r')
plt.title('Diagramme des éboulis de valeurs propres avec somme cumulée')
plt.bar(np.arange(1, len(cumsum)+1), pca.explained_variance_ratio_*100,
        color='darkred')


# La première composante du pca résume plus de 35% de l'informations.Les cinq dernières composantes résument peu d'informations(< 5%). Nous pouvons retenir seulement 8 composantes (on perd seulement 10% d'informations) et comme ça on arrive à réduire le nombre de variables de 22 à 8.

# In[78]:


pca = decomposition.PCA(n_components=8)
pca.fit(data_acp)
data_acp1 = pca.transform(data_acp)
data_pca = pd.DataFrame(data_acp1)
col = ["pca " + str(n + 1) for n in data_pca.columns]
data_pca.columns = col
data_pca.head()


# In[79]:


def cercle_correlation(pca, features, x, y):
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(0, pca.components_.shape[1]):
        ax.arrow(0, 0,
                 pca.components_[x, i],
                 pca.components_[y, i],
                 head_width=0.07,
                 head_length=0.07,
                 width=0.02)

        plt.text(pca.components_[x, i] + 0.05,
                 pca.components_[y, i] + 0.05,
                 features[i])

    # affichage des lignes horizontales et verticales
    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')
    # nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel('F{} ({}%)'.format(x+1,
               round(100*pca.explained_variance_ratio_[x], 1)))
    plt.ylabel('F{} ({}%)'.format(y+1,
               round(100*pca.explained_variance_ratio_[y], 1)))
    plt.title("Cercle des corrélations (F{} et F{})".format(x+1, y+1))
    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
    plt.axis('equal')
    plt.show(block=False)


# In[80]:


i = 0
while i < 7:
    cercle_correlation(pca, data_acp.columns.to_list(), i, i+1)
    i = i+2


# D'aprés les cercles de corrélations, nous remarquons :
# 
# L'axe F1 contient les clients qui ont dépensé beaucoup et qui ont commandé plusieurs produits
# 
# L'axe F2 contient les clients qui ont un grand delai de livraison et ayant donné une mauvaise note
# 
# L'axe F3 contient les clients qui ont commandé dpuis longtemps et ont donné une bonne note
# 
# L'axe F4 contient les clients qui ont passé plusieurs commandes avec un montant faible
# 
# L'axe F5 contient les clients qui ont donné une bonne note malgrés le delai de livraison important
# 
# L'axe F6 contient les clients qui ont passé plusieurs commandes avec un grand montant d'achat
# 
# L'axe F7 contient les clients qui ont commandé avec une carte de crédit et non par boleto
# 
# L'axe F8 contient les clients qui ont commandé des produits de catégorie consumption et non multimédia

# ## <span style='background:Thistle'>12. Sauvegarde du jeu de données</span>

# In[81]:


data.to_csv('data.cleaned.csv', index=False)
df_rfm.to_csv('df_rfm.csv', index=False)


# In[ ]:




