import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import plotly.graph_objs as go
import plotly.offline as py
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import seaborn as sns

# Importer les 2 fichiers (entrainement et test) combinés
abonnes = pd.concat(map(pd.read_csv, ["D:\Documents\Github\Telecom-churn\churn-bigml-80.csv", "D:\Documents\Github\Telecom-churn\churn-bigml-20.csv"]))
print("# # # abonnes.head() # # #\n", abonnes.head().transpose())
print("\nabonnes.info() :\n")
print(abonnes.info())
print("# # # Valeurs uniques # # #\n", abonnes.nunique(), "\n")

# Transformer les attributs non-numeriques en numeriques (sauf State)
print("# # # Avant la transformation # # #\n")
print("Service des appels internationaux :\n", abonnes['International plan'].value_counts(), "\n")
print("Service de la messagerie vocale :\n", abonnes['Voice mail plan'].value_counts(), "\n")
print("Désabonnement :\n", abonnes['Churn'].value_counts(), "\n")
# La variable col_bin contient les noms des 3 attributs
col_bin = abonnes.nunique()[abonnes.nunique() == 2].keys().tolist()
# Encoder les 3 attributs en 0 ou 1
encodeur = LabelEncoder()
for i in col_bin:
  abonnes[i] = encodeur.fit_transform(abonnes[i])
# Vérifier les 3 attributs
print("# # # Apres la transformation # # #\n")
print("Service des appels internationaux :\n", abonnes['International plan'].value_counts(), "\n")
print("Service de la messagerie vocale :\n", abonnes['Voice mail plan'].value_counts(), "\n")
print("Désabonnement :\n", abonnes['Churn'].value_counts(), "\n")
print("abonnes.info() :")
print(abonnes.info())

# Tableau de sommaire des 19 colonnes numériques (de type int ou float)
sommaire = (abonnes[[i for i in abonnes.columns]].describe().transpose().reset_index())
print("\n# # # Sommaire # # #\n", sommaire)


# Corrélation de Pearson
numerics = ['int32', 'int64', 'float64'] 
abonnes_num = abonnes.select_dtypes(include=numerics)
correlation = abonnes_num.corr()
col_mat = correlation.columns.tolist()
corr_array = np.array(correlation) # convertir en array
trace = go.Heatmap(z = corr_array,
                   x = col_mat,
                   y = col_mat,
                   colorscale = "rainbow",
                   colorbar = dict(title = "Coefficients de corrélation de Pearson", titleside = "right"),
                  )
layout = go.Layout(dict(title = "Matrice de corrélation",
                        autosize = False,
                        height = 720,
                        width = 800,
                        margin = dict(r = 0, l = 210, t = 25, b = 210),
                        yaxis = dict(tickfont = dict(size = 9)),
                        xaxis = dict(tickfont = dict(size = 9))
                       )
                  )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.plot(fig)

''' Le resultat de la corrélation de Pearson montre qu'il y a 5 couples d'attributs qui sont fortement corrélés :
1. Total intl minutes et Total intl charge
2. Total night minutes et Total night charge
3. Total eve minutes et Total eve charge
4. Total day minutes et Total day charge
5. Voice mail plan et Number vmail messages
Alors, on garde le premier attribut seulement de chaque couple pendant la modélisation. On crée la liste 
'supprimer' qui contient les noms des attributs qu'on va enlever.
'''
supprimer = ['Total intl charge', 'Total night charge', 'Total eve charge', 'Total day charge', 'Number vmail messages']


# Corrélation entre 'Churn' et les autres attributs
print("\n# # # Correlation de Pearson avec Churn # # #")
#print(abonnes_num.corrwith(abonnes.iloc[:,19]))
abonnes_num = abonnes_num.drop(supprimer, axis=1)
print(abonnes_num.corrwith(abonnes.iloc[:,19]))
print("\n# # # Correlation de rang (Spearman) avec Churn # # #")
#print(abonnes_num.corrwith(abonnes.iloc[:,19], method='spearman'))
print(abonnes_num.corrwith(abonnes.iloc[:,19], method='spearman'))


# Visualisation de Churn
nbr_churn = abonnes['Churn'].value_counts()
nbr_churn[0] = round(nbr_churn[0]/abonnes.Churn.count(), 6)
nbr_churn[1] = round(nbr_churn[1]/abonnes.Churn.count(), 6)
ax = nbr_churn.plot.bar(x=nbr_churn, )
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_title('Churn')
ax.set_ylim((0,0.9))
for c in ax.containers:
  ratios = c.datavalues / c.datavalues.sum()
  ax.bar_label(c, labels=[f'{r:.1%}' for r in ratios])
#plt.show()

# Visualisation de ‘State’ vs ‘Churn’
fig, axz = plt.subplots(figsize=(15,15))
axz = sns.countplot(x='State', hue='Churn', data=abonnes, palette='Blues')
axz.set_ylabel('Nombre de Churn', size=20)
axz.set_xlabel('State', size=20)
axz.legend(loc=0, fontsize=20);
#plt.show()

# Filtrer les abonnés ayant Churn = 1 (True)
abonnes_churn = abonnes[abonnes.Churn == 1]
# Nombre d'abonnes_churn par State
print("# # # Nombre d'abonnés par State ayant Churn=1 # # #\n", abonnes_churn.State.value_counts())

# Visualisation de ‘Customer service calls’ vs ‘Churn’
fig, axz = plt.subplots(figsize=(5,3))
axz = sns.countplot(x='Customer service calls', hue='Churn', data=abonnes, palette='Reds')
axz.set_ylabel('Nombre de Churn', size=10)
axz.set_xlabel('Customer service calls', size=10)
axz.legend(loc=0, fontsize=10, labels=['False', 'True']);
#plt.show()


# Fichier d'entrainement
train = pd.read_csv("D:\Documents\Github\Telecom-churn\churn-bigml-80.csv")
print("\n# # # Entrainement # # #\n")
print(train.info())
print("\nValeurs uniques:\n", train.nunique())
col_bin = train.nunique()[train.nunique() == 2].keys().tolist()
# Encoder les 3 attributs en 0 ou 1
encodeur = LabelEncoder()
for i in col_bin:
  train[i] = encodeur.fit_transform(train[i])
print("\nApres la transformation de Entrainement :")
print(train.info())

# Fichier de test
test = pd.read_csv("D:\Documents\Github\Telecom-churn\churn-bigml-20.csv")
print("\n# # # Test # # #\n")
print(test.info())
print("Valeurs uniques:\n", test.nunique())
col_bin = test.nunique()[test.nunique() == 2].keys().tolist()
# Encoder les 3 attributs en 0 ou 1
encodeur = LabelEncoder()
for i in col_bin:
  test[i] = encodeur.fit_transform(test[i])
print("\nApres la transformation de Test :")
print(test.info())


#################################################################
# Fonction pour appeler tous les algorithmes de classification 
def classement(data, data_test):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier, plot_importance
    from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
    
    # Preparation des données
    # Données d’entrainement
    data = data[data["Churn"].notnull()] # supprimer les rangées qui l’attribut Churn manquant (NA)
    x, y = data.drop("Churn", axis = 1), data[["Churn"]]
    supprimer.append('State')
    #print("supprimer : ", supprimer)
    x = x.drop(supprimer, axis = 1) # supprimer les colonnes dont les noms sont dans la liste ‘supprimer’

    # Données de test
    data_test = data_test[data_test["Churn"].notnull()]
    x_test, y_test = data_test.drop("Churn", axis = 1), data_test[["Churn"]]
    x_test = x_test.drop(supprimer, axis = 1)

    # Les algorithmes
    KNC = KNeighborsClassifier()
    DTC = DecisionTreeClassifier()
    RFC = RandomForestClassifier()
    XGB = XGBClassifier()

    algorithmes = [KNC, DTC, RFC, XGB]
    noms_algo = ['KNeighborsClassifier', 'DecisionTreeClassifier', 'RandomForestClassifier', 'XGBClassifier']

    # Les métriques
    score_accuracy = []
    score_precision = []
    score_recall = []
    score_f1 = []

    # Apprentissage, prédiction et évaluation
    for item in algorithmes:
        item.fit(x, y)
        prediction = item.predict(x_test)
        score_accuracy.append(accuracy_score(y_test, prediction))
        score_precision.append(precision_score(y_test, prediction))
        score_recall.append(recall_score(y_test, prediction))
        score_f1.append(f1_score(y_test, prediction))

    # Importance des attributs
    plot_importance(XGB)
    plt.title('Importance d’attribut selon XGB')
    plt.grid(False)

    apprenti = RFC.fit(x,y)
    feature_importance = np.array(apprenti.feature_importances_).tolist()
    feature_names = list(x.columns)
    fi_df = pd.DataFrame({'feature_names':feature_names,'feature_importance':feature_importance})
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    plt.figure(figsize=(10,8))
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    plt.title('Importance d’attribut selon Random Forest')
    plt.xlabel('F score')
    plt.ylabel('Attributs')
    #plt.show()

    # créer matrice de confusion pour XGB
    fig = plt.figure(figsize=(5, 5))
    fig.set_facecolor("#F3F3F3")
    mat = confusion_matrix(y_test, XGB.predict(x_test))
    sns.heatmap(mat, annot=True, fmt = "d", square = True,
                xticklabels=["Non churn", "Churn"],
                yticklabels=["Non churn", "Churn"],
                linewidths = 2, linecolor = "w", cmap = "Set1")
    plt.title('Matrice de confusion XGB', color = "b")
    plt.subplots_adjust(wspace = .3, hspace = .3)

    # créer matrice de confusion pour RandomForest
    fig = plt.figure(figsize=(5, 5))
    fig.set_facecolor("#F3F3F3")
    mat = confusion_matrix(y_test, RFC.predict(x_test))
    sns.heatmap(mat, annot=True, fmt = "d", square = True,
                xticklabels=["Non churn", "Churn"],
                yticklabels=["Non churn", "Churn"],
                linewidths = 2, linecolor = "w", cmap = "Set1")
    plt.title('Matrice de confusion RFC', color = "b")
    plt.subplots_adjust(wspace = .3, hspace = .3)
    plt.show()


    # créer dataframe des résultats
    result = pd.DataFrame(columns = ['score_accuracy','score_f1', 'score_recall','score_precision'],index = noms_algo)
    result['score_accuracy'] = score_accuracy
    result['score_f1'] = score_f1
    result['score_recall'] = score_recall
    result['score_precision'] = score_precision
    print(result.sort_values('score_accuracy', ascending = False))
    return result.sort_values('score_accuracy', ascending = False)

# Appel de la fonction précédente
classement(train, test)
