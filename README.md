# SCI 1402 Projet en science des données
Travail noté 3

Nom du projet : Telecom-churn

---


# Introduction

Ce projet utilise le jeu de données [Telecom Churn Dataset](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets/data). Il contient des paramètres des abonnés d’un réseau de télécommunication.

L’objectif est d’analyser les données du réseau et de construire un modèle qui prédit quels abonnés ont de grandes chances de se désabonner pour changer de réseau. Ces prédictions peuvent aider la gestion de ce réseau à prendre des décisions/actions pour minimiser les causes de désabonnement et garder le maximum d’abonnés. En général, le coût d’acquérir un nouvel abonné est beaucoup plus que celui de garder un abonné actuel.

C’est un apprentissage supervisé ayant la variable réponse dans la colonne « Churn ».

Le répertoire GitHub de ce projet est https://github.com/dataminer-2021/Telecom-churn.


# La description du jeu de données (dataset)

Il y a deux fichiers de données :
* churn-bigml-80.csv qui contient les données d’entrainement.
* churn-bigml-20.csv qui contient les données de test.

## Description des attributs
En combinant les 2 fichiers de données, on constate que :
1. Il y a 20 attributs (colonnes) :
	* 8 attributs ont le type float64.
	* 8 attributs ont le type int64.
	* 3 attributs ont le type objet.
	* 1 attribut a le type booléen.
3. Il y a 3333 abonnés (rangées).
4. La signification des attributs :
	* **State** : l’abréviation du nom de l’état (des États Unis) où l’abonne réside
	* **Account length** : le nombre de jours pendant lesquels le compte est actif
	* **Area code** : l’indicatif régional du numéro de téléphone de l’abonné
	* **International plan** : si l’abonné a un forfait d’appels internationaux
	* **Voice mail plan** : si l’abonné a le service de la messagerie vocale
	* **Number vmail messages** : le nombre moyen de messages vocaux par mois
	* **Total day minutes** : le nombre total de minutes d'appel utilisées pendant la journée
	* **Total day calls** : le nombre total d'appels passés pendant la journée
	* **Total day charge** : le coût facturé des appels de la journée
	* **Total eve minutes** : le nombre total de minutes d'appel utilisées pendant la soirée
	* **Total eve calls** : le nombre total d'appels passés pendant la soirée
	* **Total eve charge** : le coût facturé des appels de la soirée
	* **Total night minutes** : le nombre total de minutes d'appel utilisées pendant la nuit
	* **Total night calls** : le nombre total d'appels passés pendant la nuit
	* **Total night charge** : le coût facturé des appels de la nuit
	* **Total intl minutes** : le nombre total de minutes internationales
	* **Total intl calls** : le nombre total d’appels internationaux
	* **Total intl charge** : le coût facturé des appels internationaux
	* **Customer service calls** : le nombre d'appels passés au service client
	* **Churn** : si l’abonné a quitté le service.
5. La 20<sup>e</sup>  colonne/attribut (Churn) est la variable réponse qu’il faut prédire.

# Le problème et l’objectif

L’entreprise Orange télécom a besoin de minimiser le nombre des abonnés qui se désabonnent de leurs services, ou quittent le réseau pour un autre. Les données fournies par l’entreprise sont celles d’environ 3300 abonnés, dont une partie se sont désabonné. Minimiser le nombre de désabonnement (Churn en anglais) correspond à minimiser le coût d’augmenter le nombre d’abonnés.
L’objectif est de construire un modèle de prédiction qui peut alerter les responsables à Orange télécom quand un ou plusieurs abonnés risquent de se désabonner. Ces responsables devront avoir un plan à exécuter pour retenir le maximum de ces abonnés.

# La préparation des données

Les attributs non-numeriques sont :
* State (object)
* International plan (object)
* Voice mail plan (object)
* Churn (boolean)

En voyant les valeurs uniques de chaque attribut, on remarque que 3 des 4 attributs précédents ont 2 valeurs uniques chacun : International plan, Voice mail plan et Churn.
Pour les calculs statistiques, on transforme ces 3 attributs pour avoir des valeurs numériques (binaires).

# Corrélation
## La matrice de corrélation de Pearson (heatmap)

La matrice de corrélation de Pearson montre que l’attribut ‘Churn’ a une corrélation faible avec tous les autres attributs.
Elle montre aussi qu'il y a 5 couples d'attributs qui sont fortement corrélés :
1.	Total intl minutes et Total intl charge
2.	Total night minutes et Total night charge
3.	Total eve minutes et Total eve charge
4.	Total day minutes et Total day charge
5.	Voice mail plan et Number vmail messages

Alors, on garde le premier attribut seulement de chaque couple pendant la modélisation. 

## La corrélation de Pearson entre Churn et les autres attributs
Cette corrélation montre qu’il y a une corrélation faible entre ‘Churn’ et tous les attributs.

## La corrélation de range (Spearman) entre Churn et les autres attributs
Cette corrélation montre qu’il y a une corrélation faible entre ‘Churn’ et tous les attributs.

# Observations
* Les états qui ont les plus grands nombres d’abonnés qui ont quitté le service sont : New Jersey, Texas, Maryland, Michigan, Minnesota and New York.
* Il y a des abonnés qui ont changé de réseau après un seul appel au service client, et même il y en a qui ne l’ont pas appelé du tout.
