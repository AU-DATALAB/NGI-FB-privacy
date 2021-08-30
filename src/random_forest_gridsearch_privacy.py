from numpy.core.numeric import correlate
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier

def balance(dataframe, label, n=500):
    """
    Create a balanced sample from imbalanced datasets.
    
    dataframe: 
        Pandas dataframe with a column called 'text' and one called 'label'
    n:         
        Number of samples from each label, defaults to 500
    """
    # Use pandas select a random bunch of examples from each label
    import random
    random.seed(2022)
    out = (dataframe.groupby(label, as_index=False)
            .apply(lambda x: x.sample(n=n))
            .reset_index(drop=True))
    
    return out

def prep_data(data, dependent_var, predictor_list):
    '''
    data = entire dataframe, pandas DataFrame
    dependent_var = name of dependent variable, str
    predictor_list = list of predictor columns, list of str variables
    returns =  X_train, X_test, y_train, y_test
    '''
    # Split X and y
    X = np.array(data[predictor_list])
    y = data[dependent_var]
    # train and test
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    random_state=22,
                                                    train_size=.7, 
                                                    test_size=.3)
    return X_train, X_test, y_train, y_test

def rf_search_params(X_train, y_train):
    """
    Create a grid search of values in random forest
    
    X_train & y_train: 
        train sets
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10, 20, 40, 80]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 5]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune

    rf = RandomForestClassifier(n_estimators=100, random_state=2022)
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
    n_iter = 200, 
    cv = 3, 
    verbose=2, 
    random_state=2022, 
    n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X_train, y_train)

    return rf_random.best_params_, rf_random.best_estimator_

def model_evaluation(data,dependent_var, clf, predictor_list, X_train, y_train, X_test, y_test):
    from sklearn import metrics
    # Fit model
    clf.fit(X_train, y_train)
    # Predictions
    y_pred=clf.predict(X_test)
    # Accuracy
    acc = metrics.accuracy_score(y_test, y_pred)
    # Random classification
    import random
    random.seed(2022) 
    y_pred_random = data[dependent_var].sample(n=len(y_test))
    random_accuracy = metrics.accuracy_score(y_test, y_pred_random)
    print(f"[INFO] Predicting {dependent_var}: Empirical classification returns an accuracy of {round(acc, ndigits = 3)} and random classification returns an accuracy of {round(random_accuracy, ndigits = 3)}")
    # Important features
    import pandas as pd
    feature_imp = pd.Series(clf.feature_importances_, 
                            index=predictor_list).sort_values(ascending=False)
    
    return acc, random_accuracy, feature_imp

# Data
data = pd.read_csv("../../FACEBOOK/FB/CSVs/danish_p_gender_clean&ready.csv")

# Adding activity, network and age categories
data['activity'] = pd.qcut(data['post_comment_total'],q=3, labels=[1,2,3])
data['network'] = pd.qcut(data['total_unique_P_C'],q=3, labels=[1,2,3])
data['age'] = pd.qcut(data['new_days'],q=3, labels=[1,2,3])
data['privacy'] = data['privacy'].apply(lambda x : 0 if x == 'OPEN' else (1 if x == 'CLOSED' else 2))

#### Activity ####
data_balanced = balance(data, 'activity', n=1500)
dependent_var = 'activity'
predictor_list = ['dominant_topic', 'privacy', 'dominance', 'new_days']
X_train, X_test, y_train, y_test = prep_data(data_balanced, dependent_var, predictor_list)
clf = RandomForestClassifier(n_estimators=100, random_state=2022)
acc, random_accuracy, __ = model_evaluation(data,dependent_var, clf, predictor_list, X_train, y_train, X_test, y_test)
# Predicting activity: Empirical classification returns an accuracy of 0.427 and random classification returns an accuracy of 0.333
predictor_list = ['dominant_topic', 'privacy', 'dominance']
X_train, X_test, y_train, y_test = prep_data(data_balanced, dependent_var, predictor_list)
clf = RandomForestClassifier(n_estimators=100, random_state=2022)
acc, random_accuracy, feat_imp = model_evaluation(data,dependent_var, clf, predictor_list, X_train, y_train, X_test, y_test)
# Predicting activity without days Empirical classification returns an accuracy of 0.464 and random classification returns an accuracy of 0.333


#### Network ####
data_balanced = balance(data, 'network', n=1500)
dependent_var = 'network'
predictor_list = ['dominant_topic', 'privacy', 'dominance', 'new_days']
X_train, X_test, y_train, y_test = prep_data(data_balanced, dependent_var, predictor_list)
clf = RandomForestClassifier(n_estimators=100, random_state=2022)
acc, random_accuracy, feat_imp = model_evaluation(data,dependent_var, clf, predictor_list, X_train, y_train, X_test, y_test)
# Predicting network: Empirical classification returns an accuracy of 0.507 and random classification returns an accuracy of 0.316
predictor_list = ['dominant_topic', 'privacy', 'dominance']
X_train, X_test, y_train, y_test = prep_data(data_balanced, dependent_var, predictor_list)
clf = RandomForestClassifier(n_estimators=100, random_state=2022)
acc, random_accuracy, __ = model_evaluation(data,dependent_var, clf, predictor_list, X_train, y_train, X_test, y_test)
# Predicting network without new_days: Empirical classification returns an accuracy of 0.491 and random classification returns an accuracy of 0.336


#### Age ####
data_balanced = balance(data, 'age', n=1500)
dependent_var = 'age'
predictor_list = ['dominant_topic', 'privacy', 'dominance', 'total_unique_P_C']
X_train, X_test, y_train, y_test = prep_data(data_balanced, dependent_var, predictor_list)
clf = RandomForestClassifier(n_estimators=100, random_state=2022)
acc, random_accuracy, __ = model_evaluation(data,dependent_var, clf, predictor_list, X_train, y_train, X_test, y_test)
# Predicting age: Empirical classification returns an accuracy of 0.448 and random classification returns an accuracy of 0.305
predictor_list = ['dominant_topic', 'privacy', 'dominance', 'post_total']
X_train, X_test, y_train, y_test = prep_data(data_balanced, dependent_var, predictor_list)
clf = RandomForestClassifier(n_estimators=100, random_state=2022)
acc, random_accuracy, __ = model_evaluation(data,dependent_var, clf, predictor_list, X_train, y_train, X_test, y_test)
# Predicting age with post total: Empirical classification returns an accuracy of 0.497 and random classification returns an accuracy of 0.336
predictor_list = ['dominant_topic', 'privacy', 'dominance']
X_train, X_test, y_train, y_test = prep_data(data_balanced, dependent_var, predictor_list)
clf = RandomForestClassifier(n_estimators=100, random_state=2022)
acc, random_accuracy, __ = model_evaluation(data,dependent_var, clf, predictor_list, X_train, y_train, X_test, y_test)
# Predicting without post-total or unique post/comment: Empirical classification returns an accuracy of 0.448 and random classification returns an accuracy of 0.305

#### Privacy ####
data_balanced = balance(data, 'privacy', n=1500)
dependent_var = 'privacy'
predictor_list = ['dominant_topic', 'post_total', 'new_days', 'dominance']
X_train, X_test, y_train, y_test = prep_data(data_balanced, dependent_var, predictor_list)
clf = RandomForestClassifier(n_estimators=100, random_state=2022)
acc, random_accuracy, __ = model_evaluation(data,dependent_var, clf, predictor_list, X_train, y_train, X_test, y_test)
# Predicting privacy: Empirical classification returns an accuracy of 0.327 and random classification returns an accuracy of 0.333
predictor_list = ['dominant_topic', 'total_unique_P_C', 'new_days', 'dominance']
X_train, X_test, y_train, y_test = prep_data(data_balanced, dependent_var, predictor_list)
clf = RandomForestClassifier(n_estimators=100, random_state=2022)
acc, random_accuracy, imp_feats = model_evaluation(data,dependent_var, clf, predictor_list, X_train, y_train, X_test, y_test)
# Predicting privacy with unique post: Empirical classification returns an accuracy of 0.356 and random classification returns an accuracy of 0.332

#### Gender ####
data_balanced = balance(data, 'dominance', n=1500)
dependent_var = 'dominance'
predictor_list = ['dominant_topic', 'post_total','new_days', 'privacy']
X_train, X_test, y_train, y_test = prep_data(data_balanced, dependent_var, predictor_list)
clf = RandomForestClassifier(n_estimators=100, random_state=2022)
acc, random_accuracy, __ = model_evaluation(data,dependent_var, clf, predictor_list, X_train, y_train, X_test, y_test)
# Predicting dominance: Empirical classification returns an accuracy of 0.511 and random classification returns an accuracy of 0.321
predictor_list = ['dominant_topic', 'total_unique_P_C','new_days', 'privacy']
X_train, X_test, y_train, y_test = prep_data(data_balanced, dependent_var, predictor_list)
clf = RandomForestClassifier(n_estimators=100, random_state=2022)
acc, random_accuracy, __ = model_evaluation(data,dependent_var, clf, predictor_list, X_train, y_train, X_test, y_test)
# Predicting dominance with unique posts: Empirical classification returns an accuracy of 0.523 and random classification returns an accuracy of 0.321


#%%
## Correlation test
import seaborn as sns
import pandas as pd
data = pd.read_csv("../../FACEBOOK/FB/CSVs/danish_p_gender_clean&ready.csv")

num_data = data[['post_total', 'new_days', 'total_unique_P_C']]
corr = num_data.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
from scipy import stats
from scipy.stats import chi2_contingency

# Chi square test
tab = pd.crosstab(data.dominant_topic, data.privacy)
chi2, p, df, exp = chi2_contingency(tab, correction=False) # chi2: the test statistic, p:the p-value of the test, dof:degrees of freedom

# Chi square test
tab = pd.crosstab(data.dominance, data.dominant_topic)
chi2, p, df, exp = chi2_contingency(tab, correction=False) # chi2: the test statistic, p:the p-value of the test, dof:degrees of freedom

# Chi square test
tab = pd.crosstab(data.dominance, data.privacy)
chi2, p, df, exp = chi2_contingency(tab, correction=False) # chi2: the test statistic, p:the p-value of the test, dof:degrees of freedom


# Anova
F, p = stats.f_oneway(data[data.privacy=='OPEN'].post_total,
                      data[data.privacy=='CLOSED'].post_total,
                      data[data.privacy=='SECRET'].post_total)

# Anova
F, p = stats.f_oneway(data[data.privacy=='OPEN'].total_unique_P_C,
                      data[data.privacy=='CLOSED'].total_unique_P_C,
                      data[data.privacy=='SECRET'].total_unique_P_C)

F, p = stats.f_oneway(data[data.privacy=='OPEN'].new_days,
                      data[data.privacy=='CLOSED'].new_days,
                      data[data.privacy=='SECRET'].new_days)


# Anova
F, p = stats.f_oneway(data[data.dominance==1].post_total,
                      data[data.dominance==2].post_total,
                      data[data.dominance==3].post_total)

# Anova
F, p = stats.f_oneway(data[data.dominance==1].total_unique_P_C,
                      data[data.dominance==2].total_unique_P_C,
                      data[data.dominance==3].total_unique_P_C)

F, p = stats.f_oneway(data[data.dominance==1].new_days,
                      data[data.dominance==2].new_days,
                      data[data.dominance==3].new_days)


# Anova
F, p = stats.f_oneway(data[data.dominant_topic==1].post_total,
                      data[data.dominant_topic==2].post_total,
                      data[data.dominant_topic==3].post_total,
                      data[data.dominant_topic==4].post_total,
                      data[data.dominant_topic==5].post_total,
                      data[data.dominant_topic==6].post_total,
                      data[data.dominant_topic==7].post_total,
                      data[data.dominant_topic==8].post_total,
                      data[data.dominant_topic==9].post_total,
                      data[data.dominant_topic==10].post_total,
                      data[data.dominant_topic==11].post_total,
                      data[data.dominant_topic==12].post_total,
                      data[data.dominant_topic==13].post_total,
                      data[data.dominant_topic==14].post_total,
                      data[data.dominant_topic==15].post_total,
                      data[data.dominant_topic==16].post_total,
                      data[data.dominant_topic==17].post_total,
                      data[data.dominant_topic==18].post_total,
                      data[data.dominant_topic==19].post_total,
                      data[data.dominant_topic==20].post_total,
                      data[data.dominant_topic==21].post_total,
                      data[data.dominant_topic==22].post_total,
                      data[data.dominant_topic==23].post_total,
                      data[data.dominant_topic==24].post_total,
                      data[data.dominant_topic==25].post_total,
                      data[data.dominant_topic==26].post_total,
                      data[data.dominant_topic==27].post_total,
                      data[data.dominant_topic==28].post_total,
                      data[data.dominant_topic==29].post_total,
                      data[data.dominant_topic==30].post_total,
                      data[data.dominant_topic==31].post_total,
                      data[data.dominant_topic==32].post_total,
                      data[data.dominant_topic==33].post_total,
                      data[data.dominant_topic==34].post_total,
                      data[data.dominant_topic==35].post_total,
                      data[data.dominant_topic==36].post_total,
                      data[data.dominant_topic==37].post_total,
                      data[data.dominant_topic==38].post_total,
                      data[data.dominant_topic==39].post_total,
                      data[data.dominant_topic==40].post_total,
                      data[data.dominant_topic==41].post_total,
                      data[data.dominant_topic==42].post_total,
                      data[data.dominant_topic==43].post_total,
                      data[data.dominant_topic==44].post_total,
                      data[data.dominant_topic==45].post_total,
                      data[data.dominant_topic==46].post_total,
                      data[data.dominant_topic==47].post_total,
                      data[data.dominant_topic==48].post_total,
                      data[data.dominant_topic==49].post_total,
                      data[data.dominant_topic==50].post_total)


# Anova
F, p = stats.f_oneway(data[data.dominant_topic==1].total_unique_P_C,
                      data[data.dominant_topic==2].total_unique_P_C,
                      data[data.dominant_topic==3].total_unique_P_C,
                      data[data.dominant_topic==4].total_unique_P_C,
                      data[data.dominant_topic==5].total_unique_P_C,
                      data[data.dominant_topic==6].total_unique_P_C,
                      data[data.dominant_topic==7].total_unique_P_C,
                      data[data.dominant_topic==8].total_unique_P_C,
                      data[data.dominant_topic==9].total_unique_P_C,
                      data[data.dominant_topic==10].total_unique_P_C,
                      data[data.dominant_topic==11].total_unique_P_C,
                      data[data.dominant_topic==12].total_unique_P_C,
                      data[data.dominant_topic==13].total_unique_P_C,
                      data[data.dominant_topic==14].total_unique_P_C,
                      data[data.dominant_topic==15].total_unique_P_C,
                      data[data.dominant_topic==16].total_unique_P_C,
                      data[data.dominant_topic==17].total_unique_P_C,
                      data[data.dominant_topic==18].total_unique_P_C,
                      data[data.dominant_topic==19].total_unique_P_C,
                      data[data.dominant_topic==20].total_unique_P_C,
                      data[data.dominant_topic==21].total_unique_P_C,
                      data[data.dominant_topic==22].total_unique_P_C,
                      data[data.dominant_topic==23].total_unique_P_C,
                      data[data.dominant_topic==24].total_unique_P_C,
                      data[data.dominant_topic==25].total_unique_P_C,
                      data[data.dominant_topic==26].total_unique_P_C,
                      data[data.dominant_topic==27].total_unique_P_C,
                      data[data.dominant_topic==28].total_unique_P_C,
                      data[data.dominant_topic==29].total_unique_P_C,
                      data[data.dominant_topic==30].total_unique_P_C,
                      data[data.dominant_topic==31].total_unique_P_C,
                      data[data.dominant_topic==32].total_unique_P_C,
                      data[data.dominant_topic==33].total_unique_P_C,
                      data[data.dominant_topic==34].total_unique_P_C,
                      data[data.dominant_topic==35].total_unique_P_C,
                      data[data.dominant_topic==36].total_unique_P_C,                  
                      data[data.dominant_topic==37].total_unique_P_C,
                      data[data.dominant_topic==38].total_unique_P_C,
                      data[data.dominant_topic==39].total_unique_P_C,
                      data[data.dominant_topic==40].total_unique_P_C,
                      data[data.dominant_topic==41].total_unique_P_C,
                      data[data.dominant_topic==42].total_unique_P_C,
                      data[data.dominant_topic==43].total_unique_P_C,
                      data[data.dominant_topic==44].total_unique_P_C,
                      data[data.dominant_topic==45].total_unique_P_C,
                      data[data.dominant_topic==46].total_unique_P_C,
                      data[data.dominant_topic==47].total_unique_P_C,
                      data[data.dominant_topic==48].total_unique_P_C,
                      data[data.dominant_topic==49].total_unique_P_C,
                      data[data.dominant_topic==50].total_unique_P_C)

F, p = stats.f_oneway(data[data.dominant_topic==1].new_days,
                      data[data.dominant_topic==2].new_days,
                      data[data.dominant_topic==3].new_days,
                      data[data.dominant_topic==4].new_days,
                      data[data.dominant_topic==5].new_days,
                      data[data.dominant_topic==6].new_days,
                      data[data.dominant_topic==7].new_days,
                      data[data.dominant_topic==8].new_days,
                      data[data.dominant_topic==9].new_days,
                      data[data.dominant_topic==10].new_days,
                      data[data.dominant_topic==11].new_days,
                      data[data.dominant_topic==12].new_days,
                      data[data.dominant_topic==13].new_days,
                      data[data.dominant_topic==14].new_days,
                      data[data.dominant_topic==15].new_days,
                      data[data.dominant_topic==16].new_days,
                      data[data.dominant_topic==17].new_days,
                      data[data.dominant_topic==18].new_days,
                      data[data.dominant_topic==19].new_days,
                      data[data.dominant_topic==20].new_days,
                      data[data.dominant_topic==21].new_days,
                      data[data.dominant_topic==22].new_days,
                      data[data.dominant_topic==23].new_days,
                      data[data.dominant_topic==24].new_days,
                      data[data.dominant_topic==25].new_days,
                      data[data.dominant_topic==26].new_days,
                      data[data.dominant_topic==27].new_days,
                      data[data.dominant_topic==28].new_days,
                      data[data.dominant_topic==29].new_days,
                      data[data.dominant_topic==30].new_days,
                      data[data.dominant_topic==31].new_days,
                      data[data.dominant_topic==32].new_days,
                      data[data.dominant_topic==33].new_days,
                      data[data.dominant_topic==34].new_days,
                      data[data.dominant_topic==35].new_days,
                      data[data.dominant_topic==36].new_days,                  
                      data[data.dominant_topic==37].new_days,
                      data[data.dominant_topic==38].new_days,
                      data[data.dominant_topic==39].new_days,
                      data[data.dominant_topic==40].new_days,
                      data[data.dominant_topic==41].new_days,
                      data[data.dominant_topic==42].new_days,
                      data[data.dominant_topic==43].new_days,
                      data[data.dominant_topic==44].new_days,
                      data[data.dominant_topic==45].new_days,
                      data[data.dominant_topic==46].new_days,
                      data[data.dominant_topic==47].new_days,
                      data[data.dominant_topic==48].new_days,
                      data[data.dominant_topic==49].new_days,
                      data[data.dominant_topic==50].new_days)




# %%
