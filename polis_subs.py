# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sklearn.preprocessing as skp
import sklearn.feature_extraction as skf
import scipy.sparse as ss
import sklearn.model_selection as sms
import sklearn.linear_model as slm
import tqdm 
import sklearn.neighbors as skn
import sklearn.ensemble as se
import seaborn as sns


renesans=pd.read_csv("/Users/sveta/Downloads/Задача 1/Book1.csv",sep=",",header=0)



renesans=pd.DataFrame(renesans)


# lets have a look into our data
renesans["POLICY_ID"].unique()
renesans["POLICY_BEGIN_MONTH"].unique()
renesans["POLICY_END_MONTH"].unique()
renesans["POLICY_SALES_CHANNEL"].unique() #много каналов
renesans["POLICY_SALES_CHANNEL_GROUP"].unique() #less chanels, probably will use it
renesans["POLICY_BRANCH"].unique() # Moscow and St-Peter
renesans["POLICY_MIN_DRIVING_EXPERIENCE"].unique() #needs to be cleaned. update: done
renesans["POLICY_MIN_AGE"].unique()
renesans["VEHICLE_MAKE"].unique() #car brands
renesans["VEHICLE_MODEL"].unique() #car model
renesans["VEHICLE_ENGINE_POWER"].unique()
renesans["VEHICLE_IN_CREDIT"].unique()
renesans["VEHICLE_SUM_INSURED"].unique()
renesans["POLICY_INTERMEDIARY"].unique()
renesans["INSURER_GENDER"].unique()
renesans["POLICY_CLM_N"].unique() #количество убытков по полису
renesans["POLICY_CLM_GLT_N"].unique() #hell knows
renesans["POLICY_COURT_SIGN"].unique() #hell knows
renesans["POLICY_PRV_CLM_N"].unique() # hell knows 0/1
renesans["POLICY_PRV_CLM_GLT_N"].unique() 
renesans["CLAIM_AVG_ACC_ST_PRD"].unique() # avg claim idk
renesans["POLICY_HAS_COMPLAINTS"].unique() #bi var
renesans["POLICY_YEARS_RENEWED_N"].unique() #what is N???
renesans["POLICY_DEDUCT_VALUE"].unique() #smth numerical
renesans["CLIENT_REGISTRATION_REGION"].unique() #regions
renesans["POLICY_PRICE_CHANGE"].unique() #price change num


#так как данные грязные и у нас год начала стажа и количество лет стажа в одной колонке, мы фильтруем 
# годы и вычитаем из из 2018, чтобы получить количество лет стажа
renesans.loc[renesans["POLICY_MIN_DRIVING_EXPERIENCE"]>100,"POLICY_MIN_DRIVING_EXPERIENCE"]=2018-renesans.loc[renesans["POLICY_MIN_DRIVING_EXPERIENCE"]>100,"POLICY_MIN_DRIVING_EXPERIENCE"]

# делаем длительность
renesans["POLICY_END_MONTH"]='2019'+'-'+renesans["POLICY_END_MONTH"].map(str)+'-'+'01'
renesans["POLICY_END_MONTH"]=pd.to_datetime(renesans["POLICY_END_MONTH"])

renesans["POLICY_BEGIN_MONTH"]='2018'+'-'+renesans["POLICY_BEGIN_MONTH"].map(str)+'-'+'01'
renesans["POLICY_BEGIN_MONTH"]=pd.to_datetime(renesans["POLICY_BEGIN_MONTH"])

renesans["POLICY_LEN"]=renesans["POLICY_END_MONTH"]-renesans["POLICY_BEGIN_MONTH"]
renesans["POLICY_LEN"]=renesans["POLICY_LEN"].dt.days/30
renesans["POLICY_LEN"]=renesans["POLICY_LEN"].round(0)
renesans["POLICY_LEN"]=[1 if (x==12) else 0 for x in renesans["POLICY_LEN"].values]

renesans["POLICY_LEN"].unique()

### продолжим готовить данные
renesans["POLICY_BRANCH"]=[1 if (x=='Москва') else 0 for x in renesans["POLICY_BRANCH"].values]
renesans["INSURER_GENDER"]=[1 if (x=='F') else 0 for x in renesans["INSURER_GENDER"].values]

#избавимся от Na
renesans.drop(renesans.loc[renesans['POLICY_YEARS_RENEWED_N']=='N'].index, inplace=True)


#что войдёт в анализ

policyidtest=renesans.loc[renesans['DATA_TYPE']=='TEST ','POLICY_ID']

renesans=renesans.drop(['POLICY_ID','POLICY_BEGIN_MONTH', 'POLICY_END_MONTH','POLICY_SALES_CHANNEL','POLICY_INTERMEDIARY'],axis=1)


### и готовимся к перекодировке

standard = skp.StandardScaler()
maxabs = skp.MaxAbsScaler()
label = skp.LabelEncoder()
onehot = skp.OneHotEncoder()
season = skp.LabelBinarizer()
labelbin = skp.LabelBinarizer()

absvar = ['VEHICLE_ENGINE_POWER', 'VEHICLE_SUM_INSURED']
scalevar = ['POLICY_MIN_AGE', 'POLICY_MIN_DRIVING_EXPERIENCE', 'CLAIM_AVG_ACC_ST_PRD', 
                'POLICY_YEARS_RENEWED_N', 'POLICY_DEDUCT_VALUE', 'POLICY_PRICE_CHANGE'] 
bivar = ['POLICY_BRANCH', 'VEHICLE_IN_CREDIT', 'CLIENT_HAS_DAGO', 'CLIENT_HAS_OSAGO',
         'POLICY_COURT_SIGN', 'POLICY_HAS_COMPLAINTS', 'INSURER_GENDER','POLICY_LEN']
onehotvar = ['POLICY_SALES_CHANNEL_GROUP']
# 'POLICY_CLM_N', 'POLICY_CLM_GLT_N', 'POLICY_PRV_CLM_N','POLICY_PRV_CLM_GLT_N' идут по отдельности


### нормализуем

renesans_abs=maxabs.fit_transform(renesans[absvar])
renesans_scale = standard.fit_transform(renesans[scalevar])
renesans_bi = renesans[bivar]
renesans_onehot = onehot.fit_transform(renesans[onehotvar])
renesans_N = labelbin.fit_transform(renesans['POLICY_CLM_N'])
renesans_GLT_N = labelbin.transform(renesans['POLICY_CLM_GLT_N'])
renesans_PRV_N = labelbin.transform(renesans['POLICY_PRV_CLM_N'])
renesans_PRV_GLT_N = labelbin.transform(renesans['POLICY_PRV_CLM_GLT_N'])

reg=['CLIENT_REGISTRATION_REGION']
Vreg = skf.DictVectorizer()
renesans_reg = Vreg.fit_transform(renesans[reg].fillna('-').T.to_dict().values())

cars = ['VEHICLE_MAKE', 'VEHICLE_MODEL']
Vcars = skf.DictVectorizer()
renesans_cars = Vcars.fit_transform(renesans[cars].fillna('-').T.to_dict().values())

####

R_train_ = ss.hstack([renesans_abs, renesans_scale, renesans_bi, renesans_onehot,  renesans_N, 
                                renesans_GLT_N, renesans_PRV_N, renesans_PRV_GLT_N, renesans_reg, renesans_cars])

## в предыдущем варианте кода под самый конец вылезла ошибка - валидационная выборка была меньше, чем тестова,
# потому что туда не вошли некоторые значения переменных, как следствие, недосоздались столбцы. 
# Пришлось извращаться - тк у нас не было сортировок, то порядок всюду сохраняется. 
# Поэтому я вытащу номера строк тренировочной выборки, а потом тестовой.
# Почле чего отфильтрую нужные строки по выборкам

renesans_DT=np.array([1 if (x=='TRAIN') else 0 for x in renesans["DATA_TYPE"].values])

Train_rows=(renesans_DT==1).nonzero()[0]
Test_rows=(renesans_DT==0).nonzero()[0]

# делим выборки

R_train_all=R_train_.tocsc()[Train_rows]
R_test_all=R_train_.tocsc()[Test_rows]

y_train = renesans.loc[renesans['DATA_TYPE']=='TRAIN','POLICY_IS_RENEWED']

R_train_2, R_valid, y_train_2, y_valid = sms.train_test_split(R_train_all, y_train,  test_size = 0.2, 
                                                              random_state = 1)
# мера качества моделей - сумма true positive и true negative
# больше можно посмотреть тут https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234

### Начнём с классики. logit классификатор.

stkf = sms.StratifiedKFold(n_splits = 5, random_state = 1, shuffle = True)

C_space = np.logspace(-3, 2, 6)
for c in tqdm.tqdm(C_space):lr = slm.LogisticRegression(C = c, random_state = 1)
print(c, sms.cross_val_score(lr, R_train_2, y_train_2, scoring='accuracy', cv=stkf).mean())

#  я не возлагаю много надежд на логит, поэтому сразу подтюним его

### ЛР
lr = slm.LogisticRegression(C = 0.1, random_state = 1)
bg_lr = se.BaggingClassifier(base_estimator = lr, n_estimators = 100, random_state = 1, n_jobs=1)

params = {'max_features': [3,6,12,24,48,96,192,384], 
          'max_samples': [0.5, 0.75, 0.9]}
rs_lr = sms.RandomizedSearchCV(estimator = bg_lr, n_jobs = 2, cv = stkf, verbose = 2, 
                              param_distributions = params, scoring = 'accuracy', n_iter = 20, random_state=1)
rs_lr.fit(R_train_2, y_train_2)
print(rs_lr.best_score_, rs_lr.best_params_) #0.6651854963805585 {'max_samples': 0.9, 'max_features': 384}


lr = slm.LogisticRegression(C = 0.1, random_state = 1)
bg_lr = se.BaggingClassifier(base_estimator = lr, n_estimators = 10, random_state = 1, n_jobs=2, max_features=0.7)
print(sms.cross_val_score(bg_lr, R_train_2, y_train_2, scoring='accuracy', cv=stkf).mean()) #0.6688857348064529


###далее попробуем KNN, однако уберем все категореальные признаки, чтобы найти нужный размер выборки и гиперпараметры для уменьшения времени, а потом багганём 
#(зачем это надо см https://www.quora.com/What-is-bagging-in-machine-learning)

R_train_knn = ss.hstack([renesans_abs, renesans_scale, renesans_bi])
R_train_knn=R_train_knn.tocsc()[Train_rows]
R_train_2, R_valid, y_train_2, y_valid = sms.train_test_split(R_train_knn, y_train,  test_size = 0.2, random_state = 1)
R_train_2_short = R_train_2[:10000,:]
y_train_2_short = y_train_2[:10000]

knn = skn.KNeighborsClassifier()
clf = sms.GridSearchCV(estimator = knn, n_jobs = 1, cv = stkf, return_train_score = True, verbose = 1, 
                       param_grid = {"n_neighbors": [1,3,5,10,20,50], "weights": ["uniform", "distance"]})
clf.fit(R_train_2_short, y_train_2_short)
clf.cv_results_['mean_test_score'].mean()#0.61
clf.best_params_#n_neighbors': 50, 'weights': 'uniform'
#
clf = sms.GridSearchCV(estimator = knn, n_jobs = 1, cv = stkf, return_train_score = True, verbose = 0,
                   param_grid = {"n_neighbors":[30,70,100,150], "weights": ['uniform', 'distance']})
clf.fit(R_train_2_short, y_train_2_short)
clf.best_score_ #best score
clf.best_params_ # best parametrs

#
knn = skn.KNeighborsClassifier(n_neighbors = 100, weights = 'distance')
R_train_all = ss.hstack([renesans_abs, renesans_scale, renesans_bi, renesans_N,renesans_onehot, 
                         renesans_GLT_N, renesans_PRV_N, renesans_PRV_GLT_N])
R_train_knn=R_train_.tocsc()[Train_rows]
R_train_2, R_valid, y_train_2, y_valid = sms.train_test_split(R_train_knn, y_train,  test_size = 0.2, random_state = 1)

R_train_2_short = R_train_2[:10000,:]
y_train_2_short = y_train_2[:10000]

print(sms.cross_val_score(knn, R_train_2_short, y_train_2_short, scoring='accuracy', cv=stkf).mean()) #0.65 растём-с

### 
R_train_all = ss.hstack([renesans_abs, renesans_scale, renesans_bi, renesans_onehot,  renesans_N, 
                                renesans_GLT_N, renesans_PRV_N, renesans_PRV_GLT_N, renesans_reg, renesans_cars])
R_train_all=R_train_all.tocsc()[Train_rows]

R_train_2, R_valid, y_train_2, y_valid = sms.train_test_split(R_train_all, y_train,  test_size = 0.2, random_state = 1)
R_train_2_short = R_train_2[:10000,:]
y_train_2_short = y_train_2[:10000]

print(sms.cross_val_score(knn, R_train_2_short, y_train_2_short, scoring='accuracy', cv=stkf).mean())

bg = se.BaggingClassifier(base_estimator = knn, max_samples = 10000, random_state = 1, verbose = 1)
print(sms.cross_val_score(bg, R_train_2, y_train_2, scoring='accuracy', cv=stkf).mean())# 0.658

### делаем RF, гиперпараметр - ДЖини, см https://www.quora.com/Machine-Learning/Are-gini-index-entropy-or-classification-error-measures-causing-any-difference-on-Decision-Tree-classification\

rf = se.RandomForestClassifier(random_state = 1, n_estimators=100, max_depth=1000, oob_score=True, class_weight='balanced')
print(sms.cross_val_score(rf, R_train_2, y_train_2, scoring='accuracy', cv=stkf).mean()) #0.697 лучшее, что пока есть

## Улучшаем RF

params = {"max_depth": [50, 150, 550, 800, 1500], 
         "min_samples_leaf": [1, 3, 5, 8],
         "max_features": [2, 5, 15, 40, 100]}
rs = sms.RandomizedSearchCV(estimator = rf, n_jobs = 2, cv = stkf, verbose = 2, param_distributions = params,
                        scoring = 'accuracy', n_iter = 20)
rs.fit(R_train_2, y_train_2)
print(rs.best_score_, rs.best_params_) #0.7083117890382626 {'min_samples_leaf': 1, 'max_features': 100, 'max_depth': 300}


### Градиентный бустинг
# проверь параметры!!!!!!!!!

params = {'n_estimators': [100, 400, 700, 1000],
              'max_depth': [2, 4, 6, 8, 10],
              'min_samples_leaf': [1, 2, 3, 5],
              'max_features': [2, 4, 8, 16, 32, 64, 128]}
gb = se.GradientBoostingClassifier(random_state = 1)
rs_gb = sms.RandomizedSearchCV(estimator = gb, n_jobs = 2, cv = stkf, verbose = 2, param_distributions = params,
                               scoring = 'accuracy', n_iter = 50, random_state=1)
rs_gb.fit(R_train_2, y_train_2)
print(rs_gb.best_score_, rs_gb.best_params_)#0.7157607290589452 {'n_estimators': 700, 'min_samples_leaf': 2, 'max_features': 128, 'max_depth': 4}

#У нас есть 4 модели, обучим их на тренировочной выборке: проверим accuracy, получим предсказания на валидационной выборке и построим модель 2-ого уровня

###ПРОВЕРЬ ПАРАМЕТРЫ!!!!!!
rf = se.RandomForestClassifier(random_state = 1, n_estimators=100, max_depth=300, oob_score=True,
                               class_weight='balanced', max_features = 100)
gb = se.GradientBoostingClassifier(random_state = 1, n_estimators = 700, min_samples_leaf = 2, max_depth = 4,
                                   max_features = 128)
lr.fit(R_train_2, y_train_2)
print("lr:", lr.score(R_valid, y_valid)) #0.6700058165837265
bg.fit(R_train_2, y_train_2)
print("bg:", bg.score(R_valid, y_valid))#0.6592128223356815
rf.fit(R_train_2, y_train_2)
print("rf:", rf.score(R_valid, y_valid))#0.7093000710915789
gb.fit(R_train_2, y_train_2)
print("gb:", gb.score(R_valid, y_valid))#0.718606605053965

#больших отличий от кросс-валидации нет, переобучения или недообучения тоже вроде нет, 
# так как скоринг не зашкаливает, но и не критично низкий
# Далее, надо получить вероятности классов

pred_lr = lr.predict_proba(R_valid)[:,1]
pred_bg = bg.predict_proba(R_valid)[:,1]
pred_rf = rf.predict_proba(R_valid)[:,1]
pred_gb = gb.predict_proba(R_valid)[:,1]


# "усредним" все результаты с помощью логистической регресс (используем мета-алгоритм)
meta_features = [pred_lr, pred_bg, pred_rf, pred_gb]
meta_X_valid = pd.DataFrame(meta_features).T
meta_X_valid.columns = ['lr', 'bg', 'rf', 'gb']
meta_X_valid.head()

meta_lr = slm.LogisticRegression(random_state = 1)
print(sms.cross_val_score(lr, meta_X_valid, y_valid, scoring='accuracy', cv=stkf).mean())
meta_lr.fit(meta_X_valid, y_valid)

##
pred_lr = lr.predict_proba(R_test_all)[:,1]
pred_bg = bg.predict_proba(R_test_all)[:,1]
pred_rf = rf.predict_proba(R_test_all)[:,1]
pred_gb = gb.predict_proba(R_test_all)[:,1]
meta_features = [pred_lr, pred_bg, pred_rf, pred_gb]
meta_X_test = pd.DataFrame(meta_features).T
meta_X_test.columns = ['lr', 'bg', 'rf', 'gb']
prediction = meta_lr.predict(meta_X_test)

prediction=pd.DataFrame(prediction)
prediction.columns=['prediction']
outfile=pd.DataFrame(prediction,policyidtest)

# Выгружаем
outfile.to_csv('/Users/sveta/Downloads/Задача 1/prediction.csv')

### картинка

sns.set(style="darkgrid")
ax = sns.countplot(x=outfile['prediction'], data=outfile)
ax.figure.savefig('/Users/sveta/Downloads/Задача 1/output.png')
# Добби свободен