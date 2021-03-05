import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
import script.views as views
import pickle


def model_cal(data, cols, model='ETR', data_types=['conv', 'prop1', 'prop2'], shap=True):
    cvf = KFold(n_splits=10, shuffle=True, random_state=1126)
    for type_ in data_types:
        os.makedirs('{type_}', exit_ok=True)
        print(type_)
        feat, target = data.loc[:, cols[type_]], data.loc[:, cols['target']]
        model = grid_search(feat, target, cvf, model)
        views.one_shot_plot(feat, target, model, xylim=[0, 35],
                            random_state=1126, save=f'{type_}/{model}')
        views.plot_importance(model, feat.columns, 20, save=f'{type_}/{model}')
        if shap:
            shap.initjs()
            shap_importance(model, feat, target, save=f'{type_}/{model}')
        pickle.dump(model, save=f'{type_}/{model}.binaryfile'))


def grid_search(feat, target, cvf, model = 'ETR'):
    if 'ETR' in model:
        cvmodel=GridSearchCV(ExtraTreesRegressor(n_jobs=1, random_state=1126),
                               param_grid = {"n_estimators": [250, 500, 1000]},
                               n_jobs = 5)
        crossvalid(feat, target, cvmodel, cvf)
        model=ExtraTreesRegressor(n_estimators = cvmodel.best_params_['n_estimtors'],
                                    n_jobs = -1, random_state = 1126)

    if 'XGB' in model:
        cvmodel=GridSearchCV(ExtraTreesRegressor(n_jobs=1, random_state=1126),
                               param_grid = {"n_estimators": [250, 500, 1000]},
                               n_jobs = 5)
        crossvalid(feat, target, cvmodel, cvf)
        model=XGBRegressor(n_estimators = cvmodel.best_params_['n_estimtors'],
                             n_jobs=-1, random_state=1126)

    return model


def crossvalid(xx, yy, model, cvf):
    err_trn = []
    err_tes = []
    r_2_tes = []
    r_2_trn = []
    for train_index, test_index in cvf.split(xx):
        x_trn = np.array(xx)[train_index]
        x_tes = np.array(xx)[test_index]
        y_trn = np.array(yy)[train_index]
        y_tes = np.array(yy)[test_index]
        model.fit(x_trn, y_trn)
        x_trn_pred = model.predict(x_trn)
        x_tes_pred = model.predict(x_tes)

        err_tes.append(mean_squared_error(x_tes_pred, y_tes))
        err_trn.append(mean_squared_error(x_trn_pred, y_trn))
        r_2_tes.append(r2_score(y_tes, x_tes_pred))
        r_2_trn.append(r2_score(y_trn, x_trn_pred))
    v_tes = np.sqrt(np.array(err_tes))
    v_trn = np.sqrt(np.array(err_trn))
    print("RMSE %1.3f (sd: %1.3f, min:%1.3f, max:%1.3f, det:%1.3f) ... train" % (
        v_trn.mean(), v_trn.std(), v_trn.min(), v_trn.max(), np.array(r_2_trn).mean()))
    print("RMSE %1.3f (sd: %1.3f, min:%1.3f, max:%1.3f, det:%1.3f) ... test" % (
        v_tes.mean(), v_tes.std(), v_tes.min(), v_tes.max(), np.array(r_2_tes).mean()))
    ret = {}
    ret['trn_mean'] = v_trn.mean()
    ret['trn_std'] = v_trn.std()
    ret['trn_r2'] = np.array(r_2_trn).mean()
    ret['tes_mean'] = v_tes.mean()
    ret['tes_std'] = v_tes.std()
    ret['tes_r2'] = np.array(r_2_tes).mean()
    return ret
