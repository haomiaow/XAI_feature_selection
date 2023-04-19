from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

from skfeature.function.similarity_based import fisher_score, reliefF, SPEC
from skfeature.function.statistical_based import f_score, chi_square, CFS
from skfeature.function.sparse_learning_based import RFS
from skfeature.function.information_theoretical_based import CMIM, JMI, MRMR
from BorutaShap import BorutaShap

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.metrics import balanced_accuracy_score, accuracy_score

from lime.lime_tabular import LimeTabularExplainer
import shap

import time
import numpy as np
import pandas as pd
import copy
import pickle
from tqdm import tqdm
import rfecv
default_out_path = 'feature_selection/rsl/'
out_path_temp = 'feature_selection/rsl/temp'

fs_functions = {
    # 'acr' : (func, need_y, coef)
    'fisher': (fisher_score.fisher_score, True, False, 'similarity_based'),
    'reliefF': (reliefF.reliefF, True, False, 'similarity_based'),
    'spec': (SPEC.spec, False, False, 'similarity_based'),
    'f': (f_score.f_score, True, False, 'statistical_based'),
    'chi2': (chi_square.chi_square, True, False, 'statistical_based'),
    'RFS': (RFS.rfs, True, True, 'sparse_learning_based'), 
    'mrmr': (MRMR.mrmr, True, False, 'information_theoretical_based'),
    'jmi': (JMI.jmi, True, False, 'information_theoretical_based'),
    'cmim': (CMIM.cmim, True, False, 'information_theoretical_based'),
    'rf': (RandomForestClassifier(random_state=42), True, False, 'embedded'),
    'lasso_svc': (LinearSVC(penalty="l1", dual=False), True, True, 'embedded'),
    'lasso_lg': (LogisticRegressionCV(penalty="l1", dual=False, solver='liblinear', random_state=42), True, True, 'embedded'),
    'borutashap': (BorutaShap(importance_measure='shap', classification=True), True, False, 'borutashap')
}

models = {
    'MLP': MLPClassifier(random_state=0),
    'SVM': SVC(random_state=0, probability=True),
    'RF': RandomForestClassifier(random_state=0),
}


def select(dataset: int, dic_param: dict=None, preference: str=None, mask=None,
           rsl: dict=None, rsl_path: str=default_out_path):
    # print(dataset, time.ctime())

    if type(dataset) == int:
        dataset_id = dataset
        X, y = fetch_openml(data_id=dataset_id, as_frame=True, return_X_y=True)
    else:
        data_obj = fetch_openml(name=dataset, as_frame=True)
        X = data_obj.data
        y = data_obj.target
        dataset_id = int(data_obj.details['id'])

    if mask is not None: X= X.iloc[:, mask]
    feature_names = X.columns
    nb_features = len(feature_names)
    X_numpy = X.to_numpy().astype(float)

    le = LabelEncoder()
    y_numpy = le.fit_transform(y)

    if rsl is None:
        rsl = {}
    if dataset_id not in rsl:
        rsl[dataset_id] = {}
        r = feature_rankings(X, y_numpy)
        s = feature_subsets(X, y, feature_names)

        candidates = rfe_ratings(r, X_numpy, y_numpy)
        candidates = merge_subsets(candidates, s, nb_features)
        # print(candidates, time.ctime())
        nb_candidates = len(candidates)
        # print(nb_candidates, " candidates of ", dataset_id)

        rsl[dataset_id] = {'feature_names': feature_names,
                           'rankings': r,
                           'subsets': s,
                           'candidates': candidates}
        explained = False
        save_close(rsl, out_path_temp + str(dataset_id))
    else:
        candidates = rsl[dataset_id]['candidates']
        explained = True
        if type(dic_param) is not dict:
            explained = False
        else:
            for m, es in dic_param.items():
                for candidate in candidates.values():
                    if 'models' not in candidate.keys():
                        explained=False
                        break
                    
                    if m not in candidate['models']:
                        explained = False
                        break
                    if not all(i in list(candidate['models'][m]['explainers'].keys()) for i in list(es.keys())):
                        explained = False
                        break
    if not explained:
        i = 1
        for c, candidate in rsl[dataset_id]['candidates'].items():
            # to refrefactoring :  add mask for rankings
            mask = candidate['mask'] # all candidates should have mask in the merging step
            for fs in candidate['fs_acr']:
                if fs != 'all': rsl[dataset_id]['rankings'][fs]['mask'] = mask

            candidate['models'] = {}
            param = {}  # gridcv removed
            for m in dic_param:
                candidate['models'][m] = {}
                model = copy.deepcopy(models[m])
                model.set_params(**param)
                explain = (preference != 'accuracy') and (type(dic_param) is dict)
                dic_param_e = dic_param[m] if explain else None

                train = train_explain_one_model(X, y_numpy, mask, model, dic_param_e=dic_param_e, silent=False, explain=explain)

                candidate['models'][m] = {**candidate['models'], **train}

                save_close(rsl, out_path_temp + str(dataset_id))
            i = i + 1

    save_close(rsl, rsl_path + str(dataset_id))
    return rsl


def feature_rankings(X_df, y) -> dict:
    rsl = {}
    X = X_df.to_numpy().astype(float)

    n_features = X.shape[1]

    for fs, method in fs_functions.items():
        print(fs)
        if fs != 'RFS' and len(np.unique(y)) == 2:
            if fs == 'chi2' and (np.all(X > 0) is False):
                X_train = MinMaxScaler().fit_transform(X)
            else:
                X_train = X
            y_train = copy.deepcopy(y)
            if fs == 'RFS':
                lb = LabelBinarizer()
                y_train = lb.fit_transform(y_train)

            if method[3] == 'embedded':
                clf = copy.deepcopy(method[0])
                t = time.time()
                clf.fit(X_train, y_train)
                score = clf.coef_ if method[2] else clf.feature_importances_
            elif method[3] == 'borutashap':
                Feature_Selector = BorutaShap(importance_measure='shap', classification=True)
                Feature_Selector.fit(X=X_df, y=y_train, random_state=42, verbose=False)
                t = time.time()
                # feature_importance = self.shap_values
                # rsl['shap'] = {"duration": duration_bs, "score": Feature_Selector.feature_importance(normalize=False),
                #                "order": np.argsort(Feature_Selector.feature_importance(normalize=False)[0])[::-1]}
                score = Feature_Selector.history_x.iloc[1:, :n_features].mean(axis=0).values
            elif method[3] == 'information_theoretical_based':
                t = time.time()
                score = method[0](X_train, y_train, n_selected_features=n_features)
            else:
                t = time.time()
                score = method[0](X_train, y_train) if method[1] else method[0](X_train)

            rsl[fs] = save_ranking(time.time() - t, score, coef=method[2], it_based=method[3] == 'information_theoretical_based')

    return rsl


def feature_subsets(X: pd.DataFrame, y: pd.DataFrame,
                    feature_names) -> dict:
    rsl = {}
    n_features = feature_names.size
    rsl['all'] = save_subset(0, np.array(range(n_features)), n_features)

    '''
    # statistical_based  CFS
    # F: {numpy array} index of selected features
    t = time.time()
    cfs = CFS.cfs(X.to_numpy(), y.to_numpy())
    rsl['cfs'] = save_subset(time.time()-t, cfs, n_features)
    '''
    return rsl


def train_explain_one_model(X, y_or, mask, model, dic_param_e, silent=False, explain=True):
    rsl = {}
    X_selected = X.iloc[:, mask]

    clf = copy.deepcopy(model)
    clf = clf.fit(X_selected.values, y_or)
    yPredicted = clf.predict(X_selected.values)
    b_score = balanced_accuracy_score(y_or, yPredicted)
    a_score = accuracy_score(y_or, yPredicted)

    rsl = {'accuracy': a_score,
           'balanced_accuracy': b_score,
           'classifier': clf,
           'yPredicted': yPredicted,
           'explainers': {}
           }
    if explain:
        # calculating explanation
        for e, param_e in dic_param_e.items():
            explainer = explanation_values_lime if e == 'LIME' else explanation_values_shap

            explanation, t = explainer(X=X_selected, clf = clf, mask = mask,
                                       mode='classification', param_e = param_e, silent=silent)

            rsl['explainers'][e] = {'time': t, 'explanation': explanation}
            # save_close(rsl, f'{out_path_temp}_{e}')
    return rsl


def save_ranking(time, score, coef=False, it_based=False) -> dict:
    rsl = {'duration': time,
           'score': score}
    if coef:
        if score.shape[0] > 1:  # multi_classe
            rsl['order'] = np.argsort(np.abs(score.ravel()))[::-1] if score.shape[1] == 1 \
                else  np.argsort(np.sum(np.abs(coef), axis=0))[::-1]
        else:
            rsl['order'] = np.argsort(np.abs(score))[::-1].flatten()
    elif it_based:
        if type(score) is tuple:
            rsl['order'] = score[0]
        else:
            rsl['order'] = score
            rsl.pop('score')
    else:
        rsl['order'] = np.argsort(score)[::-1]

    return rsl


def selected_to_mask(selected_index_tuple: tuple,
                     n_features: int) -> np.ndarray:
    selected_index = sorted(selected_index_tuple)
    return np.isin(range(n_features), selected_index)


def name_to_index(selected_columns: list, feature_names) -> np.ndarray:
    return np.array([feature_names.get_loc(f) for f in selected_columns],
                    dtype=int)


def name_to_mask(selected_columns: list, feature_names) -> np.ndarray:
    # return np.array([feature_names.index(f) for f in selected_columns], dtype=int)
    return np.isin(feature_names, selected_columns)


def mask_to_index(mask: np.ndarray) -> np.ndarray:
    return np.array([i for i, x in enumerate(mask) if x],
                    dtype=int)


def save_close(data, file_name):
    file = open(file_name, 'wb')
    pickle.dump(data, file)
    file.close()


def save_subset(time, selected_i, n_features) -> dict:
    return {"duration": time,
            "subset": selected_i,
            "mask": np.isin(range(n_features), selected_i)}


def rfe_ratings(raw_ranking: np.array, X, y) -> dict:
    merge = {}
    for k, v in raw_ranking.items():
        if tuple(v['order']) in merge:
            merge[tuple(v['order'])].append(k)
        else:
            merge[tuple(v['order'])] = [k]
    # print(merge)
    rfe = {}
    for k, v in merge.items():
        ranks_keys = list(k)
        n_features_to_select, best_param, time = rfecv.get_n_features_to_select_grid_rf(X, y, ranks_keys)
        subsets = sorted(ranks_keys[:n_features_to_select])
        # print(subsets, n_features_to_select, best_param, time)

        if tuple(subsets) in rfe:
            rfe[tuple(subsets)]['fs_acr'].update(v)
            rfe[tuple(subsets)]['fs'][tuple(v)] = {'param': str(best_param),
                                                   'duration': time}
        else:
            rfe[tuple(subsets)] = {'n': len(subsets),
                                   'mask': selected_to_mask(subsets, len(ranks_keys)),
                                   'fs_acr': set(v),
                                   'fs': {tuple(v): {'param': str(best_param),
                                                     'duration': time}}}
    return rfe


def merge_subsets(merged: dict, raw_subsets: dict, nb_features) -> dict:
    for k, v in raw_subsets.items():
        if v['subset'] is not None:
            s = sorted(v['subset'])
            if tuple(s) in merged:
                merged[tuple(s)]['fs'][k] = None
                merged[tuple(s)]['fs_acr'].update([k])
            else:
                merged[tuple(s)] = {'n': len(s),
                                    'mask': selected_to_mask(subsets, nb_features),
                                    'fs_acr': set([k]),
                                    'fs': {k: None}}

    return merged


def explanation_values_lime(X: pd.DataFrame, clf, mask, mode, look_at=1, param_e={},
                            silent=False, discretize_continuous=True):

    X_numpy = X.to_numpy().astype(float)
    explainer = LimeTabularExplainer(X_numpy, mode=mode,
                                     feature_names=X.columns,
                                     discretize_continuous=discretize_continuous)
    num_samples = int(param_e['num_samples'])
    t0 = time.time()
    if silent:
        inf_values = np.array([[v for (k, v) in sorted(
            explainer.explain_instance(X_numpy[i], clf.predict_proba, labels=(look_at,), num_samples=num_samples,
                                       num_features=X_numpy.shape[1]).as_map()[look_at])] for i in range(X.shape[0])])
    else:
        inf_values = np.array([[v for (k, v) in sorted(
            explainer.explain_instance(X_numpy[i], clf.predict_proba, labels=(look_at,), num_samples=num_samples,
                                       num_features=X_numpy.shape[1]).as_map()[look_at])] for i in
                               tqdm(range(X.shape[0]))])

    for i, b in enumerate(mask):
        if not b:
            inf_values = np.insert(inf_values, i, np.zeros(X_numpy.shape[0]), axis=1)

    t1 = time.time()

    # Generate explanation compatible with shap
    explanation = shap.Explanation(inf_values,
                                   base_values=np.zeros(X.shape[0]),
                                   data=X_numpy,
                                   feature_names=X.columns.to_list())

    return explanation, t1 - t0


def explanation_values_shap(X: pd.DataFrame, clf, mask, mode, look_at=1, param_e={},
                            silent=False, discretize_continuous=True):
    X_numpy = X.to_numpy().astype(float)

    if param_e['summarize'] == "Sampling":
        explainer = shap.explainers.Sampling(clf.predict_proba, X_numpy)
    else:
        explainer = shap.KernelExplainer(clf.predict_proba, X_numpy)

    nsamples = param_e['nsamples']
    if nsamples != 'auto': nsamples = int(nsamples)
    l1_reg = param_e['l1_reg']
    t0 = time.time()

    shap_values = explainer.shap_values(X, nsamples = nsamples,l1_reg=l1_reg)[look_at]

    t1 = time.time()
    for i, b in enumerate(mask):
        if not b:
            shap_values = np.insert(shap_values, i, np.zeros(X_numpy.shape[0]), axis=1)

    explanation = shap.Explanation(shap_values,
                                   base_values=explainer.expected_value,
                                   data=X_numpy,
                                   feature_names=X.columns.to_list())
    return explanation, t1 - t0
