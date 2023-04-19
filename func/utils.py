import os
import itertools
import operator
import pickle
import glob
from typing import Union

import numpy as np
import pandas as pd
import seaborn as sns
import shap

import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from matplotlib import gridspec

import importlib
from func import pseudo_shap, metric

importlib.reload(pseudo_shap)
importlib.reload(metric)

# constant
default_rsl_path = os.path.abspath(os.path.dirname(os.getcwd())) + '/fs_user/out_1202/rsl_1202_ds_'
default_ds = [int(f.split('_')[-1]) for f in glob.glob(default_rsl_path+'*')]
default_all_models = ['en', 'knn', 'nb', 'xg']
default_fs = {'all': (sns.xkcd_rgb['black'], 'all', 'w', None),
              # 'similarity_based'
              'fisher': (sns.xkcd_rgb['light blue'], 'similarity_based', 'c', None),
              'relief': (sns.xkcd_rgb['cyan'], 'similarity_based', 'c', 'reliefF'),
              'spec': (sns.xkcd_rgb['robin egg blue'], 'similarity_based', 'c', None),
              # 'statistical_based'
              'f': (sns.xkcd_rgb['green'], 'statistical_based', 'g', None),
              # 'chi2': (sns.xkcd_rgb['teal'], 'statistical_based', 'g', 'chi_positive'),
              'chi2_nor': (sns.xkcd_rgb['light green'], 'statistical_based', 'g', 'chi2'),
              # 'cfs': ('statistical_based', 'statistical_based', 'g', 'cfs_subset'),
              # 'sparse_learning_based'
              # 'rfs': (sns.xkcd_rgb['purple'], 'sparse_learning_based', 'm', 'rfs_error'),
              'RFS': (sns.xkcd_rgb['purple'], 'sparse_learning_based', 'm', 'rfs'),
              # 'information_theoretical_based':
              'mrmr': (sns.xkcd_rgb['bright blue'], 'information_theoretical_based', 'b', None),
              'cmim': (sns.xkcd_rgb['royal blue'], 'information_theoretical_based', 'b', None),
              'jmi': (sns.xkcd_rgb['cerulean'], 'information_theoretical_based', 'b', None),
              # 'wrapper':
              'borutashap': (sns.xkcd_rgb['gold'], 'wrapper', 'y', None),
              # 'embedded'
              'rf': (sns.xkcd_rgb['salmon'], 'embedded', 'r', None,),
              # 'svc': (sns.xkcd_rgb['rose'], 'embedded', 'r', 'svc_error'),
              'lasso_svm': (sns.xkcd_rgb['rose'], 'embedded', 'r', 'svm'),
              # 'lg': (sns.xkcd_rgb['burnt orange'], 'embedded', 'r', 'lg_err'),
              'lasso_lg': (sns.xkcd_rgb['burnt orange'], 'embedded', 'r', 'lg'),
              }

default_metrics_dic = {
    'tk': ('Kendall rank correlation coefficient', metric.tau_kendall, 1, -1),
    'rinfc': ('Relative influence change', metric.relative_influence_changes, 0, 1),
    'RI': ('RI', metric.RI, 0, 1),
    'RIA': ('RIA', metric.RIA, 1, -1),
}


def selected_to_mask(selected_index_tuple: tuple,
                     n_features: int) -> np.ndarray:
    selected_index = sorted(selected_index_tuple)
    return np.isin(range(n_features), selected_index)


## explanation related
def get_candidate(dataset_id: int, fs: str, rsl_path=default_rsl_path) -> tuple:
    rsl = pickle.load(open(rsl_path + str(dataset_id), 'rb'))[dataset_id]
    if fs in rsl['subsets']:
        return tuple(sorted(rsl['subsets'][fs]['subset']))

    if 'subset' in rsl['rankings'][fs]:
        return tuple(sorted(rsl['rankings'][fs]['subset']))
    for c, can in rsl['candidates'].items():
        if fs in can['fs_acr']:
            return tuple(sorted(c))


def get_explanation(dataset_id: int, model: str, fs: str = None, candidate: tuple = None,
                    explainer: str = 'lime', selected: bool = True,
                    rsl_path=default_rsl_path) -> shap._explanation.Explanation:
    assert not (fs is None and candidate is None)

    # selected = do not keep 0 in influences values
    # when selected=True ex.feature_names==ex.selected_names, else  ex.feature_names==all_selected_names
    if fs is not None:
        candidate_key = get_candidate(dataset_id=dataset_id, fs=fs, rsl_path=rsl_path)
    else:
        candidate_key = candidate

    rsl = pickle.load(open(rsl_path + str(dataset_id), 'rb'))[dataset_id]
    ex = rsl['candidates'][candidate_key]['models'][model]['explainers'][explainer]['explanation']

    selected_index = sorted(candidate_key)
    ex.selected_index = sorted(candidate_key)
    ex.selected_names = ex.feature_names
    ex.all_features = list(rsl['feature_names'])

    if selected:
        ex.values = np.asarray([ex.values[:, i] for i in selected_index]).T
    else:
        ex.feature_names = list(rsl['feature_names'])
        # raise ValueError('no explanation found: dataset_id=',dataset_id,' fs=',fs, ' model=',model)
    return ex


def get_fs_from_candidate(dataset_id: int, candidate: tuple, rsl_path=default_rsl_path) -> list:
    return pickle.load(open(rsl_path + str(dataset_id), 'rb'))[dataset_id]['candidates'][candidate]['fs_acr']


## selection related
def get_all_fs(datasets, rsl_path=default_rsl_path) -> set:
    fss = set()
    for dataset_id in datasets:
        candidates = pickle.load(open(rsl_path + str(dataset_id), 'rb'))[dataset_id]['candidates']
        for c, candidate in candidates.items():
            fss.update(candidate['fs_acr'])
    return fss


def get_selection_rates(dataset_id: int, sort=True, selection_rates_df=None, rsl_path=default_rsl_path) -> dict:
    selection_rates = {}
    if selection_rates_df is None:
        rsl = pickle.load(open(rsl_path + str(dataset_id), 'rb'))[dataset_id]
        n_features = len(rsl['feature_names'])
        candidates = rsl['candidates']
        for c, candidate in candidates.items():
            for fs in candidate['fs_acr']:
                selection_rates[fs] = candidate['n'] / n_features
    else:
        selection_rates = selection_rates_df.loc[dataset_id].to_dict()

    if sort:
        selection_rates = dict(sorted(selection_rates.items(),
                                      key=operator.itemgetter(1), reverse=True))
    return selection_rates


def get_fs_without_selection(all_ds: list, rsl_path=default_rsl_path) -> (dict, dict):
    rsl_fs = {}
    rsl_ds = {}
    for dataset_id in all_ds:
        rsl = pickle.load(open(rsl_path + str(dataset_id), 'rb'))[dataset_id]
        n_features = len(rsl['feature_names'])
        key_all = tuple(set(range(n_features)))

        candidate_all = rsl['candidates'][key_all]
        if len(candidate_all['fs_acr']) > 1:
            rsl_ds[dataset_id] = (list(candidate_all['fs_acr']),
                                  candidate_all['n'])

            for fs in candidate_all['fs_acr']:
                if fs in rsl_fs.keys():
                    rsl_fs[fs].append((dataset_id, candidate_all['n']))
                else:
                    rsl_fs[fs] = [(dataset_id, candidate_all['n'])]
    rsl_fs.pop('all')
    return rsl_fs, rsl_ds


def get_mask(dataset_id: int, fs: str, rsl_path=default_rsl_path) -> np.array:
    rsl = pickle.load(open(rsl_path + str(dataset_id), 'rb'))[dataset_id]
    n_features = len(rsl['feature_names'])
    if fs in rsl['subsets']:
        selected_index = sorted(rsl['subsets'][fs]['subset'])
    else:
        selected_index = sorted(rsl['rankings'][fs]['subset'])

    return np.isin(range(n_features), selected_index)

    # raise ValueError('no mask found: dataset_id=',dataset_id,' fs=',fs)


## candidate related
def get_ranking_dic(dataset_id: int, fs: str, rsl_path=default_rsl_path) -> dict:
    return pickle.load(open(rsl_path + str(dataset_id), 'rb'))[dataset_id]['rankings'][fs]


# retuen fs_acr and its n_selected_feature
def display_candidates(dataset_id: int, rsl_path=default_rsl_path) -> list:
    candidates = pickle.load(open(rsl_path + str(dataset_id), 'rb'))[dataset_id]['candidates']
    methods = []
    for c, candidate in candidates.items():
        methods.append((candidate['n'], candidate['fs_acr'], c))

    sorted_methods = sorted(methods, key=lambda t: t[0])
    return sorted_methods


def quick_metric(metric_arc: str, dataset_id: int, fs: str, model: str,
                 rsl_df_raw=None, metrics_funs: dict = default_metrics_dic):
    metric_name = metrics_funs[metric_arc][0]
    print(metric_name)
    if rsl_df_raw is not None and fs in rsl_df_raw.columns:
        return rsl_df_raw.loc[(model, metric_name, dataset_id), fs]
    return metrics_funs[metric_arc][1](
        get_explanation(dataset_id=dataset_id, model=model, fs='all', selected=False).values,
        get_explanation(dataset_id=dataset_id, model=model, fs=fs, selected=False).values,
        get_mask(dataset_id, 'all'),
        get_mask(dataset_id, fs))


def fs_score_ranking(dataset: int, model: str, metric_arc: str,
                     rsl_df_raw: pd.DataFrame, metrics_funs: dict = default_metrics_dic):
    ascending = (metrics_funs[metric_arc][2] == 0)
    return rsl_df_raw.loc[(model, metrics_funs[metric_arc][0], dataset), slice(None)].sort_values(ascending=ascending)


## plot function
def get_all_scores_rate(used_fs: list = list(default_fs.keys()), metrics_info: dict = default_metrics_dic,
                        datasets: list = default_fs,  models: dict = default_all_models,
                        explainers: dict = {'lime'},  accuracies=['accuracy', 'balanced_accuracy'],
                        keep_all=False, rsl_path=default_rsl_path):
    # keep_all : if take th ds with selection into account
    multi_i = pd.MultiIndex.from_product([models,
                                          [t[0] for k, t in metrics_info.items()],
                                          datasets])
    rsl_df_raw = pd.DataFrame(index=multi_i, columns=used_fs, dtype='float64')

    index_score = pd.MultiIndex.from_product([models, accuracies, datasets])
    accuracies_raw = pd.DataFrame(index=index_score, columns=used_fs, dtype='float64')

    selection_rate_raw = pd.DataFrame(index=datasets, columns=used_fs,
                                      dtype='float64')

    for ds in datasets:
        rsl = pickle.load(open(rsl_path + str(ds), 'rb'))[ds]
        n_features = len(rsl['feature_names'])
        mask_all = np.ones(n_features)
        key_all = tuple(set(range(n_features)))

        candidates = rsl['candidates']
        for m in models:
            for e in explainers:
                for m_f, m_tuple in metrics_info.items():
                    m_function = m_tuple[1]
                    serie = pd.Series(dtype='float64')

                    values_all = candidates[key_all]['models'][m]['explainers'][e]['explanation'].values

                    acc_all = candidates[key_all]['models'][m][accuracies[0]]

                    for c, candidate in candidates.items():
                        mask = selected_to_mask(c, n_features)
                        values = candidate['models'][m]['explainers'][e]['explanation'].values

                        acc_new = candidate['models'][m][accuracies[0]]
                        score = m_function(values_all, values, mask_all, mask, acc_all, acc_new)

                        for fs_keys in candidate['fs_acr']:
                            if keep_all or c != key_all or fs_keys == 'all':
                                serie.loc[fs_keys] = score

                                selection_rate_raw.loc[ds, fs_keys] = len(c) / n_features
                                for acc in accuracies:
                                    accuracies_raw.loc[(m, acc, ds), fs_keys] = acc_new
                    rsl_df_raw.loc[m, m_tuple[0], ds] = serie
    print(f"Done for {len(datasets)} datasets : {str(datasets)}")
    return rsl_df_raw, selection_rate_raw, accuracies_raw


def display_heatmap(metrics_rsl_df: pd.DataFrame, selection_rate_df: pd.DataFrame,
                    accuracy_df: pd.DataFrame, metric_arc: str, RIA_rescale : int=3,
                    fs_on_plt: dict=default_fs, models:list=default_all_models, datasets: list = default_ds,
                    metrics_f: dict = default_metrics_dic,
                    # to be implemented
                    # display_stat: bool = False, cluster: bool = False,
                    # display style
                    circle_size: str = 'mean', circle_color: str = 'std',
                    rename_xaxis: bool = True, rename_yaxis: bool = True,  legend_vs=True,
                    title: str = None, title_size: int = 18, label_size: int = 16, tick_size: int = 14,
                    font_size: int = 14, rnd:int=2, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))
    ds_slice = slice(None) if datasets is None else datasets

    # aggregation
    metric_fname = metrics_f[metric_arc][0]

    data = metrics_rsl_df.loc[slice(None), metric_fname, ds_slice]
    if metric_arc == 'RIA' and RIA_rescale > 0:
        data = data * (10 ^ RIA_rescale)
    selection_rate = selection_rate_df.loc[ds_slice]
    accuracy = accuracy_df.loc[slice(None), slice(None), ds_slice]

    # fs selection
    fs_order = [fs for fs in fs_on_plt if fs in data.columns]
    data = data[fs_order]
    data_std = data.groupby(level=0).std()
    data_mean = data.groupby(level=0).mean()

    data_dic = {'mean': data_mean,
                'std': data_std}

    # clustermap style

    cmap = 'rocket'
    norm = mcolors.Normalize(vmin=min(0, data_dic[circle_color].values.min()), vmax=data_dic[circle_color].values.max())
    if circle_color != 'std' and np.isclose(np.max(data_dic[circle_color]['all']), 0):
        cmap = 'rocket_r'
    # if data_dic[circle_color].values.min() < 0:
    if metric_arc == 'RIA' and data_dic[circle_color].values.min() < 0:
        cmap = 'vlag'
        norm = mcolors.TwoSlopeNorm(vmin=data_dic[circle_color].values.min(), vcenter=0,
                                    vmax=data_dic[circle_color].values.max())

    '''
    color_map = cm.get_cmap(cmap)
    if metric_arc == 'RIA':
        norm_size = None
    else:
        norm_size = mcolors.Normalize(min(0, data_dic[circle_size].values.min()),
                                      vmax=data_dic[circle_size].values.max())
    # data = data.rename(columns=dic_rename_fs, index=dic_rename_model)
    '''
    if metric_arc == "RIA" and (rnd+RIA_rescale) <4:
        rnd = 4-RIA_rescale
    fmt = f".{rnd}f"
    sns.heatmap(data_mean, cmap=mcolors.ListedColormap('gainsboro'), annot=True, fmt=fmt, alpha=0.2, linewidths=0.1,
                cbar=False, linecolor='dimgray', annot_kws={'size': font_size, 'color': 'black'}, ax=ax, square=True)
    dic_x = {inf.get_text(): inf.get_position()[0] for inf in ax.xaxis.get_ticklabels()}
    dic_y = {inf.get_text(): inf.get_position()[1] for inf in ax.yaxis.get_ticklabels()}

    # color bar
    ax.text(1.005, 0.96, 'std', transform=ax.transAxes, verticalalignment='bottom', ha='left', size=label_size, )
    cax = ax.inset_axes([1 + 0.01, 0, 0.02, 0.95])
    cb = ax.figure.colorbar(mappable=cm.ScalarMappable(norm, cmap=cmap), cax=cax)
    cb.ax.tick_params(labelsize=tick_size - 2)
    cb.outline.set_linewidth(0)

    # create radius
    radius = pd.DataFrame(index=models, columns=fs_order)

    def minmax_sc(v, min_v, max_v):
        return (v - min_v) / (max_v - min_v)

    if metrics_f[metric_arc][2] == 0:
        # if metric_arc == 'comp_rinf_w' or \
        #         abs(data_dic[circle_size].values).min() == abs(np.min(data_dic[circle_size]['all'])):
        min_v = data_dic[circle_size].values
        min_v = min_v[min_v != 0].min()
        min_nor = min_v / 1.2

        max_v = data_dic[circle_size].values.max()
        max_nor = max_v * 1.2

        for fs, model in itertools.product(fs_order, models):
            radius.loc[model, fs] = 0 if fs == 'all' \
                else 0.5 - minmax_sc(data_dic[circle_size].loc[model, fs], min_nor, max_nor) / 2
    else:
        max_v = data_dic[circle_size].values
        max_v = max_v[max_v != 1].max()  # if metric_arc != 'RIA' else max_v.max() # needless if in practice
        max_nor = max_v * 1.2

        min_v = data_dic[circle_size].values.min()
        min_nor = min_v / 1.2
        for fs, model in itertools.product(fs_order, models):
            radius.loc[model, fs] = 0 if fs == 'all' and metric_arc != 'RIA' \
                else minmax_sc(data_dic[circle_size].loc[model, fs], min_nor, max_nor) / 2

    circles = [plt.Circle((dic_x[fs], dic_y[model]), radius=radius.loc[model, fs])
               for fs, model in itertools.product(fs_order, models)]
    col = PatchCollection(circles, array=data_dic[circle_color].values.flatten(order='F'),
                          cmap=cmap, norm=norm, alpha=0.725)
    ax.add_collection(col)

    median_v = np.median(data_dic[circle_size].values)
    legend_v = [max_v, median_v, min_v]

    legend_radius = [0.5 - minmax_sc(v, min_nor, max_nor) / 2 if metrics_f[metric_arc][2] == 0
                     else minmax_sc(v, min_nor, max_nor) / 2 for v in legend_v]
    # if metric_arc == 'RIA' and RIA_rescale > 0:
    # legend_radius = [r/2 for r in legend_radius]

    legend_text = [ "%.2f" % round(v, 2)for v in legend_v]
    legend_circle = [mpatches.Circle((), radius=r, facecolor="white", edgecolor='black') for r in legend_radius]

    class HandlerCircle(HandlerPatch):
        def create_artists(self, legend, orig_handle,
                           xdescent, ydescent, width, height, fontsize, trans):
            center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
            # center = 0, 0.5 * height - 0.5 * ydescent
            p = mpatches.Circle(xy=center,
                                radius=orig_handle.radius *
                                       45
                                       # * (ax.get_position().height + ax.get_position().width) / 2
                                )
            self.update_prop(p, orig_handle, legend)
            p.set_transform(trans)
            return [p]

    if legend_vs:
        legend = ax.legend(legend_circle, legend_text, handler_map={mpatches.Circle: HandlerCircle()},
                           # title='value',
                           # title_fontsize=tick_size,
                           loc='upper left', bbox_to_anchor=(1.12, 0.9, 0.1, 0.01),
                           labelspacing=1.8,
                           handlelength=0, handleheight=3.5,
                           # markerfirst=False,
                           handletextpad=0,
                           borderaxespad=0,
                           # mode='expand',
                           frameon=False,
                           fontsize=tick_size - 2)
        legend.get_title().set_ha('left')
        for txt in legend.get_texts():
            if len(txt.get_text()) < 5:
                txt.set(x=-20, y=-35, ha='center')
            else:
                txt.set(x=-30, y=-35, ha='center')
    else:
        legend = ax.legend(legend_circle, legend_text, handler_map={mpatches.Circle: HandlerCircle()},
                           loc='upper left', bbox_to_anchor=(1.1, 0.9, 0.1, 0.01),
                           title_fontsize=label_size,
                           fontsize=tick_size - 2,
                           labelspacing=1.8,
                           handlelength=0.5, handleheight=3,
                           # markerfirst=False,
                           handletextpad=1.5,
                           borderaxespad=0.5,
                           # mode='expand',
                           frameon=False)
    if metric_arc == 'RIA' and RIA_rescale>0:
        ax.text(1.13, 0.96, fr'$10^{RIA_rescale}\times$value', transform=ax.transAxes, verticalalignment='bottom', ha='center',
                size=label_size)
    else:
        ax.text(1.13, 0.96, 'value', transform=ax.transAxes, verticalalignment='bottom', ha='center',
                size=label_size)

    if rename_xaxis:
        rate_order = [corresponding_fs_names_all_models(selection_rate, accuracy, fs, dss=datasets, fs_info=fs_on_plt) for
                      fs in fs_order]
        if rename_xaxis == 'top':
            ax.xaxis.tick_top()
            rename_fs = [fs_on_plt[fs][3] if (fs_on_plt[fs][3] is not None) else fs for fs in fs_order]
            ax.set_xticklabels(rename_fs, rotation=25, size=tick_size)


        # ax.text(-ax.xaxis.get_ticklabels()[1].get_position()[0], -ax.yaxis.get_ticklabels()[-1].get_position()[1],'dhuchizvcuhiz')
        # ax.set_xlabel('FS method | average retention rate | average accuracy | number of datasets involved', size=labelsize)
        else:
            ax.set_xticklabels(rate_order, rotation=45, ha='right', size=tick_size)
    
        # ax.set_xlabel('feature selection method', size=labelsize)
    else:
        ax.get_xaxis().set_visible(False)

    if rename_yaxis:
        model_order = [corresponding_models_names_w_accuracy(model, selection_rate, accuracy) for model in models]
        ax.set_yticklabels(model_order, rotation='horizontal', size=tick_size)
        ax.set_ylabel('model | average accuracy', size=label_size, weight='bold')
    else:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, size=tick_size)
        # ax.set_ylabel('model', size=labelsize)
        # ax.get_yaxis().set_visible(False)

    if title is not False:
        if title is None:
            title = metric_fname
            if metric_arc in ["RI", "RIA"]:
                metric_arc += " metric"

        ax.set_title(title, size=title_size, weight='bold')

    return ax.figure


def corresponding_fs_names_all_models(selection_rate_df: pd.DataFrame,
                                      accuracy_df: pd.DataFrame, fs: str,
                                      dss: list,
                                      fs_info: dict = default_fs,
                                      accuracy_type: str = 'accuracy'):
    rename_fs = fs_info[fs][3] if (fs_info[fs][3] is not None) else fs

    return f"{rename_fs} | {'%.0f%%' % (get_avg_selection_rate(fs, all_ds=dss, selection_rate_df=selection_rate_df) * 100)} | {'%.0f%%' % (get_avg_accuracy_of_fs(fs, accuracy_type=accuracy_type, accuracy_df=accuracy_df) * 100)} | {selection_rate_df[fs].notnull().sum()}"


def corresponding_models_names_w_accuracy(model: str, selection_rate_df: pd.DataFrame,
                                          accuracy_df: pd.DataFrame = None, rsl_path=None,
                                          accuracy_type='accuracy'):
    if rsl_path is not None:
        return f"{model} | {'%.0f%%' % (get_avg_accuracy_of_model(model, accuracy_type=accuracy_type, rsl_path=rsl_path) * 100)}"
    else:
        return f"{model} | {'%.0f%%' % (get_avg_accuracy_of_model(model, accuracy_type=accuracy_type, accuracy_df=accuracy_df) * 100)}"


def get_avg_selection_rate(fs: str, all_ds: list = None, selection_rate_df=None,
                           remove_all=True) -> float:
    if fs == 'all': return 1

    if selection_rate_df is None:
        rates = []
        for dataset_id in all_ds:
            rsl = pickle.load(open(default_rsl_path + str(dataset_id), 'rb'))[dataset_id]
            nb_features = len(rsl['feature_names'])
            nb_selection = 0
            if fs in rsl['subsets']:
                nb_selection = len(rsl['subsets'][fs]['subset'])
            else:
                nb_selection = len(rsl['rankings'][fs]['subset'])

            rates.append(nb_selection / nb_features)
        rates = pd.Series(rates)
    else:
        rates = selection_rate_df[selection_rate_df.index.isin(all_ds)]
        rates = rates[fs]
    if remove_all:
        rates = rates[rates != 1]
    return rates.mean()


def get_avg_accuracy(fs: str, model: str, datasets: list = None,
                     accuracy_type='accuracy', accuracy_df=None, rsl_path=default_rsl_path) -> float:
    if rsl_path is not None:
        scores = []
        for dataset_id in datasets:
            scores.append(get_accuracy(dataset_id, fs, model, accuracy_type, rsl_path=rsl_path))
        scores = pd.Series(scores)
    else:
        scores = accuracy_df.loc[(model, accuracy_type, slice(None)), fs]

    return scores.mean()


def get_accuracy(dataset_id: int, fs: str, model: str,
                 accuracy_type: str = 'accuracy',
                 accuracy_df=None, rsl_path=default_rsl_path) -> float:
    if accuracy_df is not None:
        accuracy = accuracy_df.loc[(model, accuracy_type, dataset_id), fs]
    else:
        rsl = pickle.load(open(rsl_path + str(dataset_id), 'rb'))[dataset_id]
        candidates_key = get_candidate(dataset_id, fs, rsl_path=rsl_path)
        accuracy = rsl['candidates'][candidates_key]['models'][model][accuracy_type]
    return accuracy


def get_avg_accuracy_of_fs(fs: str, accuracy_df=None, rsl_path=None,
                           accuracy_type='accuracy', ) -> float:
    if rsl_path is not None:
        return 0
    scores = accuracy_df.loc[(slice(None), accuracy_type, slice(None)), fs]
    return scores.mean()


def get_avg_accuracy_of_model(model: str, accuracy_df=None, rsl_path=None,
                              accuracy_type: str = 'accuracy') -> float:
    if rsl_path is not None:
        return 0
    scores = accuracy_df.loc[(model, accuracy_type, slice(None)), slice(None)]
    return scores.mean().mean()


def show_3d_plot(model: str, metric: str, rsl_metric: pd.DataFrame, rsl_accuracy: pd.DataFrame,
                 rsl_retention: pd.DataFrame,
                 anno: bool = False, size_anno: int = 10, labelsize: int = 12, ticksize: int = 12):
    
    ##  color=default_fs[all_fs[i]][0] to modified !!!
    mean_retention = rsl_retention.mean()[default_fs]
    mean_accuracy = rsl_accuracy.loc[(slice(None), 'accuracy'),].groupby(level=0).mean()[default_fs]
    mean_metric = rsl_metric.loc[(slice(None), metric, slice(None))].groupby(level=0).mean()[default_fs]

    all_fs = list(mean_metric.columns)
    
    rename_fs_dic = {fs: default_fs[fs][3] if (default_fs[fs][3] is not None) else fs for fs in all_fs}
    mean_metric = mean_metric.rename(columns=rename_fs_dic)
    mean_retention = mean_retention.rename(rename_fs_dic)
    mean_accuracy = mean_accuracy.rename(columns=rename_fs_dic)
    all_fs_rename = list(mean_metric.columns)
    
    all_fs.remove('all')
    all_fs_rename.remove('all')
    

    fig = plt.figure(figsize=(7, 4))
    ax13 = fig.add_subplot(122, projection='3d')
    ax11 = fig.add_subplot(221)  # add subplot into first position in a 2x2 grid (upper left)
    ax12 = fig.add_subplot(223)  # add to third position in 2x2 grid (lower left) and sharex with ax11

    ax11.invert_xaxis()
    ax11.invert_yaxis()
    ax12.invert_xaxis()

    plt.subplots_adjust(wspace=0, hspace=0.2)

    for i, fs in enumerate(all_fs_rename):  # plot each point + it's index as text above
        label = fs
        if default_fs[all_fs[i]][3] is not None:
            label = default_fs[all_fs[i]][3]
        ax11.scatter(mean_retention[fs], mean_metric.loc[model, fs], color=default_fs[all_fs[i]][0])
        
       
        # ax11.invert_yaxis()
        ax12.scatter(mean_retention[fs], mean_accuracy.loc[model, fs],  color=default_fs[all_fs[i]][0])

        ax13.scatter(mean_metric.loc[model, fs], mean_accuracy.loc[model, fs], mean_retention[fs],
                      color=default_fs[all_fs[i]][0])
        if anno:
            ax11.text(mean_retention[fs], mean_metric.loc[model, fs], label, size=size_anno, zorder=1, color='k')
            ax12.text(mean_retention[fs], mean_accuracy.loc[model, fs], label, size=size_anno, zorder=1, color='k')
            ax13.text(mean_metric.loc[model, fs], mean_accuracy.loc[model, fs], mean_retention[fs], label,
                      size=size_anno + 1,
                      zorder=1, color='k')
            # ax11.annotate(fs, (mean_retention[fs], mean_metric.loc[model, fs]))

    ax11.set_ylabel('Δ explanation', size=labelsize)
    ax12.set_ylabel('accuracy', size=labelsize)
    ax12.set_xlabel('retention', size=labelsize)

    ax13.set_xlabel('Δ explanation', size=labelsize)
    ax13.set_ylabel('accuracy', size=labelsize)
    ax13.set_zlabel('retention', size=labelsize)
    ax13.legend(all_fs_rename, loc=2, bbox_to_anchor=(1.2, 1))

    for ax in [ax11, ax12, ax13]:
        ax.tick_params(axis='both', which='major', labelsize=ticksize)
    return fig


## summary plot
def summary_comparison(dataset_id: int, explainer: str = 'lime', feature_names=None,
                       fss: list = default_fs, models: list = default_all_models, plot_size=None,
                       nb_col: int = None, col_length: int = 6, nb_row: int = None, row_height: Union[int, list] = 5,
                       show=False, display_bacc=False, title: str = None,
                       accuracy_df=None, rsl_path=default_rsl_path,
                       feature_names_size=13, wspace=None, hspace=0.2):
    if len(fss) * len(models) > nb_col * nb_row:
        raise Exception(f'{nb_col} columns, {nb_row} rows: not enough for {len(fss)} FS and {len(models)} models.')

    if type(row_height) is int:
        fig = plt.figure(figsize=(nb_col * col_length, nb_row * row_height), constrained_layout=True)
    elif type(row_height) is list:
        fig = plt.figure(figsize=(nb_col * col_length, sum(row_height)), constrained_layout=True)
        gs = gridspec.GridSpec(nb_row, nb_col, height_ratios=row_height)
    else:
        raise Exception(f"Type of row_height {row_height}, {type(row_height)} not accepted")

    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    n_m = len(models)
    for count_m, model in enumerate(models):
        f_names = get_ordered_fnames(dataset_id=dataset_id, model=model, feature_names=feature_names, rsl_path=rsl_path)
        for count_f, fs in enumerate(fss):
            if type(row_height) is int:
                ax = fig.add_subplot(nb_row, nb_col, count_f * n_m + count_m + 1)
            else:
                ax = plt.subplot(gs[count_f * n_m + count_m])

            summary_plot(dataset_id=dataset_id, model=model, fs=fs, explainer=explainer, feature_names=f_names,
                         show=show, color_bar=False,
                         plot_size=plot_size,
                         rsl_path=rsl_path, feature_names_size=feature_names_size)
            # plt.gcf().axes[-1].set_ylabel('')
            # plt.gcf().axes[-1].set_yticklabels([])

            plt.xlabel(f"{explainer.upper()} value (impact on model output)", size=15)

            display_name = default_fs[fs][3]
            if default_fs[fs][3] is None:
                display_name = fs

            accur = get_accuracy(dataset_id=dataset_id, fs=fs, model=model, accuracy_type='accuracy',
                                 accuracy_df=accuracy_df,
                                 rsl_path=rsl_path)
            ax_title = f"Model: {model}, FS: {display_name}, acc={'%.2f%%' % (accur * 100)}"
            if display_bacc:
                b_accur = get_accuracy(dataset_id=dataset_id, fs=fs, model=model, accuracy_type='balanced_accuracy',
                                       accuracy_df=accuracy_df, rsl_path=rsl_path)
                ax_title = f"Model: {model}, FS: {display_name}, acc={'%.2f%%' % (accur * 100)}, b_acc={'%.2f%%' % (b_accur * 100)}"

            ax.set_title(ax_title, size=20)

            count_f += 1
        count_m += 1
    fig_title = f"Summary plot of Dataset {dataset_id}" if title is None else title
    if title is not False:
        fig.suptitle(fig_title, size=20)
    return fig


def get_ordered_fnames(dataset_id: int, model: str,
                       explainer: str = 'lime', feature_names=None,
                       add_ranking=True, rsl_path=None) -> list:
    ex = get_explanation(dataset_id=dataset_id, model=model, fs='all', explainer=explainer, selected=False,
                         rsl_path=rsl_path)

    if feature_names is None:
        feature_names = ex.all_features

    feature_names = ['v' + str(i) + '_' + feature_names[i] for i in range(len(feature_names))]

    if not add_ranking: return feature_names

    order = np.argsort(np.mean(abs(ex.values), axis=0))[::-1].tolist()
    return [f"{feature_names[i]}_{order.index(i)}" for i in range(len(feature_names))]


def summary_plot(dataset_id: int, model: str, fs: str = None, candidate: tuple = None,
                 explainer: str = 'lime', feature_names=None, show=True, color_bar=False,
                 plot_size='auto', max_display=None, rsl_path=None,
                 feature_names_size=13):
    ex = get_explanation(dataset_id=dataset_id, model=model, fs=fs, candidate=candidate, explainer=explainer,
                         selected=True, rsl_path=rsl_path)

    if feature_names is not None and len(feature_names) == len(ex.all_features):
        f_names = np.array(feature_names)[ex.selected_index]
    else:
        f_names = ex.selected_names

    if max_display is None:
        max_display = len(f_names)

    return pseudo_shap.summary_legacy(ex.values, features=ex.data, feature_names=f_names, show=show,
                                      plot_size=plot_size,
                                      color_bar=False, max_display=max_display, feature_names_size=feature_names_size)
    #
    # return shap.summary_plot(ex.values, features=ex.data, feature_names=f_names, show=show, plot_size=plot_size,
    #                          color_bar=False, max_display=max_display)


# @deprecated
def display_heatmap_old(rsl_raw: pd.DataFrame, selection_rate_raw: pd.DataFrame,
                        accuracy_raw: pd.DataFrame, metric_arc: str,
                        fs_info=None, models=None, datasets: list = None,
                        metrics_f: dict = None,
                        display_best: bool = False, cluster: bool = False,
                        rename_axis: bool = True, title: str = None,
                        titlesize: int = 18, labelsize: int = 16, ticksize: int = 14):
    if models is None: models = default_all_models
    if metrics_f is None: metrics_f = default_metrics_dic

    ds_slice = slice(None) if datasets is None else datasets

    # aggregation
    metric_fname = metrics_f[metric_arc][0]

    data = rsl_raw.loc[slice(None), metric_fname, ds_slice]
    selection_rate = selection_rate_raw.loc[ds_slice]
    accuracy = accuracy_raw.loc[slice(None), slice(None), ds_slice]

    # fs selection
    if fs_info is None: fs_info = default_fs
    fs_order = [fs for fs in fs_info if fs in data.columns]
    data = data[fs_order]
    data_std = data.groupby(level=0).std()
    data = data.groupby(level=0).mean()

    """
    if display_stat and agg:
        stat = {'best': {},
                'worst': {}}
        for index, row in data.iterrows():
            if (agg == 'std') or np.isclose(np.max(data['all']), 0):
                best = row.drop('all').min()
                worst = row.max()
            else:
                best = row.drop('all').max()
                worst = row.min()
            best_fs = row[row == best].index
            worst_fs = row[row == worst].index
            for fs in best_fs:
                if fs in stat['best']:
                    stat['best'][fs] += 1/len(best_fs)
                else:
                    stat['best'][fs] = 1/len(best_fs)

            for fs in worst_fs:
                if fs in stat['worst']:
                    stat['worst'][fs] += 1/len(worst_fs)
                else:
                    stat['worst'][fs] = 1/len(worst_fs)
        print(agg, metric_arc, stat)
    """

    # clustermap style

    norm = mcolors.Normalize(vmin=data.values.min(), vmax=data.values.max())

    cmap = 'rocket'
    if np.isclose(np.max(data['all']), 0):
        cmap = 'rocket_r'
    if metric_arc == 'comp_rinf_w' and data.values.min() < 0:
        cmap = 'vlag'
        norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=data.values.min(), vmax=data.values.max())

    color_map = cm.get_cmap(cmap)

    norm_std = mcolors.Normalize(vmin=data_std.values.min(), vmax=data_std.values.max())

    # data = data.rename(columns=dic_rename_fs, index=dic_rename_model)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax = sns.heatmap(data, cmap=cmap, annot=True, alpha=0.2, linewidths=0.1, cbar=False)

    dic_x = {inf.get_text(): inf.get_position()[0] for inf in ax.xaxis.get_ticklabels()}
    dic_y = {inf.get_text(): inf.get_position()[1] for inf in ax.yaxis.get_ticklabels()}
    circles = [plt.Circle((dic_x[fs], dic_y[model]), radius=data_std.loc[model, fs] / data_std.values.max() / 2)
               for fs, model in itertools.product(data.columns, data.index)]
    col = PatchCollection(circles, array=data.values.flatten(order='F'), cmap=cmap)
    ax.add_collection(col)

    cb = ax.figure.colorbar(mappable=cm.ScalarMappable(norm, cmap=cmap))
    cb.outline.set_linewidth(0)

    if rename_axis:
        rate_order = [corresponding_fs_names_all_models(selection_rate, accuracy, fs, dss=datasets, fs_info=fs_info) for
                      fs in fs_order]
        model_order = [corresponding_models_names_w_accuracy(model, selection_rate, accuracy) for model in models]

        ax.set_xticklabels(rate_order, rotation=45, ha='right', size=ticksize)
        ax.set_yticklabels(model_order, rotation='horizontal', size=ticksize)
        # ax.text(-ax.xaxis.get_ticklabels()[1].get_position()[0], -ax.yaxis.get_ticklabels()[-1].get_position()[1],'dhuchizvcuhiz')
        ax.set_xlabel('FS method | average retention rate | average accuracy | number of datasets involved',
                      size=labelsize)
        ax.set_ylabel('model | average accuracy', size=labelsize)
    else:
        rename_fs = [fs_info[fs][3] if (fs_info[fs][3] is not None) else fs for fs in fs_order]
        ax.set_xticklabels(rename_fs, rotation=25, size=ticksize)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, size=ticksize)

        ax.set_xlabel('feature selection method', size=labelsize)
        ax.set_ylabel('model', size=labelsize)

    if title is None: title = metric_fname
    ax.set_title(title, size=titlesize)

    return ax


# @deprecated
def summary_comparison_old(dataset_id: int, explainer: str = 'lime', feature_names=None,
                           fss: list = None, models: list = None,
                           nb_col: int = None, col_length: int = 6, nb_row: int = None, row_height: int = 5,
                           plot_size=None,
                           show=False, display_bacc=False, title: str = None,
                           accuracy_df=None, rsl_path=default_rsl_path,
                           feature_names_size=13, wspace=None):
    if fss is None: fss = default_fs
    if models is None: models = default_all_models

    if len(fss) * len(models) > nb_col * nb_row:
        raise Exception(f'{nb_col} columns, {nb_row} rows: not enough for {len(fss)} FS and {len(models)} models.')

    count_m = 1
    n_m = len(models)
    fig = plt.figure(figsize=(nb_col * col_length, nb_row * row_height), constrained_layout=True)
    plt.subplots_adjust(wspace=wspace, hspace=0.2)

    for model in models:
        f_names = get_ordered_fnames(dataset_id=dataset_id, model=model, feature_names=feature_names, rsl_path=rsl_path)
        count_f = 0
        for fs in fss:
            ax = fig.add_subplot(nb_row, nb_col, count_f * n_m + count_m)
            summary_plot(dataset_id=dataset_id, model=model, fs=fs, explainer=explainer, feature_names=f_names,
                         show=show, color_bar=False,
                         plot_size=plot_size,
                         rsl_path=rsl_path, feature_names_size=feature_names_size)
            # plt.gcf().axes[-1].set_ylabel('')
            # plt.gcf().axes[-1].set_yticklabels([])

            plt.xlabel(f"{explainer.upper()} value (impact on model output)", size=15)

            display_name = default_fs[fs][3]
            if default_fs[fs][3] is None:
                display_name = fs

            accur = get_accuracy(dataset_id=dataset_id, fs=fs, model=model, accuracy_type='accuracy',
                                 accuracy_df=accuracy_df,
                                 rsl_path=rsl_path)
            ax_title = f"Model: {model}, FS: {display_name}, acc={'%.2f%%' % (accur * 100)}"
            if display_bacc:
                b_accur = get_accuracy(dataset_id=dataset_id, fs=fs, model=model, accuracy_type='balanced_accuracy',
                                       accuracy_df=accuracy_df, rsl_path=rsl_path)
                ax_title = f"Model: {model}, FS: {display_name}, acc={'%.2f%%' % (accur * 100)}, b_acc={'%.2f%%' % (b_accur * 100)}"

            ax.set_title(ax_title, size=20)

            count_f += 1
        count_m += 1
    fig_title = f"Summary plot of Dataset {dataset_id}" if title is None else title
    if title is not False:
        fig.suptitle(fig_title, size=20)
    return fig


# @deprecated
def summary_row_height(dataset_id: int, explainer: str = 'lime', feature_names=None,
                       fss: list = None, models: list = None,
                       nb_col: int = None, col_length: int = 6, nb_row: int = None, height_ratios: list = None,
                       plot_size=None,
                       show=False, display_bacc=False, title: str = None,
                       accuracy_df=None, rsl_path=default_rsl_path,
                       feature_names_size=13, wspace=None):
    if fss is None: fss = default_fs
    if models is None: models = default_all_models

    if len(fss) * len(models) > nb_col * nb_row:
        raise Exception(f'{nb_col} columns, {nb_row} rows: not enough for {len(fss)} FS and {len(models)} models.')

    n_m = len(models)
    fig = plt.figure(figsize=(nb_col * col_length, sum(height_ratios) / 2), constrained_layout=True)
    gs = gridspec.GridSpec(nb_row, nb_col, height_ratios=height_ratios)
    plt.subplots_adjust(wspace=wspace, hspace=0.2)

    for count_m, model in enumerate(models):
        f_names = get_ordered_fnames(dataset_id=dataset_id, model=model, feature_names=feature_names, rsl_path=rsl_path)
        for count_f, fs in enumerate(fss):
            # ax = fig.add_subplot(nb_row, nb_col, count_f * n_m + count_m)
            ax = plt.subplot(gs[count_f * n_m + count_m])
            summary_plot(dataset_id=dataset_id, model=model, fs=fs, explainer=explainer, feature_names=f_names,
                         show=show, color_bar=False,
                         plot_size=plot_size,
                         rsl_path=rsl_path, feature_names_size=feature_names_size)
            # plt.gcf().axes[-1].set_ylabel('')
            # plt.gcf().axes[-1].set_yticklabels([])

            plt.xlabel(f"{explainer.upper()} value (impact on model output)", size=15)

            display_name = default_fs[fs][3]
            if default_fs[fs][3] is None:
                display_name = fs

            accur = get_accuracy(dataset_id=dataset_id, fs=fs, model=model, accuracy_type='accuracy',
                                 accuracy_df=accuracy_df,
                                 rsl_path=rsl_path)
            ax_title = f"Model: {model}, FS: {display_name}, acc={'%.2f%%' % (accur * 100)}"
            if display_bacc:
                b_accur = get_accuracy(dataset_id=dataset_id, fs=fs, model=model, accuracy_type='balanced_accuracy',
                                       accuracy_df=accuracy_df, rsl_path=rsl_path)
                ax_title = f"Model: {model}, FS: {display_name}, acc={'%.2f%%' % (accur * 100)}, b_acc={'%.2f%%' % (b_accur * 100)}"

            ax.set_title(ax_title, size=20)

    fig_title = f"Summary plot of Dataset {dataset_id}" if title is None else title
    if title is not False:
        fig.suptitle(fig_title, size=20)
    return fig


def hold_place():
    print('attention !')





































