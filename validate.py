# Validate the potential effects in given period and regions
#
# example: https://lost-stats.github.io/Model_Estimation/Research_Design/event_study.html
#
# Columns: (for counties, replace `city` by `county` in the following specs)
# ç	年份-月份
# export_city_id	出口城市行政区划代码
# export_city_name	出口城市名称
# import_city_id	进口城市行政区划代码
# import_city_name	进口城市名称
# gmv	成交金额
# deal	成交订单量
# splnum	商家数量
# csrnum	消费者数量
# itemnum	商品数量
# cate_id	商品类目代码
# cate_name	商品类目名称

import json
import datetime

from copy import deepcopy
from pathlib import Path
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import linearmodels as lm
import matplotlib.pyplot as plt

from loguru import logger


# TODO: utility functions should be put in the utils.py script
def get_timestamp(fmt: str = '%y%m%d%H%M%S') -> str:
    # default timezone for Beijing/China
    beijing = datetime.timezone(datetime.timedelta(hours=8))
    return datetime.datetime.now(beijing).strftime(fmt)


def load_config(f: Path) -> dict:
    """
    Load config file and return a dict.

    :param f: filepath of the config file
    :return: a dict of configuration
    """
    f = Path(f)  # convert string into Path object
    assert f.exists(), logger.warning(f'FileNotFound: {str(f)}')
    conf = json.loads(f.read_text(encoding='utf8'))
    return conf


def read_data(f: Path, encoding: str = 'utf8') -> pd.DataFrame:
    """
    Read data from given csv files.

    :param f: filepath of saved datafile, default for a csv file
    :param encoding: default for `utf8`
    :return: a standard pandas dataframe
    """
    f = Path(f)  # convert string into Path object
    assert f.exists(), logger.warning(f'FileNotFound: {str(f)}')
    df = pd.read_csv(f, encoding=encoding)
    logger.info(f'Loaded: {len(df)} lines of records')
    return df


def update_config(tax_file: Path = Path('taxonomy.json'),
                  event_file: Path = Path('event.json'),
                  goods_list_file: Path = Path('data') / 'goods_list.xlsx') -> None:
    """
    Update configured taxonomy and event json files by goods list data sheet.

    :param tax_file: file path for taxonomy config
    :param event_file: file path for even config
    :param goods_list_file: file path for goods list Excel file which might be updated some time
    :return: None
    """
    goods = pd.read_excel(goods_list_file, engine='openpyxl', dtype=str)
    tax = load_config(tax_file)  # mainly update its second category: index

    # event = load_config(event_file)  # TODO: not yet need updating

    # match with goods names
    def match(aspect: str):
        idx = {}
        for k, v in tax[aspect].items():
            # use cate_level2_id as the unique index
            sub = goods[goods.cate_level1_name.isin(v)]
            sub.index = sub['cate_level2_id']
            sub = sub.drop(columns=['cate_level2_id'])
            # develop the index
            for i, it in sub.iterrows():
                idx[i] = it.values.tolist()

        # update index
        tax[aspect]['index'] = deepcopy(idx)
        return tax

    # update indexes
    tax = match('adaptation')
    tax = match('ordinary')
    # saving output
    tax_file.write_text(json.dumps(tax), encoding='utf8')


def processing(data: pd.DataFrame,
               taxonomy: dict,
               event: dict,
               parallel_span: int = 3) -> dict:
    """
    Data processing procedures where event-based data being labelled and goods being split.
    Note that, the structure of data should be like:

        # yyyymm            timestamp, yyyymmdd daily instead
        # export_city_id	zone code id for export city
        # export_city_name	zone name for export city
        # import_city_id	zone code id for import city
        # import_city_name	zone name for import city
        # gmv	            transaction volume in RMB
        # deal              confirmed number of orders
        # splnum	        no. of dealers
        # csrnum	        no. of consumers
        # itemnum	        no. of goods
        # cate_id	        code id of goods category (should be confirmed if it's 2nd level codes)
        # cate_name	        name of goods category

    :param data: the original dataframe on the REMOTE cloud
    :param taxonomy: a taxonomy json indexed by goods names and codes from alibaba data
    :param event: an event json indexed by events
    :param parallel_span: a span of parallel test, default as 3
    :return: a dict of `event`:`dataframe` with additional columns:
             1. treatment, 2. post, 3. orient
    """
    # NB. we assume `data` is the original dataset
    assert event, f'Event must be specified for validation'

    # ensure data formats are consistent
    data['cate_id'] = data['cate_id'].astype(str)  # SHOULD NOT be integers and keep it in the form of dataframe
    # update and reshape the dataframe and ensure goods in data are properly labelled
    combined = pd.DataFrame()
    for orient, aspect in taxonomy.items():
        # the 1st-level idxes is the zero-idx item of `index`'s values
        asp = pd.DataFrame(aspect['index']).T
        query_idx = asp.index.values  # NB. the 2nd-level idxes
        idx_of_data = data['cate_id'].isin(query_idx)  # index-like result
        # concat dataframes horizontally
        sub = data[idx_of_data].copy(deep=True)
        sub['orient'] = orient  # ... basically, adaptation/ordinary
        combined = pd.concat([combined, sub], axis=0)  # axis=0, by row

    # rebuild data based on events
    results = {}
    for e, param in event.items():
        if param['freq'] == 'days':
            fmt = '%Y%m%d'
            key_time = 'yyyymmdd'
        else:  # ... following what is specified in the original dataset
            fmt = '%Y%m'
            key_time = 'yyyymm'

        # split data by events: key-value = event-name: event params
        # TODO: here might make mistakes if the given data have biased date format!
        comb_time = pd.to_datetime(combined[key_time], format=fmt)
        combined[key_time] = comb_time
        comb_city = combined['import_city_name'].astype(str)
        combined['import_city_name'] = comb_city
        span = relativedelta(**{param['freq']: param['span']})

        # step 1: specify time period, only
        start = datetime.datetime.strptime(param['period'][0], fmt)
        lower = start - span
        end = datetime.datetime.strptime(param['period'][1], fmt)
        upper = end + span
        period_mask = (comb_time >= lower) & (comb_time <= upper)  # a series of True/False

        # step 2: specify cities
        treated = list(param['treated'])
        untreated = list(param['untreated'])
        city_mask = comb_city.isin(treated + untreated)  # specify all the cities first

        # step 3: data processing
        res = combined[period_mask & city_mask]
        res['treatment'] = 0  # default value as zero
        # add treatment labels
        res.loc[res['import_city_name'].isin(treated), 'treatment'] = 1
        # add post labels
        res['post'] = 0
        res.loc[(res[key_time] >= start) & (res[key_time] <= end), 'post'] = 1
        # ... add post[-1], post[-2] ..., post[1], post[2] ... for parallel tests
        for i in range(1, parallel_span + 1):
            back = start - relativedelta(**{param['freq']: i})  # ... use `i` instead of 1
            ahead = start + relativedelta(**{param['freq']: i})

            # ... add a key to parallel periods
            key = f'pre{i}'  # backward
            res[key] = 0
            res.loc[res[key_time] == back, key] = 1
            key = f'post{i}'
            res[key] = 0
            res.loc[res[key_time] == ahead, key] = 1

        # output the end-use data
        results[e] = res

    return results


def checking(df: pd.DataFrame,
             taxonomy: dict,
             checking_keys: list = None,
             save: Path = None) -> tuple:
    """
    Check how many records of data we have for specific columns.

    :param df: the processed data after processing
    :param taxonomy: a taxonomy json indexed by goods names and codes from alibaba data
    :param checking_keys: keys of checking interest
    :param save: a Path-like object or None, default for `./checking-first-<timestamp>.csv`
    :return: an added post-checking dataframe
    """
    # NB. Use pandas.describe() to check,
    # the checking should be category-based and city-based within an array.
    if checking_keys is None:
        checking_keys = ['gmv', 'deal', 'csrnum', 'itemnum']

    # 1st-level categories
    def group_by(dt: pd.DataFrame, key: str, i: str = 'cate_id'):
        c = dt[[i, key]].groupby(i).count().reset_index()
        c.columns = [i, f'{key}_count']
        m = dt[[i, key]].groupby(i).mean().reset_index()
        m.columns = [i, f'{key}_mean']
        s = dt[[i, key]].groupby(i).sum().reset_index()
        s.columns = [i, f'{key}_sum']
        mi = dt[[i, key]].groupby(i).min().reset_index()
        mi.columns = [i, f'{key}_min']
        c[f'{key}_mean'] = m[f'{key}_mean']
        c[f'{key}_sum'] = s[f'{key}_sum']
        c[f'{key}_min'] = mi[f'{key}_min']
        return c

    # reshape by keys
    first = pd.DataFrame()
    for k in checking_keys:
        g = group_by(df, k, i='cate_id')
        first = pd.concat([first, g], axis=1)

    # 2nd-level categories
    foo = taxonomy['adaptation']['index']
    foo.update(taxonomy['ordinary']['index'])
    idx = {i: j[0] for i, j in foo.items()}
    df['cate_id_2nd'] = df['cate_id'].apply(lambda x: idx[x])
    # reshape by keys
    second = pd.DataFrame()
    for k in checking_keys:
        g = group_by(df, k, i='cate_id_2nd')
        second = pd.concat([second, g], axis=1)

    # save outputs
    save = Path() if save is None else Path(save)
    first.to_csv(save / f'checking-first-{get_timestamp()}.csv', encoding='utf8')
    second.to_csv(save / f'checking-second-{get_timestamp()}.csv', encoding='utf8')
    return first, second


def did(result: dict,
        if_plot: bool = True,
        parallel_span: int = 3,
        multi_index: list = None,
        save: Path = None) -> pd.DataFrame:
    """
    Difference-in-difference model for pre-validation with not-well-processed data

    :param result: a dict of event and its corresponding dataset
    :param if_plot: True=plotting, False=stop plotting
    :param parallel_span: a span of parallel test, default as 3
    :param multi_index: two indexes for panel OLS
    :param save: a Path-like object or None, default for `./checking-first-<timestamp>.csv`
    :return: a dataframe with regression results
    """
    if multi_index is None:
        multi_index = ['import_city_name', 'yyyymm']

    save = Path() if save is None else Path(save)

    # NB. the following test should be executed event by event
    for e, data in result.items():
        # doing logarithm conversion before modelling
        data['lngmv'] = data['gmv'].apply(lambda x: np.log(x + 1))
        data['treatmentxpost'] = data['treatment'] * data['post']
        # 1. baseline model spec: pooled OLS
        m1 = smf.ols('lngmv ~ treatmentxpost', data=data).fit()
        f = save / f'pooled-did-{e}-{get_timestamp()}.txt'
        f.write_text(m1.summary().as_text(), encoding='utf8')

        # NB. the default test covers the first and last 3 periods of shock and,
        #     to modify parallel params, specify `parallel_span` in `processing()`.
        # 2. parallel test using pooled OLS
        back, ahead = [], []
        for i in range(1, parallel_span + 1):
            data[f'treatmentxpre{i}'] = data[f'pre{i}'] * data['treatment']
            data[f'treatmentxpost{i}'] = data[f'post{i}'] * data['treatment']
            back += [f'treatmentxpre{i}']
            ahead += [f'treatmentxpost{i}']

        back.reverse()  # from -1, -2, ... to ... -2, -1!
        syntax = '+'.join(back) + ' + treatmentxpost + ' + '+'.join(ahead)
        m2 = smf.ols(f'lngmv ~ {syntax}', data=data).fit()
        f = save / f'pooled-parallel-{e}-{get_timestamp()}.txt'
        f.write_text(m2.summary().as_text(), encoding='utf8')

        # 1. baseline model spec: panel OLS with two-way fixed effect
        # lm requires a MultiIndex for panel data
        data = data.set_index(multi_index)
        m1_tw = lm.PanelOLS(data['lngmv'], data['treatmentxpost'], entity_effects=True, time_effects=True)
        fit = m1_tw.fit(cov_type='clustered', cluster_entity=True)
        f = save / f'panel-did-{e}-{get_timestamp()}.txt'
        f.write_text(fit.summary.as_text(), encoding='utf8')

        # 2. parallel test using panel OLS
        x = back + ['treatmentxpost'] + ahead
        m2_tw = lm.PanelOLS(data['lngmv'], data[x], entity_effects=True, time_effects=True, check_rank=False)
        fit2 = m2_tw.fit(cov_type='clustered', cluster_entity=True)
        f = save / f'panel-parallel-{e}-{get_timestamp()}.txt'
        f.write_text(fit2.summary.as_text(), encoding='utf8')

        # plotting the parallel trend
        if if_plot:
            fig = plt.figure(figsize=(12, 12))
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)
            x_range = range(parallel_span * 2 + 1)  # if span=3, will have 7 points
            # 1. pooled OLS
            ax1.plot(x_range, m2.params[1:], linewidth=2, c='blue')
            ax1.hlines(y=0, xmin=0, xmax=parallel_span * 2 + 1)
            ax1.set_ylabel('Coef.', size=16)
            ax1.set_title('Pooled OLS Parallel Test', size=16)
            ax1.set_xticks(x_range, [])

            # 2. panel OLS
            ax2.plot(x_range, fit2.params, linewidth=2, c='purple')
            ax2.hlines(y=0, xmin=0, xmax=parallel_span * 2)
            ax2.set_xlabel('Treatment x Pre/Post', size=16)
            ax2.set_ylabel('Coef.', size=16)
            ax2.set_title('Panel OLS Parallel Test', size=16)
            ticks = [t.replace('treatmentx', '') for t in x]
            ax2.set_xticks(x_range, ticks, fontsize=14)
            fig.savefig(str(save / f'validate-did-{get_timestamp()}.png'), dpi=200, format='png')


if __name__ == '__main__':
    # create directory
    save = Path('results')
    save.mkdir(parents=True, exist_ok=True)

    # general param
    parallel_span = 3

    # test 1: processing
    data = pd.read_excel('data/demo_data_modified.xlsx', engine='openpyxl')
    taxonomy = load_config(Path('taxonomy.json'))
    event = load_config(Path('event.json'))
    result = processing(data, taxonomy, event, parallel_span=parallel_span)

    # test 2: checking
    checklist = ['gmv']
    f, s = checking(result['测试用'], taxonomy, checklist, save=save)

    # test 3: did validation
    multi_index = ['import_city_name', 'yyyymm']
    did(result,
        if_plot=True,
        parallel_span=parallel_span,
        multi_index=multi_index,
        save=save)
