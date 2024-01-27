# Validate the potential effects in given period and regions
# This version contains no external dependencies in case they are not available on ali cloud
#

import json
import datetime

from copy import deepcopy
from pathlib import Path
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


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
    assert f.exists(), f'FileNotFound: {str(f)}'
    conf = json.loads(f.read_text(encoding='utf8'))
    return conf


# TODO: in case the memory cannot load up the whole dataset,
#       we have to do the loading recursively and split them in chunks,
#       and save intermediary datafiles sorted by `City`.
def read_data(f: Path,
              encoding: str = 'utf8',
              tempfolder: Path = Path('temp'),
              chunk: int = 10e5) -> pd.DataFrame:
    """
    Read data from text files by lines.

    :param f: filepath of saved datafile, default for a csv file
    :param encoding: default for `utf8`
    :param chunk: in case the file is too big, use chuck to recursively load
    :return: a standard pandas dataframe
    """
    f = Path(f)  # convert string into Path object
    assert f.exists(), f'FileNotFound: {str(f)}'

    # write iteratively into a textfile by city
    tempfolder = Path(tempfolder)
    tempfolder.mkdir(exist_ok=True, parents=True)  # create the temp folder if it's not existed

    def iterate_write(directory: Path,
                      df: pd.DataFrame,
                      city_key: str = 'csr_area',
                      encoding: str = encoding):
        """An iterative writer splitting big dataframe into datafiles by cities"""
        for city, g in df.groupby(city_key):
            print(f'Read city: {city} with {len(g)} lines of data')
            file = directory / f'{city}.txt'
            existed = file.exists()
            # use append mode to write data in lines
            with file.open('a', encoding=encoding) as f:
                if not existed:
                    col = g.columns.to_numpy().reshape((1, g.shape[1]))
                    np.savetxt(f, col, delimiter=',', fmt='%s')

                # DO NOT duplicate columns if the file is already existed
                np.savetxt(f, g.values, delimiter=',', fmt='%s')

            print(f'Finish city: {city} at {str(file)}')

    idx = 0  # since you do not know how many lines in total, use while
    count = 0  # count how many lines in total
    while True:
        print(f'Load data: {idx} chunks finished')
        # NB. when skipsrows starts with `1`, it keeps the header!
        d = pd.read_csv(f, skiprows=range(1, idx * chunk + 1), header=[0], nrows=chunk, delimiter=',')
        if d.empty:  # no more data
            break

        idx += 1  # the chunk index iteratively grows
        count += len(d)
        iterate_write(tempfolder, d, encoding=encoding)

    print(f'Completed reading {count} lines of data!')


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


def processing(event: dict,
               ord_goods: list,
               ada_goods: list,
               parallel_span: int = 3) -> None:
    """
    Data processing procedures where event-based data being labelled and goods being split.
    Note that, the structure of data should be like:

    :param event: an event json indexed by events
    :param ord_goods: a list of ordinary goods
    :param ada_goods: a list of adaptative goods
    :param parallel_span: a span of parallel test, default as 3
    :return: a dict of `event`:`dataframe` with additional columns:
             1. treatment, 2. post, 3. orient
    """
    # NB. we assume `data` is the original dataset
    assert event, f'Event must be specified for validation'

    # use event params to extract data from cities
    key_time = event['key_time']  # configured
    key_goods = event['key_goods']  # configured, can be changed!
    for e, param in event.items():
        if isinstance(param, str):  # skip those non-configured params
            continue

        print(f'Render event: {e}')
        # NB. DO NOT change the format of dates
        fmt = '%Y%m%d' if param['freq'] == 'days' else '%Y%m'
        span = relativedelta(**{param['freq']: param['span']})
        cities = param['treated'] + param['untreated']
        # create dataframes for each event
        r = pd.DataFrame()
        for c in cities:
            file = Path('temp') / f'{c}.txt'
            assert file.exists(), f'CityNotFound: {c}'
            # read data from textfiles
            d = pd.read_csv(file, delimiter=',', header=[0])
            d[key_time] = pd.to_datetime(d[key_time], format=fmt)

            # step 1: specify time period, only
            start = datetime.datetime.strptime(param['period'][0], fmt)
            lower = start - span
            end = datetime.datetime.strptime(param['period'][1], fmt)
            upper = end + span
            time_mask = (d[key_time] >= lower) & (d[key_time] <= upper)  # a series of True/False
            d = d[time_mask]  # dramatically reduce the size of dataframe

            # step 2: identify different goods
            ord_mask = d[key_goods].isin(ord_goods)
            ada_mask = d[key_goods].isin(ada_goods)
            d = d[ord_mask | ada_mask]

            # step 3: specify labels
            d['adaptation'] = 1  # NB. masked dataframe does not contain any other goods
            d.loc[ord_mask, 'adaptation'] = 0
            d['treatment'] = 1 if c in param['treated'] else 0
            d['post'] = 0  # create shock period
            d.loc[(d[key_time] >= start) & (d[key_time] <= end), 'post'] = 1

            # step 4: add parallel trend variants
            # ... add post[-1], post[-2] ..., post[1], post[2] ... for parallel tests
            for i in range(1, parallel_span + 1):
                back = start - relativedelta(**{param['freq']: i})  # ... use `i` instead of 1
                ahead = start + relativedelta(**{param['freq']: i})

                # ... add a key to parallel periods
                key = f'pre{i}'  # backward
                d[key] = 0
                d.loc[d[key_time] == back, key] = 1
                key = f'post{i}'
                d[key] = 0
                d.loc[d[key_time] == ahead, key] = 1

            # step 5: append dataframes by cities
            r = pd.concat([r, d], axis=0)  # axis=0, by row

        # output event-based data
        r[key_time] = r[key_time].apply(lambda x: x.strftime(fmt))
        r.to_csv(f'temp/{e}.txt', sep=',', index=False, encoding='utf8')
        print(f'Finish rendering event: {e}!')


def trend(event: dict,
          checking_keys: list,
          save: Path = None) -> pd.DataFrame:
    """
    Summarise trending results of either adaptation/ordinary goods

    :param event: an event json indexed by events
    :param checking_keys: keys of checking interest
    :param save: a Path-like object or None, default for `./checking-first-<timestamp>.csv`
    :return: a dataframe contains all trending results
    """
    key_time = event['key_time']  # configured
    colors = {"lightred": "#D5695D",
              "lightgreen": "#65A479",
              "lightblue": "#5D8CA8",
              "lightyellow": "#D3BA68",
              "darkred": "#B1283A",
              "darkblue": "#016392",
              "darkyellow": "#BE9C2E",
              "darkgreen": "#098154",
              "gray": "#808080"}

    output = pd.DataFrame()
    for e, param in event.items():
        if isinstance(param, str):  # skip those non-configured params
            continue

        # load event-based dataframe
        df = pd.read_csv(f'temp/{e}.txt', header=[0], delimiter=',')
        df[[key_time]] = df[[key_time]].astype(str)
        o = pd.DataFrame()
        for idx, c in enumerate(checking_keys):
            # summarise data
            uni = df[[key_time]].drop_duplicates().reset_index(drop=True)
            for k in ['adaptation', 'treatment']:
                # ... merging with true samples
                s = df.loc[df[k] == 1, [key_time, c]].groupby(key_time).mean().reset_index()
                s.columns = [key_time] + [f'{c}|{k}=1']
                uni = uni.merge(s, on=key_time, how='left')
                # ... merging with false samples
                s = df.loc[df[k] == 0, [key_time, c]].groupby(key_time).mean().reset_index()
                s.columns = [key_time] + [f'{c}|{k}=0']
                uni = uni.merge(s, on=key_time, how='left')

            # complete merging and fill NaNs with zero
            uni = uni.fillna(0)
            melt = uni.melt(id_vars=[key_time], value_vars=uni.columns.values[1:])
            # load event params
            evt = event[e]
            ticks = uni[key_time].tolist()
            # find out the max/min values
            # plot the trending
            fig = plt.figure(figsize=(8, 6))
            cvalues = list(colors.values()) * 10  # in case we have many checking keys
            for i, c in enumerate(uni.columns.values[1:]):
                x_range = range(len(uni))
                plt.plot(x_range, uni[c],
                         color=cvalues[i],
                         marker='o',
                         markersize=6,
                         label=c)

            # draw vert lines for the shock
            start, end = evt['period']
            plt.vlines(x=ticks.index(start), ymin=0, ymax=max(melt['value']), colors='grey', linestyles='--')
            plt.vlines(x=ticks.index(end), ymin=0, ymax=max(melt['value']), colors='grey', linestyles='--')
            plt.fill_betweenx(y=(0, max(melt['value'])),
                              x1=ticks.index(start),
                              x2=ticks.index(end),
                              facecolor='grey', alpha=0.15)
            # change ticks
            plt.xticks(range(0, len(uni)), labels=uni[key_time].astype(str), rotation=45)
            plt.legend(fontsize=9)
            plt.tight_layout()
            plt.margins(0.01)
            fig.savefig(save / f'trend-{e}-{c}-{get_timestamp()}.png', dpi=200, format='png')
            # append uni tables
            if idx > 0:
                uni = uni.iloc[:, 1:]

            o = pd.concat([o, uni], axis=1)  # append by cols

        # append by columns
        output = pd.concat([output, o], axis=0)

    # finished
    output = output.to_csv(save / f'trend-{get_timestamp()}.csv', encoding='utf8')
    return output


def did(event: dict,
        if_plot: bool = True,
        parallel_span: int = 3,
        save: Path = None) -> pd.DataFrame:
    """
    Difference-in-difference model for pre-validation with not-well-processed data

    :param result: a dict of event and its corresponding dataset
    :param if_plot: True=plotting, False=stop plotting
    :param parallel_span: a span of parallel test, default as 3
    :param save: a Path-like object or None, default for `./checking-first-<timestamp>.csv`
    :return: a dataframe with regression results
    """
    save = Path() if save is None else Path(save)
    # NB. the following test should be executed event by event
    for e, param in event.items():
        if isinstance(param, str):  # skip those non-configured params
            continue

        data = pd.read_csv(f'temp/{e}.txt', delimiter=',', header=[0])
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

        # plotting the parallel trend
        if if_plot:
            fig = plt.figure(figsize=(8, 6))
            x_range = range(parallel_span * 2 + 1)  # if span=3, will have 7 points
            # 1. pooled OLS
            plt.plot(x_range, m2.params[1:], linewidth=2, c='blue')
            plt.hlines(y=0, xmin=0, xmax=parallel_span * 2 + 1)
            plt.ylabel('Coef.', size=16)
            plt.title('Pooled OLS Parallel Test', size=16)
            plt.xticks(x_range,
                       syntax.replace('treatmentx', '').replace(' ', '').split('+'),
                       rotation=45)
            fig.savefig(str(save / f'validate-did-{get_timestamp()}.png'), dpi=200, format='png')


def main(datafile: Path,
         eventfile: Path,
         ord: list,
         ada: list,
         save: Path,
         checklist: list,
         chunk: int = 10e5,
         parallel_span: int = 3,
         if_plot: bool = True):
    read_data(datafile, chunk=chunk)
    event = load_config(eventfile)
    processing(event, ord, ada)
    save.mkdir(parents=True, exist_ok=True)
    trend(event, checking_keys=checklist, save=save)
    did(event, if_plot=if_plot, parallel_span=parallel_span, save=save)


if __name__ == '__main__':
    # test 1: load data by lines and split them into files
    testfile = Path('data') / 'demo_data_test.txt'
    read_data(testfile, chunk=100)

    # test 2: processing
    # ['生活用品及服务', '食品烟酒', '医疗保健', '居住', '其他用品和服务',
    #  '交通和通信', '衣着', '教育文化和娱乐']
    event = load_config(Path('event.json'))
    ord = ['生活用品及服务', '食品烟酒', '衣着']
    ada = ['医疗保健', '居住']
    processing(event, ord, ada)

    # test 3: trending
    save = Path('results')
    save.mkdir(parents=True, exist_ok=True)
    o = trend(event, checking_keys=['amount', 'ord_cnt'], save=save)

    # test 4: diff-in-diff with Pooled OLS
    checklist = ['amount']
    did(event,
        if_plot=True,
        parallel_span=3,
        save=save)

    # production operation
    # configuration: city_level.txt, 20110101-20191231
    file = Path('D:/air_pollution') / 'input' / 'city_level' / 'city_category.txt'  # the target file
    efile = Path('event-dev.json')
    main(file, efile, ord, ada, save, chunk=10e5)
