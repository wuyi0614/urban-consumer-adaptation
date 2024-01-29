# Validate the potential effects in given period and regions
# This version contains no external dependencies in case they are not available on ali cloud
#

import os
import json
import shutil
import datetime
import logging

from pathlib import Path
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# general config of logging
logging.basicConfig(format='%(asctime)s-[line:%(lineno)d]: %(message)s',
                    level=logging.INFO)


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


def iterate_write(directory: Path,
                  df: pd.DataFrame,
                  city_list: list = [],
                  city_key: str = 'csr_area',
                  encoding: str = 'utf8'):
    """An iterative writer splitting big dataframe into datafiles by cities"""
    ci = []
    for city, g in df.groupby(city_key):
        if city not in city_list:
            continue

        ci += [city]  # record how many cities have been written
        file = directory / f'{city}.txt'
        existed = os.path.exists(file)

        # use append mode to write data in lines
        with file.open('a', encoding=encoding) as f:
            if not existed:  # the first line of columns
                col = g.columns.to_numpy().reshape((1, g.shape[1]))
                np.savetxt(f, col, delimiter=',', fmt='%s')

            # DO NOT duplicate columns if the file is already existed
            np.savetxt(f, g.values, delimiter=',', fmt='%s')

        logging.info(f'Write {len(g)} lines into {city}!')
    # completed
    return ci


# TODO: in case the memory cannot load up the whole dataset,
#       we have to do the loading recursively and split them in chunks,
#       and save intermediary datafiles sorted by `City`.
def read_data(f: Path,
              event: dict,
              encoding: str = 'utf8',
              tempfolder: Path = Path('temp'),
              chunk: int = 10e5) -> None:
    """
    Read data from text files by lines.

    :param f: filepath of saved datafile, default for a csv file
    :param event: an event json indexed by events
    :param encoding: default for `utf8`
    :param tempfolder: temporary folder for data storage
    :param chunk: in case the file is too big, use chuck to recursively load
    :return: a standard pandas dataframe
    """
    f = Path(f)  # convert string into Path object
    assert f.exists(), f'FileNotFound: {str(f)}'

    # specify cities
    cities = []
    for _, param in event.items():
        if isinstance(param, str):  # skip those non-configured params
            continue

        cities += param['treated'] + param['untreated']
    # complete city loading
    logging.info(f'Found {len(cities)} cities in the config! They are: {cities}')

    # before running the iterative writing
    for c in cities:
        fi = tempfolder / f'{c}.txt'
        if os.path.isfile(str(fi)):
            print(f'Found duplicate and deleted: {fi.name}')
            os.unlink(str(fi))

    idx = 0  # since you do not know how many lines in total, use while
    count = 0  # count how many lines in total
    found_cities = []
    while True:
        # NB. when skipsrows starts with `1`, it keeps the header!
        d = pd.read_csv(f, skiprows=range(1, idx * chunk + 1), header=[0], nrows=chunk, delimiter=',')
        if d.empty:  # no more data
            break

        idx += 1  # the chunk index iteratively grows
        count += len(d)
        ci = iterate_write(tempfolder, d, city_list=cities, encoding=encoding)
        found_cities += ci

    msg = f"Expected {len(cities)} cities and found {len(set(found_cities))} cities!"
    logging.info(f'Completed reading {count} lines of data. {msg}')


def processing(event: dict,
               ord_goods: list,
               ada_goods: list,
               tempfolder: Path,
               parallel_span: int = 3) -> None:
    """
    Data processing procedures where event-based data being labelled and goods being split.
    Note that, the structure of data should be like:

    :param event: an event json indexed by events
    :param ord_goods: a list of ordinary goods
    :param ada_goods: a list of adaptative goods
    :param tempfolder: inherited tempfolder
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

        logging.info(f'Render event: {e}')
        # NB. DO NOT change the format of dates
        fmt = '%Y%m%d' if param['freq'] == 'days' else '%Y%m'
        span = relativedelta(**{param['freq']: param['span']})
        cities = param['treated'] + param['untreated']
        # create dataframes for each event
        r = pd.DataFrame()
        for c in cities:
            file = tempfolder / f'{c}.txt'
            assert os.path.exists(str(file)), f'CityNotFound: at {str(file)}'

            # read data from textfiles
            d = pd.read_csv(str(file), delimiter=',', header=[0])
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
            d['adapt'] = 1  # NB. masked dataframe does not contain any other goods
            d.loc[ord_mask, 'adapt'] = 0
            d['treat'] = 1 if c in param['treated'] else 0
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
        savefile = tempfolder / f'{e}.txt'
        r.to_csv(str(savefile), sep=',', index=False, encoding='utf8')
        logging.info(f'Finish rendering event: {e}!')


def trend(event: dict,
          tempfolder: Path,
          checking_keys: list,
          save: Path = None) -> pd.DataFrame:
    """
    Summarise trending results of either adaptation/ordinary goods

    :param event: an event json indexed by events
    :param tempfolder: inherited tempfolder
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
        f = tempfolder / f'{e}.txt'
        df = pd.read_csv(str(f), header=[0], delimiter=',')
        df = df.sort_values(key_time)
        df[[key_time]] = df[[key_time]].astype(str)
        o = pd.DataFrame()
        for idx, c in enumerate(checking_keys):
            # summarise data
            uni = df[[key_time]].drop_duplicates().reset_index(drop=True)
            # check adaptation: 0/1 first
            for j in ['adapt', 'treat']:
                # ... merging with true samples
                s = df.loc[df[j] == 1, [key_time, c]].groupby(key_time).mean().reset_index()
                s.columns = [key_time] + [f'{c}|{j}=1']
                uni = uni.merge(s, on=key_time, how='left')
                # ... merging with false samples
                s = df.loc[df[j] == 0, [key_time, c]].groupby(key_time).mean().reset_index()
                s.columns = [key_time] + [f'{c}|{j}=0']
                uni = uni.merge(s, on=key_time, how='left')

            # check treatment + adaptation: 0/1 first (4 groups)
            for i in range(0, 2):  # 0, 1
                # merging with the untreated
                mask = df['treat'] == i
                s = df.loc[(df['adapt'] == 1) & mask, [key_time, c]].groupby(key_time).mean().reset_index()
                s.columns = [key_time] + [f'{c}|adapt=1|treat={i}']
                uni = uni.merge(s, on=key_time, how='left')
                s = df.loc[(df['adapt'] == 0) & mask, [key_time, c]].groupby(key_time).mean().reset_index()
                s.columns = [key_time] + [f'{c}|adapt=0|treat={i}']
                uni = uni.merge(s, on=key_time, how='left')

            # complete merging and fill NaNs with zero
            uni = uni.fillna(0)
            # load event params
            evt = event[e]
            ticks = uni[key_time].tolist()
            # find out the max/min values
            # use a set of colors
            cvalues = list(colors.values()) * 10  # in case we have many checking keys
            fig = plt.figure(figsize=(12, 12))
            ax1 = plt.subplot(2, 1, 1)
            ax2 = plt.subplot(2, 1, 2)
            # create two parallel plots
            x_range = range(len(uni))
            start, end = evt['period']

            # draw overall adaptation: 0/1 lineplots
            for i, col in enumerate(uni.columns.values[1:5]):
                ax1.plot(x_range, uni[col],
                         color=cvalues[i],
                         marker='o',
                         linestyle='dashed',
                         markersize=6,
                         label=col)
            # find the min/max values
            arr = uni.iloc[:, 1:5].to_numpy()
            ax1.vlines(x=ticks.index(start), ymin=0, ymax=arr.max(), colors='grey', linestyles='--')
            ax1.vlines(x=ticks.index(end), ymin=0, ymax=arr.max(), colors='grey', linestyles='--')
            ax1.fill_betweenx(y=(0, arr.max()),
                              x1=ticks.index(start),
                              x2=ticks.index(end),
                              facecolor='grey', alpha=0.15)
            # change ticks
            ax1.set_xticks(range(0, len(uni)), labels=[])
            ax1.legend(fontsize=9)
            plt.tight_layout()
            plt.margins(0.01)
            # draw subgroup-based adaptation: 0/1 lineplots
            # NB. add pre-shock ATT to the adaptation=0&treatment=0 group
            pre_mask = uni[key_time] < start
            att = (uni.loc[pre_mask, f'{c}|adapt=0|treat=0'].mean() -
                   uni.loc[pre_mask, f'{c}|adapt=0|treat=1'].mean())
            post_mask = uni[key_time] > end
            uni[f'{c}|adapt=0|treat=0|att+'] = uni[f'{c}|adapt=0|treat=0']
            uni.loc[post_mask, f'{c}|adapt=0|treat=0|att+'] = uni.loc[post_mask, f'{c}|adapt=0|treat=0|att+'] + att

            for i, col in enumerate(uni.columns.values[5:]):
                x_range = range(len(uni))
                ax2.plot(x_range, uni[col],
                         color=cvalues[i],
                         marker='x',
                         markersize=6,
                         label=col)

            # draw vert lines for the shock
            arr = uni.iloc[:, 5:].to_numpy()
            ax2.vlines(x=ticks.index(start), ymin=0, ymax=arr.max(), colors='grey', linestyles='--')
            ax2.vlines(x=ticks.index(end), ymin=0, ymax=arr.max(), colors='grey', linestyles='--')
            ax2.fill_betweenx(y=(0, arr.max()),
                              x1=ticks.index(start),
                              x2=ticks.index(end),
                              facecolor='grey', alpha=0.15)
            # change ticks
            ax2.set_xticks(range(0, len(uni)), labels=uni[key_time].astype(str), rotation=45)
            ax2.legend(fontsize=9)
            # general specs
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
    output.to_csv(save / f'trend-{get_timestamp()}.csv', encoding='utf8')
    return output


def did(event: dict,
        tempfolder: Path,
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

        f = tempfolder / f'{e}.txt'
        data = pd.read_csv(str(f), delimiter=',', header=[0])
        # doing logarithm conversion before modelling
        data['lngmv'] = data['gmv'].apply(lambda x: np.log(x + 1))
        data['treatxpost'] = data['treat'] * data['post']
        # 1. baseline model spec: pooled OLS
        m1 = smf.ols('lngmv ~ treatxpost', data=data).fit()
        f = save / f'pooled-did-{e}-{get_timestamp()}.txt'
        f.write_text(m1.summary().as_text(), encoding='utf8')

        # NB. the default test covers the first and last 3 periods of shock and,
        #     to modify parallel params, specify `parallel_span` in `processing()`.
        # 2. parallel test using pooled OLS
        back, ahead = [], []
        for i in range(1, parallel_span + 1):
            data[f'treatxpre{i}'] = data[f'pre{i}'] * data['treat']
            data[f'treatxpost{i}'] = data[f'post{i}'] * data['treat']
            back += [f'treatxpre{i}']
            ahead += [f'treatxpost{i}']

        back.reverse()  # from -1, -2, ... to ... -2, -1!
        syntax = '+'.join(back) + ' + treatxpost + ' + '+'.join(ahead)
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
                       syntax.replace('treatx', '').replace(' ', '').split('+'),
                       rotation=45)
            fig.savefig(str(save / f'validate-did-{get_timestamp()}.png'), dpi=200, format='png')


def main(datafile,
         eventfile,
         ord: list,
         ada: list,
         save,
         checklist: list,
         chunk: int = 10e5,
         redo: bool = False,
         parallel_span: int = 3,
         if_plot: bool = True):

    event = load_config(eventfile)
    tempfolder = eventfile.parent / 'temp'  # ../temp/ should be in the same folder with datafile

    if redo and os.path.exists(str(tempfolder)):  # DO NOT replicate data-reading again if redo=False
        shutil.rmtree(str(tempfolder))  # delete them all every time we run it over again

    os.mkdir(tempfolder)  # create a folder if it's not existed
    # write city-specific data into files
    read_data(datafile, event, tempfolder=tempfolder, chunk=chunk)
    # processing data and save them in tempfolder
    processing(event, ord, ada, tempfolder)
    # output results
    trend(event, tempfolder, checking_keys=checklist, save=save)
    did(event, tempfolder, if_plot=if_plot, parallel_span=parallel_span, save=save)


if __name__ == '__main__':
    from pathlib import PureWindowsPath

    # NB. The following codes are valid only for local tests.
    # test 1: load data by lines and split them into files
    # event = load_config(Path('event.json'))
    # temppath = Path('temp')
    # if os.path.exists(str(temppath)):  # DO NOT replicate data-reading again if redo=False
    #     shutil.rmtree(str(temppath))  # delete them all every time we run it over again
    #
    # os.mkdir(temppath)
    # testfile = Path('data') / 'demo_data_test.txt'
    # read_data(testfile, event, tempfolder=temppath, chunk=100)
    #
    # # test 2: processing
    # # ['生活用品及服务', '食品烟酒', '医疗保健', '居住', '其他用品和服务',
    # #  '交通和通信', '衣着', '教育文化和娱乐']
    # ord = ['生活用品及服务', '食品烟酒', '衣着']
    # ada = ['医疗保健', '居住']
    # processing(event, ord, ada, tempfolder=temppath)
    #
    # # test 3: trending
    # save = Path('results')
    # save.mkdir(parents=True, exist_ok=True)
    # o = trend(event, temppath, checking_keys=['amount', 'ord_cnt'], save=save)
    #
    # # test 4: diff-in-diff with Pooled OLS
    # checklist = ['amount']
    # did(event,
    #     temppath,
    #     if_plot=True,
    #     parallel_span=3,
    #     save=save)

    # production operation
    # configuration: city_level.txt, 20110101-20191231
    root = PureWindowsPath('C:/Users/wb-ljw894653.DIPPER/PycharmProjects/UrbanConsumerAdaptation')
    efile = root / 'event-dev.json'
    save = root / 'results'
    if os.path.exists(str(save)):  # DO NOT replicate data-reading again if redo=False
        shutil.rmtree(str(save))  # delete them all every time we run it over again

    os.mkdir(str(save))  # refresh results/ folder
    datafile = PureWindowsPath('D:/air_pollution/input/city_level/city_category.txt')  # the target file
    ord = ['生活用品及服务', '食品烟酒', '衣着']
    ada = ['医疗保健', '居住']
    checklist = ['amount', 'ord_cnt']
    main(datafile, efile, ord, ada, save, checklist=checklist, chunk=100000, redo=True)
