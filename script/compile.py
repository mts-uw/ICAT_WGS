import numpy as np
import pandas as pd

USAGE_DESC = ['AN',  'AW', 'atomic radius', 'electronegativity',
              'm. p.', 'b. p.', 'delta_fus H', 'density',
              'ionization enegy', 'Surface energy']


def read_desc():
    desc = pd.read_csv('data/Descriptors_WGS.csv',
                       skiprows=[0], index_col="symbol")
    desc = desc.loc[:, ['AN',  'AW', 'atomic radius', 'electronegativity',
                        'm. p.', 'b. p.', 'delta_fus H ', 'density',
                        'ionization enegy ', 'Surface energy ']]
    desc.columns = USAGE_DESC
    desc = desc.fillna(desc.mean())
    return desc


def data_convert():
    data = pd.read_excel(
        'data/WGS.xlsx', skiprows=8).drop(['Total # of Data', 'Reference', 'Data'], axis=1)
    print('# of Original Datapoints:', len(data))
    drop_support = ['ZEO', 'HAP', 'ACC', 'YSZ']
    idx = (data.loc[:, drop_support] == 0).all(axis=1)
    data = data[idx].drop(drop_support, axis=1)
    data.index = np.arange(len(data))
    print('# of Data after preprocessing:', len(data))

    desc = read_desc()

    support = pd.read_excel('data/support.xlsx')
    element = list(desc.index)
    data = pd.concat([pd.DataFrame(columns=element), data]).fillna(0.0)

    support_wt = np.array(100 - data.loc[:, element].sum(axis=1)
                          ).reshape(-1, 1)*np.array(data.loc[:, support.support])
    support_wt = support_wt / np.array(support.ave_MW).T
    data.loc[:, element] = data.loc[:, element] / desc.AW
    data.loc[:, support.key] += support_wt
    data.loc[:, element] = data.loc[:, element] / \
        np.array(data.loc[:, element].sum(axis=1)).reshape(-1, 1) * 100
    data = data.drop(support.support, axis=1)

    swed_names = []
    for i in range(4):
        for s in list(desc.columns):
            swed_names.append(f"{s} ({i + 1})")

    swed = pd.DataFrame(comp_times_base(
        data.loc[:, element], desc.T, sort=True)).iloc[:, :40]
    swed.columns = swed_names

    data = pd.concat([data, swed], axis=1)
    data.to_csv('data/wgs.csv', index=None)
    return data, desc


def data_loader(convert=False, desc_names=USAGE_DESC):
    for s in desc_names:
        if s not in USAGE_DESC:
            print(f'{s} is not avaiable!!')
            print('Please use only in ', USAGE_DESC)
            return None

    if convert:
        data, desc = data_convert()
    else:
        data = pd.read_csv('data/wgs.csv')
        desc = read_desc()

    cols = get_columns(data, desc_names)

    return data, desc, cols


def comp_times_base(comp, base, sort=False, times=True, attention=False):
    count = 0
    for key, rows in comp.iterrows():
        stack = np.vstack((rows, base))
        if times == True:
            time = np.array(base) * np.array(rows)
            stack = np.vstack((rows, time))

        if sort == True:
            stack = pd.DataFrame(stack).sort_values(
                [0], ascending=False, axis=1)

        stack = pd.DataFrame(stack).iloc[1:, :]
        stack = np.array(stack)

        if count == 0:
            if attention:
                res = np.sum(stack, axis=1)
            else:
                res = np.array(stack.T.flatten())

            count += 1
        else:
            if attention:
                res = np.vstack((res, np.sum(stack, axis=1)))
            else:
                res = np.vstack((res, np.array(stack.T.flatten())))

            count += 1
    return res


def get_columns(data, use_cols):
    element = list(data.loc[:, 'Li':'Th'].columns)
    preparation = list(data.loc[:, 'IWI': 'DP'].columns)
    condition = list(
        data.loc[:, 'Calcination Temperture (℃)':'F/W (mg.min/ml)'].columns)

    swed_names = []
    for i in range(4):
        for s in list(use_cols):
            swed_names.append(f"{s} ({i + 1})")
    cols = {}
    cols['element'] = element
    cols['preparation'] = preparation
    cols['condition'] = condition
    cols['use_cols'] = use_cols
    cols['swed'] = swed_names
    cols['conv'] = element + preparation + condition
    cols['prop1'] = element + preparation + condition + swed_names
    cols['prop2'] = preparation + condition + swed_names
    cols['target'] = 'CO Conversion'
    return cols
