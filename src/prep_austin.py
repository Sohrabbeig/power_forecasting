import numpy as np
import pandas as pd
import pickle

def austin_hourly(dataroot, force_reload=False):
    try:
        with open(dataroot+'/prep_austin_hourly.pk', 'rb') as f:
            df_merged = pickle.load(f)
    except:
        force_reload = True

    if force_reload:
        df = pd.read_csv(dataroot+'/15minute_data_austin.csv', parse_dates=[1])
        df['month'] = df.apply(lambda x: x['local_15min'].month,axis=1)
        df['day'] = df.apply(lambda x: x['local_15min'].day, axis=1)
        df['weekday'] = df.apply(lambda x: x['local_15min'].weekday(),axis=1)
        df['hour'] = df.apply(lambda x: x['local_15min'].hour,axis=1)
        df = df.drop('local_15min', axis=1)

        keep = ['dataid', 'leg1v', 'leg2v', 'hour', 'weekday', 'day', 'month', 'solar', 'grid']
        items = list(df.columns)
        for i in keep:
            items.remove(i)

        df['target'] = df.apply(lambda x: x[items].sum(), axis=1)
        items.append('solar')
        items.append('grid')
        df = df.drop(items, axis=1)
        df = df.groupby(by=['dataid',  'month', 'day', 'weekday', 'hour' ]).\
            agg(target=('target', 'sum'), leg1v=('leg1v', 'mean'), leg2v=('leg2v', 'mean')).reset_index()
        
        for column in ["leg1v", 'leg2v', 'target']:
            mean = df[column].mean()
            std = df[column].std()
            df.loc[(abs(df[column]-mean) >= 3*std), column] = np.float64('nan')
            df.loc[df['target']<0] = np.float64('nan')

        df.interpolate(method='linear', limit_direction='both', inplace=True)


        df2  = pd.read_csv('../data/weather/Austin_weather.csv', parse_dates=[1])
        df2['month'] = df2.apply(lambda x: x['local_time'].month,axis=1)
        df2['day'] = df2.apply(lambda x: x['local_time'].day, axis=1)
        df2['weekday'] = df2.apply(lambda x: x['local_time'].weekday(),axis=1)
        df2['hour'] = df2.apply(lambda x: x['local_time'].hour,axis=1)
        df2 = df2.drop(['time', 'local_time'], axis=1)

        df_merged = pd.merge(df, df2, on=['month', 'day', 'weekday', 'hour'])
        df_merged = df_merged.sort_values(by=['dataid', 'month', 'day', 'weekday', 'hour']).reset_index(drop=True)
        df_merged = df_merged[['dataid', 'month', 'day', 'weekday', 'hour', 'leg1v', 'leg2v', \
            'temperature', 'precipitation', 'target']]

        with open(dataroot+'/prep_austin_hourly.pk', 'wb') as f:
            pickle.dump(df_merged, f)
    
    return df_merged