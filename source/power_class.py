###
### Submodule power_class
###

print('submodule "power_class" imported')

# python modules
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import source.aux as aux
import seaborn as sns
import datetime as dt
import pdb
import requests
import io
import json

# fanfare modules
import source.aux as aux

# path to save plots
d_plot = '../plots/'

class PowerData():
    ''' This class defines an object that contains the hourly power system timeseries data to be used
    '''

    def __init__(self,**kwargs):

        # handle default values and kwargs
        argkeys_needed      =   ['data_type','load','year','time_cut','d_data','file_name','line_limit']
        a                   =   aux.handle_args(kwargs,argkeys_needed)
        self.data_type      =   a.data_type
        self.file_path      =   a.file_path
        self.file_name      =   a.file_name

        if a.load:
            print('Loading previously stored dataset')
            if not self.file_name:
                print('File name not given - could not load data')
            else:
                data_df = pd.read_pickle(a.d_data+self.file_name)
        else:
            print('Reading in new dataset')
            print('Data type: %s' % self.data_type)

            if not self.data_type:
                print('Data type not given - could not load data')

            if self.data_type == 'energinet':

                url = 'https://api.energidataservice.dk/datastore_search?resource_id=electricitybalance&limit=%s' % a.line_limit
                print('Downloading Electricity Balance data from energidataservice.dk - this may take a while')
                urlData = requests.get(url).content.decode('utf-8') # this step may take a while
                data_dict = json.loads(urlData)['result']['records']
                data_df = pd.DataFrame(data_dict)[['PriceArea','HourDK','GrossCon','OffshoreWindPower','OnshoreWindPower','SolarPowerProd']]
                data_df['WindPowerProd'] = data_df['OffshoreWindPower'] + data_df['OnshoreWindPower']
                data_df = data_df.drop(['OffshoreWindPower','OnshoreWindPower'],axis=1)
                data_df = data_df.rename(columns={'HourDK':'datetime'})
                data_df_DK = data_df.groupby(['datetime']).sum()#.drop('PriceArea')
                data_df_DK_DK1 = data_df_DK.merge(data_df[data_df.PriceArea == 'DK1'].drop('PriceArea',axis=1),on='datetime',suffixes=('_DK','_DK1'))
                data_df_DK_DK2 = data_df_DK.merge(data_df[data_df.PriceArea == 'DK2'].drop('PriceArea',axis=1),on='datetime',suffixes=('_DK','_DK2'))
                data_df = data_df_DK_DK1.merge(data_df_DK_DK2.drop(['GrossCon_DK','WindPowerProd_DK','SolarPowerProd_DK'],axis=1),on='datetime')
                print(data_df.head())

                if self.file_name:
                    print('Saving new data for next time at')
                    print(a.d_data+self.file_name)
                    data_df.to_pickle(a.d_data+self.file_name)

        if a.year:
            time_cut = [np.datetime64('%s-01-01' % a.year),np.datetime64('%s-01-01' % (int(a.year)+1))]
        if a.time_cut:
            time_cut = a.time_cut 
        if a.time_cut | a.year:
            data_df         =   self.data_df.copy()
            mask            =   np.array([(data_df.datetime >= time_cut[0]) & (data_df.datetime < time_cut[1])])[0]
            data_df         =   data_df[mask].reset_index(drop=True)

        self.data_df = data_df

