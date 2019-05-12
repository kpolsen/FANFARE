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
import dateutil.parser
from iteration_utilities import duplicates



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
                # Check that delta_time and time_steps are in dataframe:
                if 'delta_time' not in data_df.keys():
                    data_df = data_df.sort_values('datetime').reset_index(drop=True)
                    datetimes = data_df['datetime']
                    delta_times =  datetimes - np.min(datetimes)
                    delta_times = [delta_time.total_seconds() for delta_time in delta_times]
                    data_df['delta_time'] = delta_times
                if 'time_steps' not in data_df.keys():
                    delta_times = data_df['delta_time']
                    time_steps = np.array([delta_times[i+1]-delta_times[i] for i in range(len(data_df)-1)])
                    time_steps = np.append(time_steps[0],time_steps)
                    data_df['time_steps'] = time_steps #s
                if 'ResLoad' not in data_df.keys():
                    try: 
                        data_df['ResLoad'] = data_df['GrossCon'] - (data_df['WindPowerProd'] + data_df['SolarPowerProd'])
                    except:
                        print('Could not estimate residual load, one of the following columns were missing: \nGrossCon, WindPowerProd, SolarPowerProd')
                        print('(if data_type = "energinet", this is fine, ResLoad_DK, ResLoad_DK1, ResLoad_DK2 should be there)')
                self.data_df = data_df

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
                data_df = data_df.rename(columns={'HourDK':'datetime'}).sort_values(by=['datetime']).reset_index(drop=True)
                data_df_DK = data_df.groupby(['datetime']).sum()#.drop('PriceArea')
                data_df_DK_DK1 = data_df_DK.merge(data_df[data_df.PriceArea == 'DK1'].drop('PriceArea',axis=1),on='datetime',suffixes=('_DK','_DK1'),how='outer')
                data_df_DK_DK2 = data_df_DK.merge(data_df[data_df.PriceArea == 'DK2'].drop('PriceArea',axis=1),on='datetime',suffixes=('_DK','_DK2'),how='outer')
                data_df = data_df_DK_DK1.merge(data_df_DK_DK2.drop(['GrossCon_DK','WindPowerProd_DK','SolarPowerProd_DK'],axis=1),on='datetime',how='outer').fillna(0)
                data_df['ResLoad_DK'] = data_df['GrossCon_DK'] - (data_df['WindPowerProd_DK'] + data_df['SolarPowerProd_DK'])
                data_df['ResLoad_DK1'] = data_df['GrossCon_DK1'] - (data_df['WindPowerProd_DK1'] + data_df['SolarPowerProd_DK1'])
                data_df['ResLoad_DK2'] = data_df['GrossCon_DK2'] - (data_df['WindPowerProd_DK2'] + data_df['SolarPowerProd_DK2'])
                self.data_df = data_df

                # Remove any duplicate time steps:
                self._RemoveDuplicateTimes(col_name='WindPowerProd_DK',verbose=a.verbose)

                # Add columns with time steps and delta time steps to dataframe:
                data_df = self.data_df.copy()
                timestamps = [data_df['datetime'][_].replace('T',' ') for _ in range(len(data_df))]
                data_df['datetime'] = [dateutil.parser.parse(timestamps[i]) for i in range(len(data_df))]
                data_df = data_df.sort_values('datetime').reset_index(drop=True)
                datetimes = data_df['datetime']
                delta_times =  datetimes - np.min(datetimes)
                delta_times = [delta_time.total_seconds() for delta_time in delta_times]
                data_df['delta_time'] = delta_times
                time_steps = np.array([delta_times[i+1]-delta_times[i] for i in range(len(data_df)-1)])
                time_steps = np.append(time_steps[0],time_steps)
                data_df['time_steps'] = time_steps #s

                if self.file_name:
                    print('Saving new data for next time at')
                    print(a.d_data+self.file_name)
                    data_df.to_pickle(a.d_data+self.file_name)

        # Optionally apply a cut in time
        if type(a.year) == str:
            time_cut = [np.datetime64('%s-01-01' % a.year),np.datetime64('%s-01-01' % (int(a.year)+1))]
        if a.time_cut:
            time_cut = a.time_cut 
        if a.time_cut | (type(a.year) == str):
            data_df         =   self.data_df.copy()
            mask            =   np.array([(data_df.datetime >= time_cut[0]) & (data_df.datetime < time_cut[1])])[0]
            data_df         =   data_df[mask].reset_index(drop=True)

        self.data_df = data_df

    def _RemoveDuplicateTimes(self,**kwargs):

        # create a custom namespace for this method
        argkeys_needed      =   ['col_name','verbose']
        a                   =   aux.handle_args(kwargs,argkeys_needed,verbose=False)

        data_df             =   self.data_df.copy().reset_index(drop=True)    

        duplicate_times     =   np.unique(np.array(list(duplicates(data_df['datetime']))))

        indices_to_remove   =   []
        if len(duplicate_times) == 0:
            if a.verbose: print('Found no duplicates!! :D')
        if len(duplicate_times) > 0:
            if a.verbose: print('Found duplicates at:')
            for duplicate_time in duplicate_times:
                print(duplicate_time)
                indices_to_remove.extend(np.where(data_df['datetime'] == duplicate_time)[0].tolist()[1::])
            # Remove extra rows from dataframe
            data_df = data_df.drop(data_df.index[indices_to_remove],axis=0).reset_index(drop=True)     
        print('Removed duplicates')

        self.data_df = data_df

    def info(self,verbose=True):
        ''' Prints basic info about this dataset.
        '''

        data_df = self.data_df
        if verbose:
            print('\n--------')
            print('Data object contains:')
            print('%s data points' % len(data_df))
            print('from %s to %s' % (np.min(data_df.datetime),np.max(data_df.datetime)))
            print('Minimum time step: %s sec' % (np.min(data_df.time_steps.values)))
            print('Maximum time step: %s sec' % (np.max(data_df.time_steps.values)))
            try:
                print('For Bornholm:')
                BO_cons = data_df['GrossCon_BO'].values
                datetime = data_df['datetime']
                print(np.min(data_df.datetime[BO_cons > 0]),np.max(data_df.datetime[BO_cons > 0]))
            except:
                print('Could not find Bornholm data')
            print('Most common time step: %s sec' % (np.median(data_df.time_steps.values)))
            print('--------')

        return(np.min(data_df.datetime),np.max(data_df.datetime))
