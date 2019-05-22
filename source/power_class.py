###
### Submodule power_class
###

print('submodule "power_class" imported')

# python modules
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
d_plot = 'plots/'

class PowerData():
    ''' This class defines an object that contains the hourly power system timeseries data to be used

    Parameters
    ----------
    data_type : str/bool
        type of data, 'energinet' for loading Energinet data, default : False

    load : bool
        if True, load data previously stored as pandas dataframe, if False, download data from url site, default : False

    year : str/bool
        year to cut out of data, default: False

    time_cut : list/bool
        start and endtime in np.datetime64-readable format, default: False

    d_data : str
        path to where data will be stored, default: 'data/'

    file_name : str/bool
        file name used to load/save dataset as pandas dataframe, default: False

    line_limit : int
        max number of lines to read from url site, default: 5

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
                        data_df['RenPower'] = data_df['WindPowerProd'] + data_df['SolarPowerProd']
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

                # url = 'https://api.energidataservice.dk/datastore_search?resource_id=electricitybalance&limit=%s' % a.line_limit
                url = 'https://api.energidataservice.dk/datastore_search_sql?sql=SELECT * from "electricitybalance" order by "HourUTC" desc LIMIT %s' % a.line_limit
                print('Downloading Electricity Balance data from energidataservice.dk - this may take a while')
                urlData = requests.get(url).content.decode('utf-8') # this step may take a while
                data_dict = json.loads(urlData)['result']['records']
                data_df = pd.DataFrame(data_dict)[['PriceArea','HourDK','GrossCon','OffshoreWindPower','OnshoreWindPower','SolarPowerProd']]
                data_df.fillna(value=0, inplace=True) # Replace None and NaNs with 0
                data_df['WindPowerProd'] = data_df['OffshoreWindPower'] + data_df['OnshoreWindPower']
                data_df = data_df.drop(['OffshoreWindPower','OnshoreWindPower'],axis=1)
                data_df = data_df.rename(columns={'HourDK':'datetime'}).sort_values(by=['datetime']).reset_index(drop=True)
                data_df_DK = data_df.groupby(['datetime']).sum()
                data_df_DK_DK1 = data_df_DK.merge(data_df[data_df.PriceArea == 'DK1'].drop('PriceArea',axis=1),on='datetime',suffixes=('_DK','_DK1'),how='outer')
                data_df_DK_DK2 = data_df_DK.merge(data_df[data_df.PriceArea == 'DK2'].drop('PriceArea',axis=1),on='datetime',suffixes=('_DK','_DK2'),how='outer')
                data_df = data_df_DK_DK1.merge(data_df_DK_DK2.drop(['GrossCon_DK','WindPowerProd_DK','SolarPowerProd_DK'],axis=1),on='datetime',how='outer').fillna(0)
                data_df['GrossCon_DK'] = data_df['GrossCon_DK1'] + data_df['GrossCon_DK2']
                data_df['RenPower_DK'] = data_df['WindPowerProd_DK'] + data_df['SolarPowerProd_DK']
                data_df['RenPower_DK1'] = data_df['WindPowerProd_DK1'] + data_df['SolarPowerProd_DK1']
                data_df['RenPower_DK2'] = data_df['WindPowerProd_DK2'] + data_df['SolarPowerProd_DK2']
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
                self.data_df = data_df

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
        ''' Method to remove duplicate timestamps in dataset

        Parameters
        ----------
        verbose : bool
            if True, print whether duplicates were found, default: False

        '''


        # create a custom namespace for this method
        argkeys_needed      =   ['verbose']
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
        if a.verbose: print('Removed duplicates')

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

    # DATA ANALYSIS -------------------------------------------------------------------------------------------

    def AddHighPenetrationIndex(self,**kwargs):
        ''' Add index to separate high wind penetration from low

        Parameters
        ----------
        alpha_cuts : list/bool
            min and max VRE hourly share, used to print diagnostics of data, default: False

        time_cut : list/bool
            start and endtime in np.datetime64-readable format, default: False

        power_name : str/bool
            name of the column containing power data to use, default: False

        load_name : str/bool
            name of the column containing gross electricity consumption data to use, default: False

        overwrite : bool
            if True, overwrite any previous calculations of VRE hourly share, default: False
    
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['alpha_cuts','power_name','load_name','time_cut','overwrite']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        data_df             =   self.data_df.copy()

        if a.load_name: load_name = a.load_name
        print('Using load: '+load_name)

        # calculate wind penetration fraction
        if ('alpha_' + a.power_name not in data_df.keys()) | a.overwrite:
            self.AddPenetrationFraction(power_names=[a.power_name],load_names=[load_name],time_cut=a.time_cut)
            data_df = self.data_df.copy()

        # High wind penetration
        alpha = data_df['alpha_' + a.power_name].values
        high_penetration_index = np.zeros(len(data_df))
        high_penetration_index[alpha > a.alpha_cuts[1]] = 1
        high_penetration_index[alpha == -1] = 0

        data_df['high_penetration_index_' + a.power_name] = high_penetration_index
        self.data_df = data_df

        percent = len(high_penetration_index[high_penetration_index == 1])/len(high_penetration_index[high_penetration_index >= 0])*100.
        print('Penetration fraction is above %s%% %.2f%% of the time' % (a.alpha_cuts[1],percent))
        print('With %s data points' % (len(high_penetration_index[high_penetration_index == 1])))
        print('Out of %s data points' % (len(high_penetration_index)))

        # Change in index: 0 - no change, +1 - going into high alpha period, -1 - going out of high alpha period
        change_in_index = np.array([high_penetration_index[_+1]-high_penetration_index[_] for _ in range(len(data_df)-1)])
        change_in_index = np.append(change_in_index,0)
        while data_df.datetime.values[change_in_index == -1][0] < data_df.datetime.values[change_in_index == 1][0]:
            change_in_index[change_in_index == -1] = np.append(0,change_in_index[change_in_index == -1][1::])
        end_times,start_times = data_df.datetime.values[change_in_index == -1], data_df.datetime.values[change_in_index == 1]
        print('Number of start times: %s' % (len(start_times)))
        print('Number of end times: %s' % (len(end_times)))
        print('First start time: %s' % (start_times[0]))
        print('First end time: %s' % (end_times[0]))
        print('Last start time: %s' % (start_times[-1]))
        print('Last end time: %s' % (end_times[-1]))

        while len(start_times) != len(end_times):
            if len(start_times) > len(end_times):
                start_times = start_times[:-1]#data_df.datetime.values[change_in_index == -1], data_df_cut.datetime.values[change_in_index_cut == 1]
            if len(end_times) > len(start_times):
                end_times = end_times[:-1]#data_df_cut.datetime.values[change_in_index_cut == -1], data_df.datetime.values[change_in_index == 1]
        try:
            durations = end_times - start_times
        except ValueError:
            print('could not calculate durations')
        durations = durations/np.timedelta64(1,'h')
        print('Min and maximum epoch durations of high penetration: %f and %f hrs' % (min(durations),max(durations)))
        setattr(self,'high_penetration_duration_hrs',durations)
        setattr(self,'high_penetration_start_times',start_times)
        setattr(self,'high_penetration_end_times',end_times)
        # Add start time as hour of day
        start_times_df = pd.DataFrame({'start_times':start_times})
        start_times_hour = np.array([start_times_df['start_times'][_].hour for _ in range(len(start_times))])
        setattr(self,'high_penetration_start_times_hour',start_times_hour)

        # Low wind penetration
        alpha = data_df['alpha_' + a.power_name].values
        low_penetration_index = np.zeros(len(data_df)) 
        low_penetration_index[alpha < a.alpha_cuts[0]] = 1
        low_penetration_index[alpha == -1] = 0

        data_df['low_penetration_index_' + a.power_name] = low_penetration_index
        self.data_df = data_df

        percent = len(low_penetration_index[low_penetration_index == 1])/len(low_penetration_index[low_penetration_index >= 0])*100.
        print('Penetration fraction is below %s%% %.2f%% of the time' % (a.alpha_cuts[0],percent))
        print('With %s data points' % (len(low_penetration_index[low_penetration_index == 1])))
        print('Out of %s data points' % (len(low_penetration_index)))

        # Change in index: 0 - no change, +1 - going into high alpha period, -1 - going out of high alpha period
        change_in_index = np.array([high_penetration_index[_+1]-high_penetration_index[_] for _ in range(len(data_df)-1)])
        change_in_index = np.append(change_in_index,0)
        while data_df.datetime.values[change_in_index == -1][0] < data_df.datetime.values[change_in_index == 1][0]:
            change_in_index[change_in_index == -1] = np.append(0,change_in_index[change_in_index == -1][1::])
        end_times,start_times = data_df.datetime.values[change_in_index == -1], data_df.datetime.values[change_in_index == 1]
        print('Number of start times: %s' % (len(start_times)))
        print('Number of end times: %s' % (len(end_times)))
        print('First start time: %s' % (start_times[0]))
        print('First end time: %s' % (end_times[0]))
        print('Last start time: %s' % (start_times[-1]))
        print('Last end time: %s' % (end_times[-1]))

        change_in_index = np.array([low_penetration_index[_+1]-low_penetration_index[_] for _ in range(len(data_df)-1)])
        change_in_index = np.append(change_in_index,0)
        change_in_index[low_penetration_index == -1] = 0
        while data_df.datetime.values[change_in_index == -1][0] < data_df.datetime.values[change_in_index == 1][0]:
            change_in_index[change_in_index == -1] = np.append(0,change_in_index[change_in_index == -1][1::])
            # print(data_df.datetime.values[change_in_index == -1][0],data_df.datetime.values[change_in_index == 1][0])
        # durations = data_df.datetime.values[change_in_index == -1] - data_df.datetime.values[change_in_index == 1][:-1]
        end_times,start_times = data_df.datetime.values[change_in_index == -1], data_df.datetime.values[change_in_index == 1]
        while len(start_times) != len(end_times):
            if len(start_times) > len(end_times):
                start_times = start_times[:-1]#data_df.datetime.values[change_in_index == -1], data_df_cut.datetime.values[change_in_index_cut == 1]
            if len(end_times) > len(start_times):
                end_times = end_times[:-1]#data_df_cut.datetime.values[change_in_index_cut == -1], data_df.datetime.values[change_in_index == 1]
        try:
            durations = end_times - start_times
        except ValueError:
            print('could not calculate durations')
        durations = durations/np.timedelta64(1,'h')
        print('Min and maximum epoch durations of low penetration: %f and %f hrs' % (min(durations),max(durations)))
        setattr(self,'low_penetration_duration_hrs',durations)
        setattr(self,'low_penetration_start_times',start_times)
        setattr(self,'low_penetration_end_times',end_times)
        # Add start time as hour of day
        start_times_df = pd.DataFrame({'start_times':start_times})
        start_times_hour = np.array([start_times_df['start_times'][_].hour for _ in range(len(start_times))])
        setattr(self,'low_penetration_start_times_hour',start_times_hour)
        print('------')

    def AddPenetrationFraction(self,**kwargs):
        ''' Calculates wind penetration in percent of total consumption, and adds result in dataframe.
        
        Parameters
        ----------
        time_cut : list/bool
            start and endtime in np.datetime64-readable format, default: False

        power_names : list
            names of the columns containing power data to use, default: []

        load_names : list
            names of the columns containing gross electricity consumption data to use, default: []

        yearly : bool
            if True, calculate annual VRE penetration instead of VRE hourly share, default: False

        time_cut : list/bool
            start and endtime in np.datetime64-readable format, default: False

        verbose : bool
            if True, print some status updates, default: False

        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['power_names','load_names','yearly','time_cut','verbose']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        data_df = self.data_df.copy()
        if a.time_cut:
            mask = np.array([(data_df.datetime > a.time_cut[0]) & (data_df.datetime < a.time_cut[1])])[0]
            data_df             =   data_df.copy()[mask].reset_index(drop=True)
        datetimes           =   data_df['datetime']
        delta_time          =   data_df['delta_time'].values/60/60

        if a.yearly:
            if a.verbose: print('Calculating penetration fraction for every year')
            years = np.unique(np.array([_.year for _ in datetimes]))
            time_periods = [0]*len(years)
            for i,year in enumerate(years):
                time_periods[i] = [dt.datetime(year, 1, 1),dt.datetime(year+1, 1, 1)]
            alpha = pd.DataFrame({'years':years,'Total':np.zeros(len(years))})
            for power_name,load_name in zip(a.power_names,a.load_names):
                power = data_df[power_name]
                load = data_df[load_name]
                alpha_temp = np.zeros(len(time_periods))
                for i,time_period in enumerate(time_periods):
                    mask = np.array([(datetimes >= np.datetime64(time_period[0])) & (datetimes <= np.datetime64(time_period[1]))])[0]
                    power_cut = power[mask]
                    load_cut = load[mask]
                    delta_time_cut = delta_time[mask]
                    # if len(load_cut[load_cut > 0]) > 3000:
                    if np.sum(load_cut[power_cut > 0]) == 0: 
                        alpha_temp[i] = 0
                    else: 
                        power_int = np.trapz(power_cut[power_cut > 0],delta_time_cut[power_cut > 0])
                        load_int = np.trapz(load_cut[power_cut > 0],delta_time_cut[power_cut > 0])
                        alpha_temp[i] = np.sum(power_int)/np.sum(load_int)*100. # %
                    # else:
                    #     print('Not enough load data for this year:')
                    #     print(time_period)
                alpha[power_name] = alpha_temp
                alpha['Total']    += alpha_temp

            self.alpha = alpha
        else:
            if a.verbose: print('Calculating penetration fraction for every time step')
            for power_name,load_name in zip(a.power_names,a.load_names):
                alpha_temp = data_df[power_name].values/data_df[load_name].values*100. 
                # Don't use low quality data in Bornholm or data during island mode:
                if 'BO' in power_name: alpha_temp[data_df['flag'].values == -1000] = -1
                if 'BO' in power_name: alpha_temp[data_df['island_mode'].values == 1] = -1
                # Don't use timestamps with no load:
                alpha_temp[data_df[load_name].values <= 0] = -1
                # print(data_df['datetime'][abs(alpha_temp - 100) < 5])
                data_df['alpha_' + power_name] =  alpha_temp
            self.data_df = data_df
        if a.verbose: print('------')

        if a.yearly: return(alpha)

    def SetFrequencySetup(self,**kwargs):
        ''' Derive frequency intervals. this method will add frequency cut information, color-coding and labels as attributes to data object.

        Parameters
        ----------
        duration_cuts : list/bool
            list of duration cuts (each a 2-element list) in hours, default: False

        labels : list
            list of labels to describe the timescales, default: False

        labels_verbose : list
            list of more verbose labels to use in plots etc, default: False

        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['duration_cuts','labels','labels_verbose']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        freq_cuts = np.sort(1/(2.*np.array(a.duration_cuts))/(60*60)) # Hz
        if np.max(a.duration_cuts) >= 1e6:
            freq_cuts[freq_cuts == np.min(freq_cuts)] = 0 # set freq cut to 0 if going to extremely long timescales
        colors = cm.rainbow(np.linspace(0, 1, len(freq_cuts)))    

        self.labels = a.labels
        self.labels_verbose = a.labels_verbose
        self.colors = colors
        self.freq_cuts = freq_cuts
        self.N_freqs = len(freq_cuts)

    def GetFluctuations(self,**kwargs):
        ''' Calculates durations and integrated energy of fluctuations within given epoch, returns in dictionary

        Parameters
        ----------
        col_name : str/bool
            name of the column containing data to use, default: False

        season: str/bool
            can be 'summer' or 'winter', default: False

        time_cut : list/bool
            start and endtime in np.datetime64-readable format, default: False

        verbose : bool
            if True, print some status updates, default: False

        overwrite : bool
            if True, overwrite any previous calculations of VRE hourly share, default: False
    
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['col_name','verbose','season','time_cut','overwrite']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        data_df_or = self.data_df.copy()
        datetimes = data_df_or['datetime'].values

        # Select object to get penetration fraction from
        int_power_dict      =   {}
        ramps_dict          =   {}
        total_int_power     =   0
        for cut,freq_cut in enumerate(self.freq_cuts):
            if a.time_cut:
                mask = np.array([(data_df_or['datetime'] >= np.datetime64(a.time_cut[0])) & (data_df_or['datetime'] <= np.datetime64(a.time_cut[1]))])[0]
                data_df = self.data_df.copy()[mask].reset_index(drop=True)
                self.data_df = data_df
            else:
                data_df = self.data_df.copy()
            if (not a.col_name+'_iFFT_freq_cut_%s' % cut in data_df.keys()) | (a.overwrite):
                self.AddiFFT(freq_cut=freq_cut,freq_cut_number=cut,power_name=a.col_name,verbose=a.verbose,overwrite=a.overwrite)
            data_df             =   self.data_df.copy()
            data_df['months']   =   [data_df['datetime'][_].month for _ in range(len(data_df))]
            if a.season == 'winter':
                data_df             =   data_df[(data_df['months'] >= 10) | (data_df['months'] < 4)].reset_index(drop=True) # Oct - March
            if a.season == 'summer':
                data_df             =   data_df[(data_df['months'] >= 4) & (data_df['months'] < 10)].reset_index(drop=True) # April - Sept
            iFFT                =   data_df[a.col_name+'_iFFT_freq_cut_%s' % cut].values.real
            power               =   iFFT[data_df[a.col_name] != 0]
            time_step           =   self.data_df.time_steps[0]
            int_power           =   np.sum(np.abs(power))/(len(power)*time_step/(60*60)) # MWh, average annual value
            power_pos           =   power[power >= 0]
            power_neg           =   power[power <= 0]
            if a.verbose:
                print('Years of data with current selection: %.2f' % (len(power)*time_step/(60*60)/(365*24.)))
            ramps               =   np.array([iFFT[_+1] - iFFT[_] for _ in range(len(iFFT)-1)])
            total_int_power     +=   np.sum(int_power)
            int_power_dict['cut'+str(cut)]  =   int_power
            ramps_dict['cut'+str(cut)]      =   ramps

        if a.verbose:
            print('Relative amount of energy in each frequency interval:')
            for cut in range(self.N_freqs):
                print('For frequency interval: %.2e to %.2e Hz: %.2e MWh' % (self.freq_cuts[cut][0],self.freq_cuts[cut][1],np.sum(int_power_dict['cut'+str(cut)])))
                print('%.1f %% of integrated energy across all frequencies' % (np.sum(int_power_dict['cut'+str(cut)])/total_int_power*100.))

        if a.verbose:
            print('------')

        fluctuations        =   dict(int_power=int_power_dict,ramps=ramps_dict)

        int_power           =   np.array([np.sum(fluctuations['int_power']['cut'+str(cut)]) for cut in range(self.N_freqs)])

        if a.time_cut: self.data_df           =   data_df_or

        return(fluctuations,int_power)

    def AddFFT(self,**kwargs):
        ''' Calculates discrete (fast) Fourier transform (DFT) for wind power and adds as attribute to data object.

        Parameters
        ----------
        power_name : str/bool
            name of the column containing power data to use, default: False

        verbose : bool
            if True, print some status updates, default: False

        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['power_name','verbose']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        if a.verbose: print('Calculating FFT')
        data_df             =   self.data_df.copy()
        power               =   data_df[a.power_name].values
        mean_power          =   np.mean(power[power != -1])
        if a.verbose: print('Number of datapoints: %s' % (len(power)))

        FFT                 =   np.fft.fft(power-mean_power)
        dT                  =   data_df.delta_time[1]-data_df.delta_time[0] #s
        if a.verbose: print('Time step used for FFT: %s sec' % dT)
        freq                =   np.fft.fftfreq(data_df.delta_time.shape[-1])*1/dT
        # omega               =   freq/(2*np.pi)

        data_df[a.power_name+'_freq'] = freq
        data_df[a.power_name+'_FFT'] = FFT
        setattr(self,a.power_name+'_FFT',FFT)

        self.data_df        =   data_df

    def AddiFFT(self,**kwargs):
        ''' Calculates inverse Fast Fourier Transform (iFFT) for dataframe data_df 
        and inserts it back into dataframe data_df. 
            
        Parameters
        ----------
        power_name : str/bool
            name of the column containing power data to use, default: False

        freq_cut : list
            list of frequency cuts to use in Hz, default: [False]

        freq_cut_number : int/bool
            actual frequency cut number to use in self.freq_cuts: False

        verbose : bool
            if True, print some status updates, default: False

        overwrite : bool
            if True, overwrite any previous calculations of DFT, default: False
    
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['power_name','freq_cut','freq_cut_number','verbose','overwrite']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        data_df             =   self.data_df.copy()
        if a.power_name+'_iFFT' in self.data_df.keys() and a.overwrite == False: 
            if a.verbose: print('Already an iFFT calculated')
        else:
            if a.verbose: print('Calculating total iFFT')
            if (a.power_name+'_FFT' not in self.data_df.keys()) | (a.overwrite == True): 
                self.AddFFT(power_name=a.power_name,verbose=a.verbose)
            data_df             =   self.data_df.copy()
            freq                =   data_df[a.power_name+'_freq'].values
            FFT                 =   data_df[a.power_name+'_FFT'].values
            iFFT                =   np.fft.ifft(FFT)

        if len(a.freq_cut) > 1: 
            data_df             =   self.data_df.copy()
            a.freq_cut          =   np.sort(a.freq_cut)
            if a.verbose: print('Now doing frequency cut: %.2e to %.2e Hz' % (a.freq_cut[0],a.freq_cut[1]))
            freq                =   data_df[a.power_name+'_freq'].values
            FFT                 =   data_df[a.power_name+'_FFT'].values
            FFT_cut             =   np.copy(FFT)
            # print(min(abs(freq)),max(abs(freq)))
            FFT_cut[abs(freq) < a.freq_cut[0]] = 0
            FFT_cut[abs(freq) > a.freq_cut[1]] = 0

            if a.verbose: print('Frequency cut contains %.2f %% of FFT power (real part) - now calculating iFFT for this cut' % (np.sum(np.abs(FFT_cut.real))/np.sum(np.abs(FFT.real))*100.))
            iFFT            =   np.fft.ifft(FFT_cut)
            if a.verbose: print('done!')
            # Add iFFT to dataframe, and add mean to the frequency interval that includes 0 (for plotting)
            if type(a.freq_cut_number) == int: 
                data_df[a.power_name+'_iFFT_freq_cut_%s' % a.freq_cut_number] = iFFT.real
            else:
                data_df[a.power_name+'_iFFT'] = iFFT.real

            # Setting signal to 0 where there is no data
            iFFT[data_df[a.power_name] == -1] = 0
            data_df[a.power_name+'_iFFT'] = iFFT
        
        self.data_df            =   data_df
        if a.verbose: print('------')

    def GetPowerReqFromiFFT(self,**kwargs):
        ''' Derive power requirements from iDFTs of different frequency intervals, and return as matrix.

        Parameters
        ----------
        col_names : list
            names of the columns containing residual load data to use, default: []

        freq_cut_numbers : list
            list of frequency cut numbers to use in self.freq_cuts: []

        time_cut : list/bool
            start and endtime in np.datetime64-readable format, default: False

        verbose : bool
            if True, print some status updates, default: False
    
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['col_names','freq_cut_numbers','time_cut','verbose']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        if a.freq_cut_numbers == []: a.freq_cut_numbers = list(range(len(self.freq_cuts)))

        # Store original dataframe before making any cuts in time
        data_df_or = self.data_df.copy()
        if a.time_cut:
            mask = np.array([(self.data_df.datetime > a.time_cut[0]) & (self.data_df.datetime < a.time_cut[1])])[0]
            self.data_df = data_df_or.copy()[mask].reset_index(drop=True)

        std_power           =   np.zeros(len(self.freq_cuts))
        mean_power          =   np.zeros(len(self.freq_cuts))
        iFFT_matrix         =   np.zeros([len(a.col_names),len(a.freq_cut_numbers),len(self.data_df)])

        for i,col_name in enumerate(a.col_names):
            for freq_cut_number in a.freq_cut_numbers:
                # print('Frequency cut #%s' % freq_cut_number)
                if col_name+'_iFFT_freq_cut_%s' % freq_cut_number not in self.data_df.keys():
                    self.AddiFFT(freq_cut=self.freq_cuts[freq_cut_number],power_name=col_name,\
                        test=a.test,freq_cut_number=freq_cut_number,overwrite=True,verbose=a.verbose)
                data_df             =   self.data_df.copy()
                iFFT                =   data_df[col_name+'_iFFT_freq_cut_%s' % freq_cut_number].values.real
                std_power[freq_cut_number]          =   np.std(np.abs(iFFT))
                mean_power[freq_cut_number]         =   np.mean(np.abs(iFFT))
                iFFT_matrix[i,freq_cut_number,:]    =   iFFT
                if a.verbose: print('Freq cut number, min and max of iFFT: %s %s %s ' % (freq_cut_number,np.min(iFFT),np.max(iFFT)))
                if a.verbose: print('Length of iFFT: %s' % (len(iFFT)))

        iFFT_matrix = np.squeeze(iFFT_matrix)

        self.std_power      =   std_power
        self.mean_power     =   mean_power
        if a.time_cut: self.data_df        =   data_df_or
        return(iFFT_matrix)

    def GetCapacityReq(self,**kwargs):
        ''' Calculate storage capacity needs based on accumulated sum of energy

        Parameters
        ----------
        col_name : str
            name of the column containing residual load data to use, default: False

        time_cut : list/bool
            start and endtime in np.datetime64-readable format, default: False

        plot : bool
            if True, plot the capacity requirements as function of frequency interval, default: False

        legend : bool
            if True, add legend to plot, default: False

        verbose : bool
            if True, print some status updates, default: False
    
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['col_name','time_cut','plot','legend','verbose']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        data_df_or = self.data_df.copy()

        if a.plot:
            fig,ax1             =   plt.subplots(figsize=(13,6))

        if a.verbose: print('Capacity requirements:')

        if not self.labels:
            self.labels = [aux.pretty_label(freq_cut[0]) for freq_cut in self.freq_cuts]

        capacity            =   np.zeros(len(self.freq_cuts))
        for cut,freq_cut in enumerate(self.freq_cuts):
            if a.time_cut:
                mask = np.array([(data_df_or['datetime'] >= np.datetime64(a.time_cut[0])) & (data_df_or['datetime'] <= np.datetime64(a.time_cut[1]))])[0]
                self.data_df = data_df_or.copy()[mask].reset_index(drop=True)
            if (not a.col_name+'_iFFT_freq_cut_%s' % cut in self.data_df.keys()) | (a.overwrite):
                self.AddiFFT(freq_cut=freq_cut,freq_cut_number=cut,power_name=a.col_name,verbose=False,test=a.test,overwrite=a.overwrite)
            data_df             =   self.data_df.copy()
            power               =   data_df[a.col_name+'_iFFT_freq_cut_%s' % cut].values.real 
            if self.freq_cuts[cut][0] == 0:
                mean_power      =   np.mean(data_df[a.col_name])
                power           +=  mean_power
            SOC                 =   np.cumsum(power) # MWh
            capacity[cut]       =   np.max(SOC) - np.min(SOC)
            # SOC                 -=  np.max(SOC) - (np.max(SOC) - np.min(SOC))/2.
            if a.verbose: print('For frequency cut #%s: %.2e GWh' % (cut,capacity[cut]/1e3))
            if a.plot:
                ax1.plot(data_df.datetime,SOC/1e6,color=self.colors[cut],alpha=0.5,label=self.labels[cut])
                ax1.set_ylabel('SOC [10$^3$ GWh]')
                ax1.set_xlabel('Time')

        if a.plot & a.legend: ax1.legend(loc='best', fancybox=True,fontsize=14) # this creates a combined legend box for both axes

        if a.time_cut: self.data_df           =   data_df_or

        return(capacity)

    # PLOTTING -------------------------------------------------------------------------------------------

    def PlotHisto(self,**kwargs):
        ''' Plot histogram of an attribute, a column in the dataframe or a supplied data array.

        Parameters
        ----------
        histogram_values : list
            list of attributes or column names in data object to make a histogram of, default: []

        data : list/numpy array / bool
            option to supply data that is not in datao bject, default: False

        time_cut : list/bool
            start and endtime in np.datetime64-readable format, default: False

        bins : int
            number of bins in histogram, default: 200

        max_val : int/float
            max value allowed in data, default: 1000

        remove_zeros : bool
            if True, remove zeros from histogram values before calculating histogram, default: False

        log : bool
            if True, take logarithm of values before calculating histogram, default: False

        ylog : bool
            if True, convert y axis to logarithmic, default: False

        labels : list
            list of labels to describe the histograms, default: False

        colors : list
            list of colors to plot the histograms, default: ['b']

        ls : list
            list of linestyles to plot the histograms, default: ['-']

        alpha : int/float
            transparency index used to plot the histograms, default: 1

        xlim : list/bool
            limits in x axis of plot, default: False

        ylim : list/bool
            limits in y axis of plot, default: False

        xlab : str
            x axis title, default: ''

        ylab : str
            y axis title, default: 'ower [MW]'

        add : bool
            if True, add to existing axis object, default: False

        legend : bool
            if True, add legend to plot, default: False

        histo_counts : bool
            if True, plot counts on y axis otherwise percentages, default: False

        fig_name : str/bool
            if True, save plot figure with this filename, in d_plot (see top), default: False
    
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['histogram_values','data','time_cut','bins','max_val','remove_zeros','log','ylog','xlog',\
                                'labels','colors','ls','alpha','xlim','xlab','ylab','add','legend','histo_counts','fig_name']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        data_df = self.data_df.copy()

        if a.add:
            fig = plt.gcf()
            ax1 = plt.gca()
        else:
            fig = plt.figure(figsize=(10,6))
            ax1 = fig.add_subplot(1,1,1)
            if a.xlab:
                ax1.set_xlabel(a.xlab)
            else:
                ax1.set_xlabel('Values')
            ax1.set_ylabel('Percentage')
            if a.histo_counts:  ax1.set_ylabel('Counts')

        for _,histogram_value in enumerate(a.histogram_values):
            try:
                array = getattr(self,histogram_value)
            except:
                try:
                    array = data_df[histogram_value].values
                except:
                    print('Could not find values for %s, will look for supplied data' % histogram_value)
                    try:
                        array = a.data
                    except ValueError:
                        print('No data supplied')

            if a.time_cut:
                mask = np.array([(data_df.datetime > a.time_cut[0]) & (data_df.datetime < a.time_cut[1])])[0]
                array = array[mask]

            if a.max_val:
                array = array[array <= a.max_val]

            if a.log:
                array = np.log10(abs(array[abs(array) > 0]))

            if a.xlim:
                array = array[(array >= a.xlim[0]) & (array <= a.xlim[1])]

            if a.bins == 'integer':
                a.bins = int(np.max(array)-np.min(array))

            hist, bins = np.histogram(array, bins=a.bins)
            delta_bin = bins[1]-bins[0]
            hist = np.append(0,hist)
            if not a.histo_counts:
                hist = hist*100/np.sum(hist)
            bins_center = np.append(bins[0]-delta_bin,bins[0:-1])+delta_bin/2.
            if a.remove_zeros:
                bins_center = bins_center[hist != 0]
                hist = hist[hist != 0]
            # Add zeros:
            hist = np.append(0,hist)
            hist = np.append(hist,0)
            bins_center = np.append(bins_center[0]-delta_bin,bins_center)
            bins_center = np.append(bins_center,bins_center[-1]+delta_bin)
            if not a.histo_counts:
                hist = hist*100/np.trapz(hist,x=bins_center)
            # hist = hist*100/np.sum(hist)

            if a.labels:
                label = a.labels[_]
            else:
                try:
                    label = aux.pretty_label(histogram_value)
                except:
                    label = histogram_value
            ax1.plot(bins_center,hist,color=a.colors[_],ls=a.ls[_],drawstyle='steps',alpha=a.alpha,label=label)
        if a.legend: plt.legend(fontsize=13)
        if a.ylog:
            ax1.set_yscale('log')
        if a.xlog:
            ax1.set_xscale('log')
        if a.xlim: ax1.set_xlim(a.xlim)
        ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

        if a.ylab: ax1.set_ylabel(a.ylab)
        if a.xlab: ax1.set_xlabel(a.xlab)

        if a.fig_name: plt.savefig(d_plot+a.fig_name+'.png', format='png', dpi=300)

    def PlotPie(self,**kwargs):
        ''' Make pie chart of integrated power fractions.

        Parameters
        ----------

        int_power : list / numpy array
            array of integrated power in different frequency intervals, default: False

        add : bool
            if True, add to existing axis object, default: False

        legend : bool
            if True, add legend to plot, default: False

        alpha : int/float
            transparency index used to plot the histograms, default: 1

        radius : int/float
            outer radius of pie chart, default: 1

        width : int/float
            width of pie chart (allowing for 'donuts'), default: 1

        fig_name : str/bool
            if True, save plot figure with this filename, in d_plot (see top), default: False
    
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['int_power','add','legend','alpha','radius','width','fig_name']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        if a.add:
            ax1 = plt.gca()
        else:
            fig, ax1 = plt.subplots(figsize=(15,10))
            ax1.axis('equal')

        # Don't count intervals with no integrated power
        labels              =   np.array(self.labels)[~np.isnan(a.int_power)]
        colors              =   np.array(self.colors)[~np.isnan(a.int_power)]
        int_power           =   np.array(a.int_power)[~np.isnan(a.int_power)]

        int_power_perc      =   int_power/np.sum(int_power)*100.
        pctdistance         =   0.9 - 0.18 * (1.3 - a.radius) # for placing text
        wedges, texts, autotexts = ax1.pie(int_power_perc, radius=a.radius, colors=self.colors, startangle=90, autopct='%1.0f%%', pctdistance=pctdistance,\
            textprops={'fontsize': 14}, wedgeprops={'width':a.width, 'edgecolor':'white', 'lw':3,'alpha':a.alpha}, counterclock=False)
        plt.setp(autotexts, size=18)#, weight="bold")
        if a.legend: plt.legend(wedges[::-1], self.labels[::-1], loc="best")
        if a.fig_name: plt.savefig(d_plot+a.fig_name+'.png', format='png', dpi=300)

    def PlotTimeSeries(self,**kwargs):
        ''' Plots time series of data for specific time period (all times by default).

        Parameters
        ----------
        col_names : list
            names of the columns containing residual load data to use, default: []

        ax : axis object / bool
            if an axis object is given, timeseries will be plotted on this axis, default: False

        time_cut : list/bool
            start and endtime in np.datetime64-readable format, default: False

        labels : list
            list of labels to describe the histograms, default: False

        ylim : list/bool
            limits in y axis of plot, default: False

        ylab : str
            y axis title, default: 'ower [MW]'

        alpha : int/float
            transparency index used to plot the histograms, default: 1

        colors : list
            list of colors to plot the histograms, default: ['b']

        add_shade : list / bool
            list of colors to optionally add shade under the graphs, default: False

        ls : list
            list of linestyles to plot the histograms, default: ['-']

        two_axes : bool
           if True, add a 2nd y axis, default: False 

        xlim : list/bool
            limits in x axis of plot, default: False

        add : bool
            if True, add to existing axis object, default: False

        legend : bool
            if True, add legend to plot, default: False

        fig_name : str/bool
            if True, save plot figure with this filename, in d_plot (see top), default: False

        fig_format : str
            format of save figure, default: 'png'
    
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['col_names','ax','time_cut','labels','ylim','ylab','alpha',\
            'colors','add_shade','ls','two_axes','add','legend','fig_name','fig_format']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        if a.time_cut:
            a.time_cut = [np.datetime64(a.time_cut[0]),np.datetime64(a.time_cut[1])]

        if a.add:
            fig = plt.gcf()
            if type(a.ax) != bool:
                ax1 = a.ax
            else:
                ax1 = plt.gca()
        else:
            fig = plt.figure(figsize=(15,8))
            ax1 = fig.add_subplot(1,1,1)
            ax1.set_xlabel('Time')
            ax1.set_ylabel(a.ylab)
        
        data_df             =   self.data_df.copy()
        datetimes           =   data_df['datetime'].values

        if not a.labels:
            a.labels = [aux.pretty_label(col_name) for col_name in a.col_names]

        if a.time_cut:
            mask = np.array([(data_df.datetime > a.time_cut[0]) & (data_df.datetime < a.time_cut[1])])[0]
            data_df = data_df[mask]

        if len(a.ls) != len(a.col_names):
            print('List of line styles not the same length as list of column names, will use solid lines')
            a.ls = ['-']*len(a.col_names)

        if a.two_axes:
            do_y_axis = True
            for col_name,color,label,ls in zip(a.col_names,a.colors,a.labels,a.ls):
                try:
                    data = data_df[col_name].values
                except:
                    data = getattr(self,col_name)
                    data = data[mask]
                time = data_df['datetime']
                if 'Price' in col_name:
                    if do_y_axis:
                        ax2 = ax1.twinx()
                        ax2.set_ylabel('Price [EUR]')
                        do_y_axis = False
                    ax2.plot(time,data.real,color=color,ls=ls,label=label,alpha=a.alpha)
                else:
                    if do_y_axis:
                        ax2 = ax1.twinx()
                        ax2.set_ylabel(a.ylab)
                        do_y_axis = False
                    ax2.plot(time,data.real,color=color,ls=ls,label=label,alpha=a.alpha)
        else:
            for col_name,color,label,ls in zip(a.col_names,a.colors,a.labels,a.ls):
                try:
                    data = data_df[col_name].values
                except:
                    data = getattr(self,col_name)
                    data = data[mask]
                time = data_df['datetime']
                ax1.plot(time,data.real,color=color,ls=ls,label=label,alpha=a.alpha)

        if type(a.add_shade) != bool:    
            for shade in a.add_shade:        
                # add shade
                data = data_df[shade].values
                color = [a.colors[_] for _ in range(len(a.colors)) if a.col_names[_] == shade]
                ax1.fill_between(time.values,data*0.,data,\
                    facecolor=color,alpha=0.2, interpolate = False)

        if a.ylim: 
            ax1.set_ylim(a.ylim)

        if a.legend: fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes,fancybox=True,fontsize=14) # this creates a combined legend box for both axes

        if a.fig_name: plt.savefig(d_plot+a.fig_name+'.'+a.fig_format, format=a.fig_format, dpi=500)

    def PlotPowerReqFromiFFT(self,**kwargs):
        ''' Plot power requirements from iDFTs on different frequency intervals

        Parameters
        ----------

        matrices : numpy array / bool
            data to plot = output from GetPowerReqFromiFFT(), default: False 

        col_names : list
            names of the columns containing residual load data to use, default: []

        freq_cut_number : int/bool
            actual frequency cut number to use in self.freq_cuts: False

        yearly : bool
            if True, plot years on x axis, default: False

        width : int/float
            width of each box, default: 1

        offset : int/float
            offset from integer numbers of first box, default: 10

        labels : list
            list of labels to describe the histograms, default: False

        colors : list
            list of colors to plot the histograms, default: ['b']

        ls : list
            list of linestyles to plot the histograms, default: ['-']

        xlabels : list
            list of labels to use on x axis, default: []

        xlim : list/bool
            limits in x axis of plot, default: False
        
        ylim : list/bool
            limits in y axis of plot, default: False

        ylab : str
            y axis title, default: 'ower [MW]'

        add : bool
            if True, add to existing axis object, default: False

        legend : bool
            if True, add legend to plot, default: False

        fig_name : str/bool
            if True, save plot figure with this filename, in d_plot (see top), default: False

        fig_format : str
            format of save figure, default: 'png'
    
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['matrices','col_names','freq_cut_number','yearly','width','offset',\
                                'labels','colors','ls','xlabels','ylim','xlim','ylab',\
                                'add','legend','fig_name','fig_format']
        a                   =   aux.handle_args(kwargs,argkeys_needed)

        if a.add: 
            ax1 = plt.gca()
        else:
            fig = plt.figure(figsize=(10,6))
            ax1 = fig.add_subplot(1,1,1)
        width = a.width
        offset = -1.*np.ceil(len(a.matrices)/2.)*width-width

        if len(a.ls) == len(a.colors):
            ls = a.ls
        else:
            ls = len(a.colors)*a.ls # default linestyle

        if a.yearly:
            offset = (-1*np.ceil(len(a.matrices[0]))+0.5)*width
            if a.offset: offset += a.offset
            for j,submatrices in enumerate(a.matrices):
                for i,iFFT_matrix in enumerate(submatrices):
                    medianprops = dict(linestyle='-', linewidth=2, color=a.colors[i])
                    boxprops = dict(linestyle='-', linewidth=1, color=a.colors[i])
                    whiskerprops = dict(linestyle='-', linewidth=2, color=a.colors[i])
                    capprops = dict(linestyle='-', linewidth=1.5, color=a.colors[i])
                    ax1.boxplot(iFFT_matrix[a.freq_cut_number,:],positions=[j+1+offset+width+i*width],widths=width,\
                        medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops,capprops=capprops)
        else:
            for i,iFFT_matrix in enumerate(a.matrices):
                medianprops = dict(linestyle='-', linewidth=2, color=a.colors[i])
                boxprops = dict(linestyle='-', linewidth=1, color=a.colors[i])
                whiskerprops = dict(linestyle=ls[i], linewidth=2, color=a.colors[i])
                capprops = dict(linestyle='-', linewidth=1.5, color=a.colors[i])
                if len(a.col_names) > 1: ax1.boxplot(iFFT_matrix[i,:,:].T,positions=np.arange(len(self.freq_cuts))+1+offset+width+i*width,widths=width,\
                    medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops,capprops=capprops)
                if len(a.col_names) == 1: ax1.boxplot(iFFT_matrix.T,positions=np.arange(len(self.freq_cuts))+1+offset+width+i*width,widths=width,\
                    medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops,capprops=capprops)

        for color,label,ls in zip(self.colors,self.labels,a.ls):
            ax1.plot([-1],[-1],linestyle=ls, linewidth=2,color=color,label=label)

        ax1.set_ylabel('Power [MW]')
        if a.ylab: ax1.set_ylabel(a.ylab)
        ax1.set_xlabel('Timescales')
        if a.legend: plt.legend(loc='best')
        ax1.set_xticks(np.arange(len(a.xlabels))+1) 
        ax1.set_xticklabels(a.xlabels) 
        if a.ylim: ax1.set_ylim(a.ylim)
        if a.xlim: ax1.set_xlim(a.xlim)
        ax1.grid()
        if a.fig_name: plt.savefig(d_plot+a.fig_name+'.'+a.fig_format, format=a.fig_format, dpi=300)