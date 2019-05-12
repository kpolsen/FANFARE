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

    # DATA ANALYSIS -------------------------------------------------------------------------------------------

    def AddHighPenetrationIndex(self,**kwargs):
        ''' Add index to separate high wind penetration from low
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['alpha_cuts','power_name','load_name','yearly','time_cut','overwrite']
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


    # PLOTTING -------------------------------------------------------------------------------------------


    def PlotHisto(self,**kwargs):
        ''' Plot histogram of an attribute, a column in the dataframe or a supplied data array.
        '''

        # create a custom namespace for this method
        argkeys_needed      =   ['histogram_values','epoch','fig_name','bins','max_val','log','ylog','power_name',\
            'xlog','labels','colors','ls','alpha','xlim','xlab','ylab','add','data','remove_zeros','legend','histo_counts','time_cut']
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

            # if a.epoch != 2:
            #     array = array[data_df['low_penetration_index_'+ a.power_name] == a.epoch]
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
