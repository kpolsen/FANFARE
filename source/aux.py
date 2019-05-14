# coding=utf-8
### 
### Submodule aux
###

print('fanfare submodule "aux" imported')

import sys
import numpy as np
import pandas as pd
from scipy import signal
from argparse import Namespace


def handle_args(kwargs,argkeys_needed,verbose=False):
    """ Make custom namespace containing the relevant parameters for a specific method

    Parameters
    ----------
    kwargs : dictionary
        keyword arguments supplied to the function by the user
    args_needed : dictionary
        arguments needed by the function
    verbose : bool
        if True, print which arguments were not supplied by user and have been replaced by default value, default: False

    """

    args                =   Namespace(add=False,\
                            add_shade=False,\
                            alpha=1,\
                            alpha_wind_cuts=[10,90],\
                            alpha_cuts=False,\
                            
                            bal_price_name='',\
                            bins=200,\
                            
                            col_indices='',\
                            col_name=False,\
                            col_names=[],\
                            color='b',\
                            colors=['b'],\
                            compare=False,\
                            comp_dev=50,\
                            cons_ob='',\

                            d_data='data/',\
                            data=False,\
                            data_df=False,\
                            data_type=False,\
                            depth = False,\
                            depth_range = [-50,400],\
                            df_cols='',\
                            duration_ramp_min = 60,\
                            duration_range = [0,8*60*60],\
                            duration_cuts = False,\

                            exc_dev=1000,\
                            epoch=-1,\

                            fig_name=False,\
                            file_name=False,\
                            file_path=False,\
                            fig_format='png',\
                            freq_cut=[False],\
                            freq_cuts=[False],\
                            freq_cut_number=False,\
                            freq_cut_numbers=[],\
                            
                            histogram_values = [],\
                            histo_counts = False,\

                            include_storms = False,\
                            int_power = [],\

                            label='',\
                            labels=False,\
                            labels_verbose=False,\
                            legend=False,\
                            load=False,\
                            load_name=False,\
                            load_names=[],\
                            log=False,\
                            line_limit=5,\
                            ls=['-'],\

                            magnitude_range=[-2000,2000],\
                            matrices=False,\
                            max_val=1000,\
                            max_ramp=30,\
                            max_step=False,\
                            method='append',\
                            min_dip=-30,\

                            new_NaN_value=0,\
                            new_y_axis=False,\
                            number_of_storms=100,\

                            offset = 10,\
                            offsets = [1,2,3,4],\
                            overwrite = False,\

                            percent = 95,
                            power_name=False,\
                            power_names=[],\
                            price_name='',\

                            radius=1,\
                            raw_data_names='',\
                            region='DK',\
                            regions=['DK','DK1','DK2','BO'],\
                            remove_zeros=False,\

                            scale_by_capacity=False,\
                            SDAtype='duration',\
                            season=False,\
                            secs_or_minutes='secs',\
                            skiprows=1,\
                            spot_price_name='',\

                            test=False,\
                            test_plot=False,\
                            time='',\
                            time_cut=False,\
                            time_diff='year',\
                            time_period='year',\
                            time_res='1hour',\
                            times=False,\
                            two_axes=False,\

                            verbose=True,\
                            
                            width=1,\

                            xlab='',\
                            xlabels=[],\
                            xlim=False,\
                            xlog=False,\

                            year=False,\
                            yearly=False,\
                            years=[2018],\
                            ylim=False,\
                            ylog=False,\
                            ylab='Power [MW]',\

                            zlog=False,\
                            zone='DK',\

                            )
    # Fill up new empty dictionary
    for key in argkeys_needed:
        if key in kwargs: 
            setattr(args,key,kwargs[key])
        else:
            # args[key] = getattr(default_args,key)
            if verbose: print('"%s" argument not passed, using default value' % key)

    return(args)



def pretty_label(name,percent=False):
    """ Creates a pretty label for plotting etc. for given column name.
    """

    label = name

    if name == 'TotalRenPower': label = 'All Renewable Power in DK' 
    if name == 'TotalRenPower_DK1': label = 'All Renewable Power in DK1' 
    if name == 'TotalRenPower_DK2': label = 'All Renewable Power in DK2' 
    if name == 'TotalRenPower_BO': label = 'All Renewable Power in Bornholm' 
    if name == 'TotalResLoad': label = 'All Residual Load in DK' 
    if name == 'TotalResLoad_DK1': label = 'All Residual Load in DK1' 
    if name == 'TotalResLoad_DK2': label = 'All Residual Load in DK2' 
    if name == 'TotalResLoad_BO': label = 'All Residual Load in Bornholm' 
    if name == 'TotalWindPower': label = 'All Wind Power in DK' 
    if name == 'TotalWindPower_DK1': label = 'Wind Power in DK1' 
    if name == 'TotalWindPower_DK2': label = 'Wind Power in DK2' 
    if name == 'TotalWindPower_BO': label = 'Wind Power in Bornholm' 
    if name == 'accum_capacity_TotalWindPower_DK': label = 'Installed capacity in DK' 
    if name == 'accum_capacity_TotalWindPower_DK1': label = 'Installed capacity in DK1' 
    if name == 'accum_capacity_TotalWindPower_DK2': label = 'Installed capacity in DK2' 
    if name == 'accum_capacity_TotalWindPower_BO': label = 'Installed capacity in Bornholm' 
    if name == 'BalancingPowerPriceUpEUR_DK1': label = 'Balancing price up in DK1 [EUR]' 
    if name == 'BalancingPowerPriceUpEUR_DK2': label = 'Balancing price up in DK2 [EUR]' 
    if name == 'BalancingPowerPriceDownEUR_DK1': label = 'Balancing price down in DK1 [EUR]' 
    if name == 'BalancingPowerPriceDownEUR_DK2': label = 'Balancing price down in DK2 [EUR]' 
    if name == 'SpotPriceEUR': 
        label = 'Spot price [EUR]'
    if name == 'ResidualPrice_DK1': 
        label = 'Residual price in DK1' 
        if percent: label = label + ' [% of spot price]'
    if name == 'ResidualPrice_DK2': 
        label = 'Residual price in DK2' 
        if percent: label = label + ' [% of spot price]'
    if name == 'OnshoreWindPower': label = 'Onshore Wind Power in DK' 
    if name == 'OffshoreWindPower': label = 'Offshore Wind Power in DK' 
    if name == 'GrossCon': 
        label = 'Gross Consumption in DK' 
        if percent: label = label + ' [% of consumption]'
    if name == 'GrossCon_DK1': 
        label = 'Gross Consumption in DK1' 
        if percent: label = label + ' [% of consumption]'
    if name == 'GrossCon_DK2': 
        label = 'Gross Consumption in DK2' 
        if percent: label = label + ' [% of consumption]'
    if name == 'GrossCon_BO': 
        label = 'Gross Consumption in Bornholm' 
    if name == 'alpha_TotalWindPower': label = r'$\alpha_{wind}$ in DK' 
    if name == 'alpha_TotalWindPower_DK1': label = r'$\alpha_{wind}$ in DK1' 
    if name == 'alpha_TotalWindPower_DK2': label = r'$\alpha_{wind}$ in DK2' 
    if name == 'alpha_TotalRenPower': label = r'$\alpha_{\rm VRE}$ in DK' 
    if name == 'alpha_TotalRenPower_DK1': label = r'$\alpha_{\rm VRE}$ in DK1' 
    if name == 'alpha_TotalRenPower_DK2': label = r'$\alpha_{\rm VRE}$ in DK2' 
    if name == 'alpha_TotalRenPower_BO': label = r'$\alpha_{\rm RE}$ in BO' 
    if name == 'SolarPower': label = 'All Solar Power in DK' 
    if name == 'SolarPower_BO': label = 'Solar Power in Bornholm' 
    if name == 'BioPower_BO': label = 'Bio Power in Bornholm' 
    if name == 'Import_BO': label = 'Import/export in Bornholm' 


    return(label)