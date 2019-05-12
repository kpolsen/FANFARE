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
    """ Make custom namespace for specific method
    """

    args                =   Namespace(add=False,\
                            alpha=1,\
                            alpha_wind_cuts=[10,90],\
                            alpha_cuts=False,\
                            
                            bal_price_name='',\
                            bins=200,\
                            
                            col_indices='',\
                            col_names=[],\
                            color='b',\
                            colors=['b','r','k'],\
                            compare=False,\
                            comp_dev=50,\
                            cons_ob='',\

                            d_data='../../data/',\
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
                            fig_format='pdf',\
                            freq_cut=[False],\
                            freq_cuts=[False],\
                            
                            histogram_values = [],\

                            include_storms = False,\
                            int_power = [],\

                            label='',\
                            labels=False,\
                            legend=False,\
                            load=False,\
                            load_names=[],\
                            log=False,\
                            line_limit=80000,\
                            ls=['-'],\

                            magnitude_range=[-2000,2000],\
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
                            percent = 95,
                            power_name='TotalWindPower',\
                            power_names=[],\
                            price_name='',\

                            raw_data_names='',\
                            region='DK',\
                            regions=['DK','DK1','DK2','BO'],\
                            remove_zeros=False,\

                            scale_by_capacity=False,\
                            SDAtype='duration',\
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
                            
                            xlab='',\
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