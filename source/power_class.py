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

# fanfare modules
import source.aux as aux

# path to save plots
d_plot = '../plots/'

class PowerData():
    ''' This class defines an object that contains the time series of power produced and consumed from a specific dataset (e.g. DK1 or Bornholm). 
    '''