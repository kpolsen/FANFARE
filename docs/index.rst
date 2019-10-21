.. FANFARE documentation master file, created by
   sphinx-quickstart on Sun May 12 15:47:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :hidden:
   :maxdepth: 1

   Overview <self>
   FANFARE module <fanfare>
   Obtaining the code <release>
   Requirements <reqs>
   Tutorials <tutorials>

.. role:: python(code)
    :language: python


.. figure:: figures/favicon.png
   :width: 150px
   :align: left

Welcome to the documentation of FANFARE!
========================================

:python:`FANFARE` stands for 'Fast Assessment Numerical tool for Frequency Analysis of Renewable Energy integration scenarios' and can be applied to power system timeseries data of both real and simulated scenarios. 



Quick example
-------------

:python:`FANFARE` reads in a timeseries of electricity production from renewable energy sources together with a timeseries of electricity consumption over the same time period. A Fourier analysis of the residual load is used to derive flexibility requirements, such as the integrated energy stored in oscillations of different timescales:

.. figure:: figures/energy_DK_pie.png
   :width: 900px
   :align: center

In the above figure, the regions of Bornholm (BO), East Denmark (DK2), West Denmark (DK1) and all of Denmark (DK) are compared.

Purpose
-------

The purpose of this code, is to quantify the need for flexibility in a system with significant share of **variable renewable energy (VRE)** and an offset between electricity consumption and VRE generation. This offset is often called residual load, and it can contain fluctuations on different timescales. Below is a view of 2 weeks of power system data from Denmark in 2017, to illustrate how wildly the residual load can fluctuate on hourly timescales:

.. figure:: figures/timeseries_shade.png
   :width: 700px
   :align: center

In order to handle these fluctuations in a fossil-free future, solutions must be found in terms of **energy storage (ES)** and **demand response (DR)**. Many of these solutions are already well on their way to being fully implemented, but each typically work on specific timescales.

In order to disentangle the residual load on different timescales, :python:`FANFARE` adopts a Discrete Fourier Transform (DFT) analysis. This has often been done before in power system analysis - see for instance `Arrigo et al. 2017 <https://www.tib.eu/en/search/id/ieee%3Adoi~10.1109%252FISGTEurope.2017.8260312/Fourier-transform-based-procedure-for-investigations/>`_, `Oh 2018 <https://www.sciencedirect.com/science/article/pii/S0960148117309904>`_, `Heggarty et al. 2019 <https://www.sciencedirect.com/science/article/pii/S0306261919302107>`_.


:python:`FANFARE` consists of a set of methods that can be applied to timeseries data of residual load, in order to study the flexibility requirements on different timescales. 
These methods are illustrated in the flow chart below.

.. figure:: figures/flow.png
   :width: 700px
   :align: center

Contact
-------
For questions, contact Karen at:
pardos at dtu.dk

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

