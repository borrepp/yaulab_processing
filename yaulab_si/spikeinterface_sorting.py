import os
import shutil
import numpy
import pickle
import json
import datetime
import time
import copy
import matplotlib.pyplot as plt

from ..yaulab_extras import(
    waveclus_path, 
    supported_probes_manufacturer, 
    get_tempdir, 
    clear_tempdir
    )

from .spikeinterface_tools import(
    getUnique_electrodeGroups,
    select_electrodeGroup_and_session_info,
    get_si_recording,
    plot_concatenations,
    plotPSD_randomChunks,
    plot_peakLocations,
    plot_motion_outputs
    )

import spikeinterface.core as si
from spikeinterface.core import get_template_extremum_channel_peak_shift
from spikeinterface.core.core_tools import check_json

import spikeinterface.preprocessing as spre
from probeinterface.plotting import plot_probe

from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.motion import estimate_motion, interpolate_motion

from spikeinterface.sorters import run_sorter, read_sorter_folder, WaveClusSorter
from spikeinterface.extractors import NpzSortingExtractor

import mountainsort5 as ms5
from warnings import warn

from spikeinterface.postprocessing import align_sorting
import spikeinterface.curation as scur
import spikeinterface.qualitymetrics as sqm
from spikeinterface.exporters import export_to_phy


####################################################################################################################################################################################
#                                                                    SOME DEFAULT PARAMETERS
####################################################################################################################################################################################
processName_prefix = 'SIprepSort'

n_cpus = os.cpu_count()
n_jobs = n_cpus - 2

job_kwargs = dict(chunk_duration="1s", n_jobs=n_jobs, progress_bar=True)

# si.set_global_job_kwargs(**job_kwargs)

######################################################################################################
# PARAMETER FOR "LOCAL REFERENCING"
exclude_radius_chans_default = 1 # Number of neighbor channels to exclude because are too close to the reference channel
include_radius_chans_default = 4 # Number of neighbor channels delineates the outer boundary of the annulus whose role is to exclude channels that are too far away

noisy_freq_default = None

######################################################################################################
# GENERAL PARAMETER FOR "PEAKS LOCATIONS"
ms_before_default = 0.6
ms_after_default = 1.5
peak_sign_default = 'both' # (“neg” | “pos” | “both”)
nearest_chans_default = 3 # Number of neighbor channels to search for the same waveform

###########################################################
# DETECT PEAKS: 
peak_detect_threshold_default = 5 #  MAD: Median Amplitude Deviations

"""
######################################################################################################
                                    PEAK DETECTION OPTIONS
######################################################################################################
peaks_options_default = {
    'method': "locally_exclusive", ('by_channel' | 'locally_exclusive')
    'peak_sign': "both", 
    'detect_threshold': 5, # MAD: Median Amplitude Deviations
    'radius_um': step_chan*3.0, # not an option in case of 'by_channel' method
}

######################################################################################################
                        PEAKS LOCATIONS OPTIONS
######################################################################################################
locations_options = {     
    'ms_before': 0.6,
    'ms_after': 1.5,
    'location_method': 'monopolar_triangulation', # Paninski Lab
    'location_kwargs': {'max_distance_um': step_chan*10.0, 'optimizer': 'least_square'}
}

locations_options = {
    'ms_before': 0.6,
    'ms_after': 1.5,
    'location_method': 'grid_convolution', # Kilosort-like
    'location_kwargs': {'upsampling_um': step_chan/2.0}
}
"""

######################################################################################################
# MOTION ESTIMATION & INTERPOLATION to run : non-rigid + decentralized + kriging
######################################################################################################

######################################################################################################
# Rigid or nonRigid
motion_rigid_default = False # (True | False) Default: False

######################################################################################################
# dredge_ap : Paninski Lab
motion_options_default = {
    'method': 'dredge_ap', # Paninski Lab
    'method_kwargs' : {} 
}
"""
######################################################################################################
                                    MOTION OPTIONS
######################################################################################################
                                    
######################################################################################################
# MOTION CORRECTION METHOD 1:
# dredge_ap : Paninski Lab
motion_options_default = {
        'method': 'dredge_ap', # Paninski Lab
        'method_kwargs' : {}, 
}
######################################################################################################
# MOTION CORRECTION METHOD 2:
# decentralized : Paninski Lab - like
motion_options_default = {'method': 'decentralized', # Paninski Lab - like
        'method_kwargs' : {
        'pairwise_displacement_method':'conv', # "conv" | "phase_cross_correlation" 
        'convergence_method': 'gradient_descent',  # 'lsmr' | 'lsqr_robust' | 'gradient_descent'
        'force_spatial_median_continuity': True
    },
}
######################################################################################################
# MOTION CORRECTION METHOD 3:
#  iterative_template 
motion_options_default = {
    'method': 'iterative_template', # Kilosort-like
    'method_kwargs' : {},
}
"""
######################################################################################################
# 'kriging' - Kilosort-like & 'remove_channels'
interpolate_options_default = {
    'method': 'kriging', # ('kriging' | 'idw' | 'nearest') 'idw': inverse distance weighted.  Default: 'kriging' - Kilosort-like
    'border_mode': 'remove_channels' # ('remove_channels' | 'force_extrapolate' | 'force_zeros') Default: 'remove_channels'
}

sorterName_default = 'kilosort4'

##########################################
# Default parameters for Mountainsort5
mountainsort5_scheme_default = '2'
mountainsort5_run_local = True


maxChans_withoutSparsity_default = 16

####################################################################################################################################################################################
####################################################################################################################################################################################
#                                                                   MAIN FUNCTIONS
####################################################################################################################################################################################
####################################################################################################################################################################################


####################################################################################################################################################################################
# Helper function to search for unique *NWB files from expDATES within a range of dates
####################################################################################################################################################################################
def get_expDay_in_range(parentRecordingFolder, year_start, month_start, day_start, year_stop, month_stop, day_stop):
    
    filesDate_log = []
    filesData_numList = []
    for _, _, files in os.walk(parentRecordingFolder):

        for name in files:

            nameSplit = os.path.splitext(name)

            if nameSplit[1]=='.nwb' and '-noNEV' not in nameSplit[0]:

                fileName = nameSplit[0]
                yearFile = int(fileName[3:7])
                monthFile = int(fileName[8:10])
                dayFile = int(fileName[11:13])

                fileIN = True
                if yearFile>=year_start and yearFile<=year_stop:
                    if yearFile==year_start:
                        if monthFile<month_start:
                            fileIN = False
                        elif monthFile==month_start and dayFile<day_start:
                            fileIN = False
                    if yearFile==year_stop:
                        if monthFile>month_stop:
                            fileIN = False
                        elif monthFile==month_stop and dayFile>day_stop:
                            fileIN = False
                if fileIN:
                    file_label = '{}-{:02d}-{:02d}'.format(yearFile, monthFile, dayFile)
                    if file_label not in filesDate_log:
                        filesDate_log.append(file_label)
                        filesData_numList.append([yearFile, monthFile, dayFile])
    
    # Force Unique and increasing 
    dateSort = numpy.unique(numpy.array(filesData_numList), axis=0)

    return dateSort



####################################################################################################################################################################################
# RUN PREPROCESSING for all the Electrode/Probes and sessions from expDays within a range of dates
####################################################################################################################################################################################
def run_prepro_expDAY_in_range(parentRecordingFolder, parentPreproSortingFolder,
        year_start, month_start, day_start, 
        year_stop, month_stop, day_stop,
        local_radius_chans = (exclude_radius_chans_default, include_radius_chans_default), 
        noisy_freq = noisy_freq_default, 
        ms_before = ms_before_default, 
        ms_after = ms_after_default, 
        peak_sign = peak_sign_default,
        nearest_chans = nearest_chans_default, 
        peak_detect_threshold = peak_detect_threshold_default, 
        do_detect_bad_channels = True,
        do_motion = True,
        motion_rigid = motion_rigid_default, 
        motion_options = motion_options_default, 
        interpolate_options = interpolate_options_default,
        localProcess_NWB = False,
        rewrite_prepro = True
    ):

    dateSort = get_expDay_in_range(parentRecordingFolder, year_start, month_start, day_start, year_stop, month_stop, day_stop)

    for n in range(dateSort.shape[0]):

        sessionYear=dateSort[n, 0]
        sessionMonth=dateSort[n, 1]
        sessionDay=dateSort[n, 2]
        
        run_prepro_expDAY(parentRecordingFolder, parentPreproSortingFolder, sessionYear, sessionMonth, sessionDay,
            local_radius_chans = local_radius_chans, 
            noisy_freq = noisy_freq, 
            ms_before = ms_before, 
            ms_after = ms_after, 
            peak_sign = peak_sign,
            nearest_chans = nearest_chans, 
            peak_detect_threshold = peak_detect_threshold, 
            do_detect_bad_channels = do_detect_bad_channels,
            do_motion = do_motion,
            motion_rigid = motion_rigid, 
            motion_options = motion_options, 
            interpolate_options = interpolate_options,
            localProcess_NWB = localProcess_NWB,
            rewrite_prepro = rewrite_prepro
        )



####################################################################################################################################################################################
# RUN PREPROCESSING for all the Electrode/Probes (ALL sessions) for a given expDay
####################################################################################################################################################################################
def run_prepro_expDAY(parentRecordingFolder, parentPreproSortingFolder, sessionYear, sessionMonth, sessionDay,
        local_radius_chans = (exclude_radius_chans_default, include_radius_chans_default), 
        noisy_freq = noisy_freq_default, 
        ms_before = ms_before_default, 
        ms_after = ms_after_default, 
        peak_sign = peak_sign_default,
        nearest_chans = nearest_chans_default, 
        peak_detect_threshold = peak_detect_threshold_default, 
        do_detect_bad_channels = True,
        do_motion = True,
        motion_rigid = motion_rigid_default, 
        motion_options = motion_options_default, 
        interpolate_options = interpolate_options_default,
        localProcess_NWB = False, 
        rewrite_prepro = True
    ):

    print('Preprocessing exp{}-{:02d}-{:02d}'.format(sessionYear, sessionMonth, sessionDay))

    electrodeGroups = getUnique_electrodeGroups(parentRecordingFolder, sessionYear, sessionMonth, sessionDay)

    for i in range(len(electrodeGroups)):

        if any([prefix in electrodeGroups[i]['probeInfo']['probeName'] for prefix in supported_probes_manufacturer]):
            
            electrodeGroup_sessions = select_electrodeGroup_and_session_info(electrodeGroups, sessionYear, sessionMonth, sessionDay, electrodeGroup_Name=electrodeGroups[i]['electrodeName'], session_index=None)

            run_prepro(parentRecordingFolder, parentPreproSortingFolder, sessionYear, sessionMonth, sessionDay, 
                electrodeGroup_sessions = electrodeGroup_sessions,
                local_radius_chans = local_radius_chans, 
                noisy_freq = noisy_freq, 
                ms_before = ms_before, 
                ms_after = ms_after, 
                peak_sign = peak_sign,
                nearest_chans = nearest_chans, 
                peak_detect_threshold = peak_detect_threshold, 
                do_detect_bad_channels = do_detect_bad_channels,
                do_motion = do_motion,
                motion_rigid = motion_rigid, 
                motion_options = motion_options, 
                interpolate_options = interpolate_options,
                localProcess_NWB = localProcess_NWB,
                rewrite_prepro = rewrite_prepro,
                return_recording = False 
            )

            print('\n\nexp{}-{:02d}-{:02d} ElectrodeGroup : {} was processed ¡¡¡\n\n'.format(sessionYear, sessionMonth, sessionDay, electrodeGroups[i]['electrodeName']))

        else:

            print('\nProbe: {} was not found as a valid Device\nFrom exp{}-{:02d}-{:02d}, ElectrodeGroup : "{}"\nIt will NOT be preprocessing\n\n'.format(
                electrodeGroups[i]['probeInfo']['probeName'], sessionYear, sessionMonth, sessionDay, electrodeGroups[i]['electrodeName']))


####################################################################################################################################################################################
# Function to set up parameters to run Sorter & to create Sorting_analyzer
####################################################################################################################################################################################
def get_sortingAnalyzer_params(nChans, step_chan, ms_before, ms_after, peak_sign, nearest_chans):

    # Validate Nearest channels relative to the number of channels in the recording object
    #"nearest_chans = 0" will perform similar to sorting single channels.
    if nChans==1:
        nearest_chans = 0 
    elif nearest_chans>nChans:
        nearest_chans = nChans

    if nearest_chans < 1:
        radius_um = step_chan/2
        location_method = "center_of_mass"
        location_kwargs = {
            'radius_um': radius_um, 
            'feature': "ptp" #"ptp" | "mean" | "energy" | "peak_voltage", default: "ptp"
        }
        unit_location_kwargs = location_kwargs.copy()

    else:
        radius_um = step_chan*nearest_chans
        location_method = "monopolar_triangulation"
        location_kwargs = {
            'radius_um': radius_um, 
            'max_distance_um': step_chan*10,
            'optimizer': 'least_square'
        }
        unit_location_kwargs = location_kwargs.copy()
        unit_location_kwargs.update({'return_alpha': True})
    
    ####################################################################################
    # Sparsity will be the first extension to be computed.
    estimate_sparsity_params = {
        'num_spikes_for_sparsity': 500, # How many spikes per units to compute the sparsity (default: int = 100) 
        'ms_before': ms_before, # Cut out in ms before spike time (default: float = 1.0)
        'ms_after':  ms_after, # Cut out in ms after spike time (default: float'= 2.5)
        'method': "radius", # ("radius" | "best_channels" | "amplitude" | "snr" | "by_property" | "ptp" (default: str = 'radius')
        'peak_sign': peak_sign, # Sign of the template to compute best channels (“neg” | “pos” | “both” (default: str = 'neg') 
        'radius_um': radius_um, # (default: float = 100.0) Radius in um for “radius” method
        'num_channels': nearest_chans, # Used for “best_channels” method (default: int = 5)
    }

    """
        Construct sparsity from N best channels with the largest amplitude.
        Use the "num_channels" argument to specify the number of channels.

        Parameters
        ----------
        templates_or_sorting_analyzer : Templates | SortingAnalyzer
            A Templates or a SortingAnalyzer object.
        num_channels : int
            Number of channels for "best_channels" method.
        peak_sign : "neg" | "pos" | "both"
            Sign of the template to compute best channels.
        amplitude_mode : "extremum" | "at_index" | "peak_to_peak", default: "extremum"
            Mode to compute the amplitude of the templates.
            
        ###########################################################
        peak_sign :  "neg" | "pos" | "both"
            Sign of the template to find extremum channels
        mode : "extremum" | "at_index" | "peak_to_peak", default: "at_index"
            Where the amplitude is computed
            * "extremum" : take the peak value (max or min depending on `peak_sign`)
            * "at_index" : take value at `nbefore` index
            * "peak_to_peak" : take the peak-to-peak amplitude
    """

    ####################################################################################
    # After sparsity is perfomed, then the rest of Postprocessing extensions can be listed:
    # All extensions.
    # Default parameters are listed as comments
    if nearest_chans>=4:
        include_multi_channel_metrics = True
        metric_names = ['peak_to_valley', 'peak_trough_ratio', 'halfwidth', 'repolarization_slope', 'recovery_slope', 'num_positive_peaks', 'num_negative_peaks',
                            'velocity_above', 'velocity_below', 'exp_decay', 'spread']
    else:
        include_multi_channel_metrics = False
        metric_names = ['peak_to_valley', 'peak_trough_ratio', 'halfwidth', 'repolarization_slope', 'recovery_slope', 'num_positive_peaks', 'num_negative_peaks']

    sorting_analyzer_params = {
            'random_spikes': {'method': 'uniform', 'max_spikes_per_unit': 500, 'margin_size': None}, # 'method': 'uniform' | 'all', 'max_spikes_per_unit': 500, 'margin_size': None
            'noise_levels': {},  # method : 'mad' | 'std', str default = 'mad' # it is not fully integrated to Extensions factory, it doesn't have the function "._set_params()" 
            'correlograms': {'window_ms' : 50.0, 'bin_ms': 1.0}, # 'window_ms' : 50 (if 50 ms, the correlations will be computed at lags -25 ms … 25 ms), 'bin_ms' : 1
            'isi_histograms': {'window_ms' : 50.0, 'bin_ms': 1.0}, # 'window_ms' : 50, 'bin_ms' : 1
            'waveforms': {'ms_before': ms_before, 'ms_after': ms_after},
            'principal_components': {'n_components': 5, 'mode': 'by_channel_local'}, # 'n_components': 5, 'mode': 'by_channel_local' | by_channel_global, default: by_channel_local
            'templates': {'operators': ["average"], 'ms_before': ms_before, 'ms_after': ms_after}, # The operators to compute. Can be "average", "std", "median", "percentile" , 'ms_before': 1, 'ms_after': 2
            'template_metrics': {'peak_sign': peak_sign, 
                                'upsampling_factor': 20, # The upsampling factor to upsample the templates, default: 10
                                'sparsity': None, # If None, template metrics are computed on the extremum channel only. If sparsity is given, template metrics are computed on all sparse channels of each unit. Default: None
                                'include_multi_channel_metrics': include_multi_channel_metrics, # Whether to compute multi-channel metrics (At least 10 channels shoulb be capturing the waveforms)
                                'metric_names': metric_names, # ['peak_to_valley', 'peak_trough_ratio', 'halfwidth', 'repolarization_slope', 'recovery_slope', 'num_positive_peaks', 'num_negative_peaks',
                                                    # 'velocity_above', 'velocity_below', 'exp_decay', 'spread'], # the following multi-channel metrics can be computed (when include_multi_channel_metrics=True)
                                'recovery_window_ms': 0.7, # the window in ms after the peak to compute the recovery_slope
                                'peak_relative_threshold': 0.2, #the relative threshold to detect positive and negative peaks, default: 0.2
                                'peak_width_ms': 0.1, # the width in samples to detect peaks, default: 0.1
                                'depth_direction': "y", # the direction to compute velocity above and below, default: "y" (see notes)
                                'min_channels_for_velocity': 5, #the minimum number of channels above or below to compute velocity, default: 5
                                'min_r2_velocity': 0.5, # the minimum r2 to accept the velocity fit, default: 0.5
                                'exp_peak_function': 'ptp', #the function to use to compute the peak amplitude for the exp decay, default: "ptp"
                                'min_r2_exp_decay': 0.5, # the minimum r2 to accept the exp decay fit, default: 0.5
                                'spread_threshold': 0.2, # the threshold to compute the spread, default: 0.2
                                'spread_smooth_um': step_chan, # the smoothing in um to compute the spread, default: 20
                                'column_range': None,   # the range in um in the horizontal direction to consider channels for velocity, default: None, If None, all channels all channels are considered, 
                                                        # If 0 or 1, only the "column" that includes the max channel is considered
                                                        # If > 1, only channels within range (+/-) um from the max channel horizontal position are used
                                    # Notes
                                    #    -----
                                    #    If any multi-channel metric is in the metric_names or include_multi_channel_metrics is True, sparsity must be None,
                                    #    so that one metric value will be computed per unit.
                                    #    For multi-channel metrics, 3D channel locations are not supported. By default, the depth direction is "y".    
                                },
            'template_similarity': {'method': 'cosine', 'max_lag_ms': 0.0, 'support': "union"}, # 'method': “cosine” | “l2” | “l1”, 'max_lag_ms': 0.0 support: “dense” | “union” | “intersection”, default: “union”
            'amplitude_scalings': {'ms_before': ms_before, 'ms_after': ms_after, 'handle_collisions': True, 'delta_collision_ms': 2, 'max_dense_channels':nChans+1}, # 'handle_collisions': True, delta_collision_ms: 2
            'spike_amplitudes': {'peak_sign': peak_sign}, # ( “neg” | “pos” | “both”, default: str = 'neg') 
            'spike_locations': {
                'ms_before': ms_before, # ms_before : 0.5,
                'ms_after': ms_after, # ms_after : 0.5, 
                'method': location_method, # 'method': "center_of_mass" | "monopolar_triangulation" | "grid_convolution", default: 'center_of_mass'
                'method_kwargs': location_kwargs,
                'spike_retriver_kwargs': {
                        'channel_from_template': False, # For each spike is the maximum channel computed from template or re estimated at every spikes. float, default: 50
                        'radius_um': radius_um, # In case channel_from_template=False, this is the radius to get the true peak. bool, default = True
                        'peak_sign': peak_sign, # In case channel_from_template=False, this is the peak sign. ( “neg” | “pos” | “both”, default: str = 'neg')
                    },
                },  
            'unit_locations': {
                'method': location_method,  #'method': "center_of_mass" | "monopolar_triangulation" | "grid_convolution", default: 'center_of_mass'
                **unit_location_kwargs
                } 
        }
    
    return nearest_chans, radius_um, estimate_sparsity_params, sorting_analyzer_params

####################################################################################################################################################################################
# Function to set up parameters to run Sorter & to create Sorting_analyzer
####################################################################################################################################################################################
def get_sorter_and_sortingAnalyzer_params(sorterName, nChans, step_chan, sampling_frequency, recording_total_duration, ms_before, ms_after, peak_sign, nearest_chans, sorter_whitening, detect_threshold=None):

    nearest_chans, radius_um, estimate_sparsity_params, sorting_analyzer_params = get_sortingAnalyzer_params(nChans, step_chan, ms_before, ms_after, peak_sign, nearest_chans)    
    
    if 'both' in peak_sign:
        sign_label = '' # Default
        detect_sign = 0
    elif 'neg' in peak_sign:
        detect_sign = -1
        sign_label = 'N'
    elif 'pos' in peak_sign:
        detect_sign = 1
        sign_label = 'P'

    snippet_T1 = int(numpy.ceil(ms_before * sampling_frequency / 1000.0))
    snippet_T2 = int(numpy.ceil(ms_after * sampling_frequency / 1000.0))
        
    nt =  snippet_T1 + snippet_T2

    if sorterName=='waveclus':
        if detect_threshold is None:
            detect_threshold = 5 # Whitened Threshold (STD from Absolute traces)
        
        if not os.path.isdir(waveclus_path):
            raise Exception('WaveClus path is WRONG.\nExpected folder with waveclus-master at:{}\nCopy the folder to this path or change useDocker to TRUE'.format(waveclus_path))
        # Set the path where the MATLAB toolbox is
        WaveClusSorter.set_waveclus_path(waveclus_path)
        sorter_info = dict(
                sorter_name = 'waveclus',
                sorter_label = 'WCLUSth{}{}'.format(sign_label, detect_threshold).replace('.', '_'),
                useDocker = False, 
                params2update = {
                        'feature_type' : 'wav', # (for wavelets) or pca, type of feature extraction applied to the spikes (type:<class 'str'>) Default = wav
                        'detect_threshold': detect_threshold,
                        'detect_sign' : detect_sign,
                        'w_pre': snippet_T1, 
                        'w_post': snippet_T2,
                        'enable_detect_filter': False, 
                        'enable_sort_filter':False,  
                        'mintemp' : 0, 
                        'maxtemp': 0.251,
                        'template_sdnum': 3
                }
        )
    elif sorterName=='mountainsort5':

        if detect_threshold is None:
            detect_threshold = 5.5 # ZCA-Whitening traces

        if nearest_chans>1:
            scheme2_detect_channel_radius = radius_um/2 # Based on default values from SpikeInterface # half of the radius from sheme2_phase1
        else:
            scheme2_detect_channel_radius = radius_um
        
        # Params about mountainsort5:
        scheme3_block_duration_sec = int(min([1800, recording_total_duration/4])) # Divide the recording at least in 4 sblocks when searching for drift
        scheme2_training_duration_sec = int(min([300, scheme3_block_duration_sec/6])) # Training for template

         # Method copyed from spikeinterface.sorters.external.mountainsort5.py
        """Parameters for MountainSort sorting scheme 1
        """
        scheme1_sorting_parameters = dict(
            detect_threshold = detect_threshold, # the threshold for detection of whitened data
            detect_channel_radius = radius_um, # the radius (in units of channel locations) for exluding nearby channels from detection
            detect_time_radius_msec = 0.5, # Default 0.5 the radius (in msec) for excluding nearby events from detection
            detect_sign = detect_sign, # the sign of the threshold for detection (1, -1, or 0)
            snippet_T1 = snippet_T1, # the number of timepoints before the event to include in the snippet
            snippet_T2 = snippet_T2, # the number of timepoints after the event to include in the snippet
            snippet_mask_radius = radius_um, # the radius (in units of channel locations) for making a snippet around the central channel
            npca_per_channel = int(3), # default 3 the number of PCA components per channel for initial dimension reduction
            npca_per_subdivision = int(10)  # default 10 the number of PCA components to compute for each subdivision of clustering
        )
        
        """Parameters for MountainSort sorting scheme 2
        See Scheme1SortingParameters for more details on the parameters below.
        """
        scheme2_sorting_parameters = dict(
            phase1_detect_channel_radius = radius_um, # detect_channel_radius in phase 1
            detect_channel_radius = scheme2_detect_channel_radius, # detect_channel_radius in phase 2
            phase1_detect_threshold = detect_threshold-0.1, # detect_threshold in phase 1
            phase1_detect_time_radius_msec = 1.5, # Default 1.5 detect_time_radius_msec in phase 1
            detect_time_radius_msec = 0.5, # Default 0.5 detect_time_radius_msec in phase 2 NOTE: This is a parameter that SI does not allow to change
            phase1_npca_per_channel = int(3), # default 3 npca_per_channel in phase 1
            phase1_npca_per_subdivision = int(10),  # default 10 npca_per_subdivision in phase 1
            detect_sign = detect_sign, 
            detect_threshold = detect_threshold, # detect_threshold in phase 2
            snippet_T1 = snippet_T1,
            snippet_T2 = snippet_T2,
            snippet_mask_radius = radius_um,
            max_num_snippets_per_training_batch = int(200), # Default 200 the maximum number of snippets to use for training the classifier in each batch
            classifier_npca = None, # Default None. the number of principal components to use for each neighborhood classifier. If None (default), the number of principal components will be automatically determined as max(12, M * 3) where M is the number of channels in the neighborhood
            training_duration_sec = scheme2_training_duration_sec,  # the duration of the training data (in seconds)
            training_recording_sampling_mode = 'uniform', # Default: uniform how to sample the training data. If 'initial', then the first training_duration_sec of the recording will be used. If 'uniform', then the training data will be sampled uniformly in 10-second chunks from the recording
            classification_chunk_sec = None # Default : None the duration of each chunk of data to use for classification (in seconds) If None, it will try to use the entire recording
        )
    
        """Parameters for MountainSort sorting scheme 3
        - block_sorting_parameters: Scheme2SortingParameters for individual blocks
        - block_duration_sec: duration of each block
        """

        scheme3_sorting_parameters = copy.deepcopy(scheme2_sorting_parameters)
        scheme3_sorting_parameters.update(dict(
            block_duration_sec=scheme3_block_duration_sec # duration of each block
            ))
        
        sorter_info = dict(
                sorter_name = 'mountainsort5',
                sorter_label = 'MS5s{}ch{}th{}{}'.format(mountainsort5_scheme_default, nearest_chans, sign_label, detect_threshold).replace('.', '_'),
                useDocker = False, 
                params2update = {
                        'scheme': mountainsort5_scheme_default, # 2 = Not searching for templates in chunks, 3 = it will implement an approach to compensate drifting
                        'detect_threshold': detect_threshold, # Default: 5.5
                        'detect_sign': detect_sign, # Default: -1
                        'detect_time_radius_msec': 0.5, # Default: 0,5
                        'snippet_T1': snippet_T1, # Default: 20
                        'snippet_T2': snippet_T2, # Default: 20
                        'npca_per_channel': int(3), # Default: 3
                        'npca_per_subdivision': int(10), # Default: 10
                        'snippet_mask_radius': radius_um,
                        'scheme1_detect_channel_radius': radius_um, # Default: 150
                        'scheme2_phase1_detect_channel_radius': radius_um, # Default: 200
                        'scheme2_detect_channel_radius': scheme2_detect_channel_radius, # Default: 50
                        'scheme2_max_num_snippets_per_training_batch': int(200), # Default: 200
                        'scheme2_training_duration_sec':  scheme2_training_duration_sec, # Default: 60 * 5
                        'scheme2_training_recording_sampling_mode': 'uniform', # Default: "uniform"
                        'scheme3_block_duration_sec': scheme3_block_duration_sec, # Default: 60 * 30
                        'filter': False, # Default: True
                        'whiten': sorter_whitening,  # Default: True
                },
                scheme1_sorting_parameters = scheme1_sorting_parameters,
                scheme2_sorting_parameters = scheme2_sorting_parameters,
                scheme3_sorting_parameters = scheme3_sorting_parameters
        )

    elif sorterName =='kilosort4':
        # see:
        # https://kilosort.readthedocs.io/en/latest/parameters.html
        if detect_threshold is None:
            detect_threshold = 3  # Whitened Threshold (STD from Absolute traces)
        sorter_info= dict(
            sorter_name = 'kilosort4', 
            sorter_label = 'KS4ch{}th{}{}'.format(nearest_chans, sign_label, detect_threshold).replace('.', '_'),
            useDocker = False,
            params2update = {
                    'Th_universal': 8, # Default 9 # To detect more units overall, it may help to reduce Th_universal.
                    'Th_learned': 9, # Default 8 # If few spikes are detected, or if you see neurons disappearing and reappearing over time when viewing results in Phy, it may help to decrease Th_learned
                    'Th_single_ch': detect_threshold, # Whitened Threshold (STD from Absolute traces)
                    'nt': nt, # Default 61 for 30K
                    'nt0min': snippet_T1, # default 20 for 30K
                    'dmin': step_chan, # A good value for dmin based on the median distance between contacts, Default None
                    'dminx': 1, #1, # Default 32 for Neuropixels (two columns ar at 32 microns). Try setting this to the median lateral distance between contacts to start
                    'min_template_size': 10, # Default 10  Standard deviation of the smallest, spatial envelope Gaussian use for universal templates for Neuropixels ([X-horizontal = 32, Y-vertical = 15|20] channel spacing in microns)
                    'nearest_chans': nearest_chans, # Number of nearest channels to consider when finding local maxima during spike detection. Default: 10
                    'max_channel_distance': step_chan, # Templates farther away than this from their nearest channel (i.e., the closest channel assigned to the template) will not be used. Default None = max(dmin, dminx)
                    'nearest_templates': min(100, nChans), # Default 100, less than or equal to the number of channels helps avoid numerical instability
                    'x_centers': 1, # Number of x-positions to use when determining center points for template groupings, Default = None
                    'do_CAR': False,
                    'do_correction': False,
                    'skip_kilosort_preprocessing': sorter_whitening==False,
                    'whitening_range': nearest_chans,
                    'keep_good_only': True, 
                    'save_extra_vars': True,
                    'torch_device': 'cuda'
            }
        )
    
    else:
        raise Exception('Setting parameters to run sorter: "{}" is not supported by this pipeline version\nSupported sorters are:\n\t{}\n\t{}\n\t{}'.format(sorterName, 'waveclus', 'mountainsort5', 'kilosort4'))
    
    return dict(
            sorter_info =  sorter_info,
            estimate_sparsity_params = estimate_sparsity_params,
            sorting_analyzer_params = sorting_analyzer_params
        )

####################################################################################################################################################################################
# LOCAL FUNCTION TO RUM MOUNTAINSORT5
####################################################################################################################################################################################
def _run_mountainsort5(folder_sorter, recording, sorter_info):

    # Method copyed from spikeinterface.sorters.external.mountainsort5.py

    # Ensure is a clean sorter folder 
    print('\nruning "mountainsort5" with local function.....\n')
    shutil.rmtree(folder_sorter, ignore_errors=True)

    if not os.path.isdir(folder_sorter):
        os.makedirs(folder_sorter)

    sorter_output =  os.path.join(folder_sorter, "sorter_output")
    os.mkdir(sorter_output)

    rec_file = os.path.join(folder_sorter, "spikeinterface_recording.json")
    if recording.check_serializability("json"):
        recording.dump(rec_file)
    elif recording.check_serializability("pickle"):
        recording.dump(os.path.join(folder_sorter, "spikeinterface_recording.pickle"), relative_to=folder_sorter)
    
    del rec_file

    # Method copyed from spikeinterface.sorters.external.mountainsort5.py
    delete_temporary_recording = False

    if not recording.is_binary_compatible():
        recording_cached = recording.save_to_folder(folder=os.path.join(sorter_output, "recording"), **job_kwargs)
        delete_temporary_recording = True
    else:
        recording_cached = recording

    now = datetime.datetime.now()
    log = {
        "sorter_name": 'mountainsort5',
        "sorter_version": str(ms5.__version__),
        "datetime": now,
        "runtime_trace": [],
    }
    t0 = time.perf_counter()

    sorter_params = {'scheme': sorter_info['params2update']['scheme']}

    if sorter_info['params2update']['scheme'] == "1":
        sorting = ms5.sorting_scheme1(
                        recording=recording_cached, 
                        sorting_parameters=ms5.Scheme1SortingParameters(**sorter_info['scheme1_sorting_parameters'])
                    )
        sorter_params.update(sorter_info['scheme1_sorting_parameters'])

    elif sorter_info['params2update']['scheme'] == "2":
        sorting = ms5.sorting_scheme2(
                        recording=recording_cached, 
                        sorting_parameters=ms5.Scheme2SortingParameters(**sorter_info['scheme2_sorting_parameters'])
                    )
        sorter_params.update(sorter_info['scheme2_sorting_parameters'])

    elif sorter_info['params2update']['scheme'] == "3":
        sorting = ms5.sorting_scheme3(
                        recording=recording_cached, 
                        sorting_parameters= ms5.Scheme3SortingParameters(
                            block_sorting_parameters = ms5.Scheme2SortingParameters(**sorter_info['scheme2_sorting_parameters']),
                            block_duration_sec = sorter_info['scheme3_sorting_parameters']['block_duration_sec']
                            )
                    )
        sorter_params.update(sorter_info['scheme3_sorting_parameters'])
    else:
        raise ValueError(f"Invalid scheme: {sorter_info['params2update']['scheme']} given. scheme must be one of '1', '2' or '3'")
    
    t1 = time.perf_counter()

    del recording_cached

    if delete_temporary_recording:
        shutil.rmtree(os.path.join(sorter_output, "recording"), ignore_errors=True)
        if os.path.isdir(os.path.join(sorter_output, "recording")):
            warn("temporal sorting-recording cleanup failed, please remove file yourself if desired")

    NpzSortingExtractor.write_sorting(sorting, os.path.join(sorter_output, "firings.npz"))

    del sorting

    with open(os.path.join(folder_sorter, "spikeinterface_params.json"), mode="w", encoding="utf8") as f:
        all_params = dict()
        all_params["sorter_name"] = 'mountainsort5'
        all_params["sorter_params"] = sorter_params
        all_params["sorter_params"].update({
            "freq_min": 500,
            "freq_max": 6000,
            "filter": False,
            "whiten": False, 
            "delete_temporary_recording": True,
        })
        json.dump(check_json(all_params), f, indent=4)
        del all_params
    
    log["error"] = False
    log["run_time"] = float(t1 - t0)
    log["runtime_trace"] = []

    # dump to json
    with open(os.path.join(folder_sorter, "spikeinterface_log.json"), mode="w", encoding="utf8") as f:
        json.dump(check_json(log), f, indent=4)

    del now, log, t0, t1, sorter_params



####################################################################################################################################################################################
# FUNCTION TO DO DEFAULT POSTPROCESSING AND CREATE THE FINAL SORTING OBJECT
####################################################################################################################################################################################
def create_postprocessed_sorting(sorting_folder, si_recording_loaded, sorting_analyzer_params, alingSorting = True, job_kwargs=job_kwargs):

    # "sorter_detect_sign" is used by all the sorting_analyzer params:
    peak_sign = sorting_analyzer_params['template_metrics']['peak_sign']

    #######################################################################################################################
    # Load Original Sorting object
    #######################################################################################################################
    sorting_loaded = si.load_extractor(sorting_folder)

    #######################################################################################################################
    #                                    DO SOME PRELIMINAR POSTPROCESSING
    #######################################################################################################################

    # Returns a new sorting object which contains only units with at least one spike.
    print('\nRemoving empty clusters:......\n')
    sorting = sorting_loaded.remove_empty_units() 
    sorting.save_to_folder(folder= sorting_folder + '1',  overwrite=True, **job_kwargs)

    ######################################################################
    # explicitly try to close the spikes.npy 
    try:
        sorting_loaded.spikes._mmap.close()
    except:
        pass
    del sorting_loaded, sorting

    # Excess spikes are the ones exceeding a recording number of samples, for each segment
    print('\nremoving excess spikes:......\n')
    sorting_loaded = si.load_extractor(sorting_folder + '1')
    sorting = scur.remove_excess_spikes(sorting_loaded, si_recording_loaded) 
    sorting.save_to_folder(folder=sorting_folder + '2',  overwrite=True, **job_kwargs)
    del sorting_loaded, sorting 
    shutil.rmtree(sorting_folder + '1', ignore_errors=True)

    # Spikes are considered duplicated if they are less than x ms apart where x is the censored period.
    print('\nRemoving duplicated spikes:......\n')
    sorting_loaded = si.load_extractor(sorting_folder + '2')
    sorting = scur.remove_duplicated_spikes(sorting_loaded, censored_period_ms=0.3, method="keep_first_iterative") 
    sorting.save_to_folder(folder=sorting_folder + '3',  overwrite=True, **job_kwargs)
    del sorting_loaded, sorting
    shutil.rmtree(sorting_folder + '2', ignore_errors=True)
    
    
    if alingSorting:

        print('\nAligning Templates:......\n')
        # Align sorter: In some situations spike sorters could return a spike index with a small shift related to the waveform peak
        # See : https://spikeinterface.readthedocs.io/en/latest/api.html#spikeinterface.core.get_template_extremum_channel_peak_shift

        analyzer_folder_to_align = sorting_folder + '_analyzer2align'

        sorting_loaded = si.load_extractor(sorting_folder + '3')

        sorting_analyzer_to_align = si.create_sorting_analyzer(
            sorting=sorting_loaded, 
            recording=si_recording_loaded, 
            format='binary_folder', 
            folder=analyzer_folder_to_align, 
            overwrite=True,
            sparse=False,
            **job_kwargs
            )

        del sorting_analyzer_to_align

        for ext_name in ['random_spikes', 'waveforms', 'templates']:
            ext_params = sorting_analyzer_params[ext_name]
            print('computing "{}" with params:'.format(ext_name))
            for k, v in ext_params.items():
                print('\t"{}" : {}'.format(k, v))

            ext_params.update(job_kwargs)

            sorting_analyzer_to_align = si.load_sorting_analyzer(folder=analyzer_folder_to_align, load_extensions=True)
            sorting_analyzer_to_align.compute_one_extension(ext_name, save=True, verbose=True, **ext_params)
            
            del sorting_analyzer_to_align

        sorting_analyzer_to_align = si.load_sorting_analyzer(folder=analyzer_folder_to_align, load_extensions=True)

        unit_peak_shifts = get_template_extremum_channel_peak_shift(sorting_analyzer_to_align, peak_sign=peak_sign)

        for unit, shift_val in unit_peak_shifts.items():
            if shift_val !=0:
                print('\nUsing peakSign = {}. Template of "si_unit_id-{}" has a peak shift of: {}'.format(peak_sign, unit, shift_val))

        sorting_aligned = align_sorting(sorting_loaded, unit_peak_shifts)

        sorting_aligned.save_to_folder(folder=sorting_folder + '_aligned',  overwrite=True, **job_kwargs)
        
        del sorting_analyzer_to_align, sorting_loaded, sorting_aligned

        # Delete sorting_analyzer_to_align folder
        shutil.rmtree(analyzer_folder_to_align, ignore_errors=True)
        shutil.rmtree(sorting_folder + '3', ignore_errors=True)

        sorting_folder_aligned = sorting_folder + '_aligned'
    
    else:

        sorting_folder_aligned = sorting_folder + '3'

    print('\nRemoving duplicate clusters:......\n')

    sorting_loaded = si.load_extractor(sorting_folder_aligned)

    # Removes redundant or duplicate units by comparing the sorting output with itself
    sorting_clean, redundant_unit_pairs = scur.remove_redundant_units(sorting_loaded, 
                                                            align=False, # 
                                                            remove_strategy="max_spikes", 
                                                            peak_sign=peak_sign, 
                                                            unit_peak_shifts=None,
                                                            extra_outputs=True
                                                            )
    
    del sorting_loaded
    
    for u1, u2 in redundant_unit_pairs:
        print('\nunit = {} was found to be redundant with unit = {}'.format(u1, u2))

    sorting_clean.save_to_folder(folder=sorting_folder + '_postProcessed',  overwrite=True, **job_kwargs)

    del sorting_clean
    shutil.rmtree(sorting_folder_aligned, ignore_errors=True)


    return os.path.abspath(sorting_folder + '_postProcessed')


####################################################################################################################################################################################
# FUNCTION TO CREATE A SORTING ANALYZER
####################################################################################################################################################################################
def createFULL_sortingAnalyzer_folder(sorting_loaded, si_recording_loaded, folder_analyzer, estimate_sparsity_params, sorting_analyzer_params, job_kwargs=job_kwargs):

    peak_sign = sorting_analyzer_params['template_metrics']['peak_sign']
        
    print('\nCreating Sorting Analyzer......\n')    
    
    ###############################################################################################################################
    estimate_sparisity = False
    if si_recording_loaded.get_num_channels() >= maxChans_withoutSparsity_default:
        estimate_sparisity =True

    sorting_analyzer = si.create_sorting_analyzer(
        sorting=sorting_loaded, 
        recording=si_recording_loaded, 
        format='binary_folder', 
        folder=folder_analyzer, 
        overwrite=True,
        sparse=estimate_sparisity, 
        **estimate_sparsity_params,
        **job_kwargs
        )
    
    del sorting_analyzer

    ###############################################################################################################################
    print('\nComputing extensions......\n')

    si.set_global_job_kwargs(**job_kwargs)

    for ext_name, ext_params in sorting_analyzer_params.items():

        print('computing "{}" with params:'.format(ext_name))
        for k, v in ext_params.items():
            print('\t"{}" : {}'.format(k, v))

        ext_params.update(job_kwargs)

        sorting_analyzer = si.load_sorting_analyzer(folder=folder_analyzer, load_extensions=True)
        sorting_analyzer.compute_one_extension(ext_name, save=True, verbose=True, **ext_params)
        
        del sorting_analyzer

        print('{} DONE¡¡\n\n'.format(ext_name))
    
    ###############################################################################################################################
    print('\nComputing quality metrics......\n')

    sorting_analyzer = si.load_sorting_analyzer(folder=folder_analyzer, load_extensions=True)

    sqm.compute_quality_metrics(sorting_analyzer, save=True, skip_pc_metrics=False, peak_sign=peak_sign, **job_kwargs)

    print('\nSorting Analyzer DONE¡¡......\n')

    del sorting_analyzer

####################################################################################################################################################################################
# FUNCTION TO TEST IF THE SORTING ANALYZER FOLDER MATCHES SPARSITY PROPERTY AND EXTENSION NAMES
####################################################################################################################################################################################
def validate_sortingAnalyzer(sorting_analyzer, sorting_analyzer_params, si_recording_loaded=None):

    validAnalyzer = True

    if si_recording_loaded is None:
        if sorting_analyzer.has_recording():
            num_channels = sorting_analyzer.recording.get_num_channels()
        else:
            raise Exception('sorting analyzer must contain recording, otherwise provide the recording object as an input to "si_recording_loaded"')
    else:
        num_channels = si_recording_loaded.get_num_channels()
    
    ###############################################################################################################################
    # Check for sparsity
    print('Checking sparsity......')
    has_to_be_sparse = False
    if  num_channels>= maxChans_withoutSparsity_default:
        has_to_be_sparse =True
    
    if has_to_be_sparse!=sorting_analyzer.is_sparse():
        validAnalyzer = False
        print('\nSorting Analyzer sparsity must be {} and it is {}\n'.format(has_to_be_sparse, sorting_analyzer.is_sparse()))

    ###############################################################################################################################
    print('Checking extensions......')
    for ext_name in sorting_analyzer_params.keys():
        if not sorting_analyzer.has_extension(ext_name):
            validAnalyzer = False
            print('\nSorting Analyzer must have extension {}\ncurrent analyzer extensions: {}\n'.format(ext_name, sorting_analyzer.get_saved_extension_names()))

    if validAnalyzer:
        print('Sorting Analyzer is VALID¡¡......\n')

    return validAnalyzer


    

####################################################################################################################################################################################
#                                                              MAIN FUNCTION TO RUN PREPRO:
####################################################################################################################################################################################
#   1) Concatenate & Attach Probe (save Plots)
#   2) Filtering
#   3) Detect Bad Channels
#   4) CommonMedianReference (CMR)
#   5) PowerDensitySpectrum (PSD) (save Plots)
#   6) Peaks Locations | Amplitudes (save plots)
#
# If the number of channels is lower than "nearest_chans" it will NOT run MOTION steps
#
#   7) Motion Estimation (save Plots)
#   8) Motion Interpolation (save Plots: new PeaksLocations)
####################################################################################################################################################################################
def run_prepro(parentRecordingFolder, parentPreproSortingFolder, sessionYear, sessionMonth, sessionDay, electrodeGroup_Name=None, session_index=None, electrodeGroup_sessions=None,
        local_radius_chans = (exclude_radius_chans_default, include_radius_chans_default), 
        noisy_freq = noisy_freq_default, 
        ms_before = ms_before_default, 
        ms_after = ms_after_default, 
        peak_sign = peak_sign_default,
        nearest_chans = nearest_chans_default, 
        peak_detect_threshold = peak_detect_threshold_default, 
        do_detect_bad_channels = True,
        do_motion = True,
        motion_rigid = motion_rigid_default, 
        motion_options = motion_options_default, 
        interpolate_options = interpolate_options_default,
        localProcess_NWB = False,
        rewrite_prepro = True,
        return_recording = False     
    ):    
    
    if electrodeGroup_sessions is None:

        if electrodeGroup_Name is None:
            raise Exception('Missing "electrodeGroupe_Name" (based on NWBfile naming of the "acquisition" container example: "raw-C1-257-288")\nNote: if there were different coordinates within the same day, you need to add "-sub1", "-sub2", etc.')
        
        electrodeGroups = getUnique_electrodeGroups(parentRecordingFolder, sessionYear, sessionMonth, sessionDay)
        electrodeGroup_sessions = select_electrodeGroup_and_session_info(electrodeGroups, sessionYear, sessionMonth, sessionDay, electrodeGroup_Name, session_index=session_index)

    #######################################################################################################################
    # Get/Create FOLDER paths to save Figures
    #######################################################################################################################
    # Create General Session Folder to save Figures
    sessionFolder = os.path.join(os.path.abspath(parentPreproSortingFolder), 'exp{}-{:02d}-{:02d}_sorting'.format(sessionYear, sessionMonth, sessionDay))
    if not os.path.isdir(sessionFolder):
        os.makedirs(sessionFolder)
    
    #############################################################################
    # Create Folder to save ElectrodeGroup-Session Preprocessing
    elecGroupSessName = '{}-{}'.format(electrodeGroup_sessions['sessID'], electrodeGroup_sessions['electrodeName'])
    elecGroupSessFolder = os.path.join(sessionFolder, elecGroupSessName)
    if not os.path.isdir(elecGroupSessFolder):
        os.mkdir(elecGroupSessFolder)

    # Create & clear SI temporal folder (it will be used to save peaks, peaks locations, and NWB files if localProcess_NWB=True)
    folder_temporal = get_tempdir(processName='{}-{}'.format(processName_prefix, elecGroupSessName), resetDir=rewrite_prepro)
    

    #############################################################################################
    # Check if preprocessed Recording exists
    if os.path.isfile(os.path.join(elecGroupSessFolder, elecGroupSessName  + '_SIrecording.pkl')) and not rewrite_prepro:  

        print('\nPreprocessed recording was found... \nelectrodeGroup Session : {}\n'.format(elecGroupSessName))

        si_recording = si.load_extractor(os.path.join(elecGroupSessFolder, elecGroupSessName  + '_SIrecording.pkl'))

        # Ensure "elecGroupSessName" & "recording_sufix" & "savePath" annotations exists
        # It is probable that previous versions did not have these annotations
        update_recording = False
        if 'elecGroupSessName' not in si_recording._annotations:
            si_recording.annotate(elecGroupSessName=elecGroupSessName)
            print('Updating annotations {} to the recording...'.format('elecGroupSessName'))
            update_recording = True
        if 'recording_sufix' not in si_recording._annotations:
            si_recording.annotate(recording_sufix='')
            print('Updating annotations {} to the recording...'.format('recording_sufix'))
            update_recording = True
        if 'savePath' not in si_recording._annotations:
            si_recording.annotate(savePath=elecGroupSessFolder)
            print('Updating annotations {} to the recording...'.format('savePath'))
            update_recording = True
        
        if update_recording:
            si_recording.dump_to_pickle(os.path.join(elecGroupSessFolder, elecGroupSessName  + '_SIrecording'))
            del si_recording
            si_recording = si.load_extractor(os.path.join(elecGroupSessFolder, elecGroupSessName  + '_SIrecording.pkl'))

        step_chan = si_recording.get_annotation('y_contacts_distance_um')

        ###########################################################
        #   DETECT PEAKS options: 
        peaks_options = {
            'method': "locally_exclusive",
            'peak_sign': peak_sign, 
            'detect_threshold': peak_detect_threshold, #  MAD: Median Amplitude Deviations
            'radius_um': step_chan*nearest_chans,
        }

        ###########################################################
        #   LOCALIZE PEAKS options:
        locations_options = {     
            'ms_before': ms_before,
            'ms_after': ms_after,
            'location_method': 'monopolar_triangulation', # Paninski Lab
            'location_kwargs': {'max_distance_um': step_chan*(nearest_chans + 1), 'optimizer': 'least_square'}
        }

        # given that the recording was found, it is assume that the original NWB path was preserved.
        keepOriginal_NWBpath = True
    
    else:

        print('\nPreprocessing electrodeGroup Session : {}\n'.format(elecGroupSessName))

        si_recording_ordered, electrodeGroup_sessions, keepOriginal_NWBpath = get_si_recording(parentRecordingFolder, sessionYear, sessionMonth, sessionDay, electrodeGroup_Name, 
                                                                         session_index=session_index, 
                                                                         electrodeGroup_sessions=electrodeGroup_sessions, 
                                                                         localProcess_NWB=localProcess_NWB, 
                                                                         folder_temporal=folder_temporal, 
                                                                         reset_tempdir=True)

        ###########################################################
        # Save updated ElectrodeGroupInformation
        if keepOriginal_NWBpath:
            pickle.dump(electrodeGroup_sessions, open(os.path.join(elecGroupSessFolder, elecGroupSessName + '_electrodeGroupInfo.pkl'), 'wb' ))

        #################################################################################################################
        #                                            PREPROCESSING
        #################################################################################################################

        step_chan = si_recording_ordered.get_annotation('y_contacts_distance_um')

        #########################################################################################################
        # If there is noise at a specefic frequency, remove it with a notch filter
        if noisy_freq is not None:
            si_recording_denoise = spre.notch_filter(si_recording_ordered, freq=noisy_freq, q=10)
        else:
            si_recording_denoise = si_recording_ordered
        
        del si_recording_ordered
        
        ##########################################################
        # High-pass filter
        si_recording_filter = spre.bandpass_filter(recording=si_recording_denoise, freq_min=500) 

        ##########################################################
        # Median zero-center signal (per channel basis)
        si_recording_center_pre = spre.center(recording=si_recording_filter, mode='median')

        ##########################################################
        # Apply Common Median Reference
        if si_recording_denoise.get_num_channels()>1:
            si_recording_cmr_pre = spre.common_reference(recording=si_recording_center_pre, reference='local', operator='median', local_radius=(local_radius_chans[0] * step_chan, local_radius_chans[1] * step_chan ))
        else:
            si_recording_cmr_pre = si_recording_center_pre

        ##########################################################
        # Plot session's concatenation
        print('\nPlotting {} concatenations from electrodeGroup: {}\n'.format(si_recording_denoise.get_annotation('nConcatenations'), elecGroupSessName))
        plot_concatenations(si_recording_dict={'Raw': si_recording_denoise, 'Filter': si_recording_filter, 'CMR': si_recording_cmr_pre}, 
                                    plot_windows_secs=0.005, sampleChans=True, showPlots=False, savePlots=True, folderPlots=elecGroupSessFolder)

        del si_recording_filter, si_recording_cmr_pre

        #############################################################################################################
        #                       Detect bad channels
        #############################################################################################################

        if do_detect_bad_channels:
            # Use recObj centered (before CMR)
            print('\nDetecting Bad Channels ....... \n')
            bad_channel_ids, channel_labels = spre.detect_bad_channels(recording=si_recording_center_pre, method="coherence+psd")

            print('Channel labels : \n\t', channel_labels)
            print('\n{} bad channels detected\n\tBad channel Index: {}'.format(len(bad_channel_ids), bad_channel_ids))

            ##########################################################
            # Remove bad channels 
            # add channel_labels:
            # 'coeherence+psd' : good/dead/noise/out 
            # 'std', 'mad', 'neighborhood_r2' : good/noise
            si_recording_center_pre.set_property(key='channel_labels', values=channel_labels)
            si_recording_center = si_recording_center_pre.remove_channels(remove_channel_ids=bad_channel_ids)

            del si_recording_center_pre, bad_channel_ids

        else:
            warn('Skipping Detection of bad channels¡¡......\nALL CHANNELS WILL BE LABELED AS "GOOD"')
            si_recording_center = si_recording_center_pre.clone()

            channel_labels = ['good']*si_recording_center.get_num_channels()
            si_recording_center.set_property(key='channel_labels', values=channel_labels)

            del si_recording_center_pre
        
        si_recording_center.set_annotation(annotation_key='detect_bad_channels', value=do_detect_bad_channels)

        ######################################################################################################
        # PLOT THE PROBE & channel labels
        print('\nPlotting PROBE from electrodeGroup: {}\n'.format(elecGroupSessName))

        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 8))
        ax[0].set_rasterized(True)
        plot_probe(probe=si_recording_denoise.get_probe(), ax = ax[0], with_contact_id=False, with_device_index=True)

        y_probe = si_recording_denoise.get_channel_locations(axes='y').flatten()
        ch_indx = si_recording_denoise.ids_to_indices()
        contact_ids = si_recording_denoise.get_channel_ids()
        for ch in ch_indx:
            ax[1].plot(0, y_probe[ch], marker="s", markeredgecolor=(0, 0, 1, 1), markerfacecolor=(0, 0, 1, 0.5), markeredgewidth = 1, markersize=6)
            ax[1].text(-0.5, y_probe[ch], 'ch={}'.format(contact_ids[ch]), horizontalalignment='left', color=(0, 0, 0, 1), fontsize=8)
            if channel_labels[ch]=='good':
                colorT = (0, 0, 1, 1)
                fweight = 'normal'
            else:
                colorT = (1, 0, 0, 1)
                fweight = 'bold'
            ax[1].text(0.5, y_probe[ch], channel_labels[ch], horizontalalignment='right', color=colorT, fontsize=8, fontweight=fweight)
        ax[1].set_xlim(-1, 1)
        ax[1].set_rasterized(True)
        fig.savefig(os.path.join(elecGroupSessFolder, elecGroupSessName + '_probe.eps'), dpi='figure', format='eps')
        plt.close(fig=fig)

        del channel_labels, si_recording_denoise

        #####################################################################
        # PLOT Power Spectrum Density 
        print('\nPlotting PowerDensitySpectrums from electrodeGroup: {}\n'.format(elecGroupSessName))
        plotPSD_randomChunks(si_recording_center, compare_CMR=True, plot_by_channel=True, chan_radius=local_radius_chans, showPlots=False, savePlots=True, folderPlots=elecGroupSessFolder)

        #############################################################################################################
        #  Reference after removing bad channels
        #############################################################################################################
        if si_recording_center.get_num_channels()>1:
            si_recording = spre.common_reference(recording=si_recording_center, reference='local', operator='median', local_radius=(local_radius_chans[0] * step_chan, local_radius_chans[1] * step_chan ))
        else:
            si_recording = si_recording_center

        ###################################################
        # Add some annotations:
        si_recording.annotate(is_filtered=True)
        si_recording.annotate(is_centered=True)
        si_recording.annotate(centered_mode = 'median')
        si_recording.annotate(is_referenced=True)    
        si_recording.annotate(reference = 'local')
        si_recording.annotate(reference_mode = 'median')

        del si_recording_center

        ####################################################################################################################################
        #  If NWB files were kept in the original location: Save Recording Object (LAZY without traces)
        if keepOriginal_NWBpath:
            # Add "savePath" annotation
            si_recording.set_annotation(annotation_key='savePath', value=elecGroupSessFolder, overwrite=True)
            si_recording.dump_to_pickle(os.path.join(elecGroupSessFolder, elecGroupSessName  + '_SIrecording'))
            del si_recording
            si_recording = si.load_extractor(os.path.join(elecGroupSessFolder, elecGroupSessName + '_SIrecording.pkl'))

        ########################################################################################################################
        #                                     PEAKS LOCATIONS AS A FUNCTION OF TIME
        ########################################################################################################################

        ###########################################################
        #   DETECT PEAKS options: 
        peaks_options = {
            'method': "locally_exclusive",
            'peak_sign': peak_sign, 
            'detect_threshold': peak_detect_threshold, #  MAD: Median Amplitude Deviations
            'radius_um': step_chan*nearest_chans,
        }

        ###########################################################
        #   LOCALIZE PEAKS options:
        locations_options = {     
            'ms_before': ms_before,
            'ms_after': ms_after,
            'location_method': 'monopolar_triangulation', # Paninski Lab
            'location_kwargs': {'max_distance_um': step_chan*(nearest_chans + 1), 'optimizer': 'least_square'}
        }

        ###########################################################
        # PLOT PEAKS LOCATIONS AS A FUNCTION OF TIME
        print('\nPlotting Peaks Locations from electrodeGroup: {}\n'.format(elecGroupSessName))

        plot_peakLocations(
            si_recording = si_recording, 
            folderPeaks = folder_temporal, 
            peaks_options = peaks_options, 
            locations_options = locations_options, 
            rewrite = False, 
            locationsSubSampled = True,
            showPlots = False,
            savePlots = True,
            folderPlots = elecGroupSessFolder
        )

    ########################################################################################################################
    # NOTE:
    # IF CHANNEL COUNT IS LOWER THAN "nearest_chans" IT WILL NOT RUN MOTION CORRECTION
    if si_recording.get_num_channels()>=nearest_chans and do_motion:

        ########################################################################################################################
        #                              MOTION ESTIMATION & INTERPOLATION
        ########################################################################################################################

        #####################################################################
        # Get peaks labels:
        if "by_channel" in peaks_options['method']:
            peaks_prefix = 'byCh'
        elif "locally_exclusive" in peaks_options['method']:
            peaks_prefix = 'loc'
        else:
            raise Exception('Peaks Detection method "{}" not recognized\nAvailable options: {}'.format(peaks_options['method'], ["by_channel", "locally_exclusive", "locally_exclusive_cl", 
                                                                                                                                 "by_channel_torch", "locally_exclusive_torch", "matched_filtering"]))

        peaks_label = '{}{}{}'.format(peaks_prefix, peaks_options['peak_sign'][0].upper(), peaks_options['detect_threshold']).replace('.', '_')

        #####################################################################
        # Get location labels:
        if locations_options['location_method']=='center_of_mass':
            loc_label = 'mass'
        elif locations_options['location_method']=='monopolar_triangulation':
            loc_label = 'mono'
        elif locations_options['location_method']=='grid_convolution':
            loc_label = 'grid'
        else:
            raise Exception('Peaks Location method "{}" not recognized\nAvailable options: {}'.format(peaks_options['method'], ['center_of_mass', 'monopolar_triangulation', 'grid_convolution']))
        
        peaksLoc_label = peaks_label + '_' + loc_label

        #####################################################################
        # Get motion labels:
        if motion_rigid:
            rigid_label = 'rigid'
            win_step_um = step_chan/2
        else:
            rigid_label = 'noRigid'
            win_step_um = step_chan            
        
        if motion_options['method']=='decentralized':
            motion_label = 'deCENTRAL' + rigid_label
        elif motion_options['method']=='iterative_template':
            motion_label = 'iterTEMP' + rigid_label
        elif motion_options['method']=='dredge_ap':
            motion_label = 'dredgeAP' + rigid_label
        elif motion_options['method']=='dredge_lfp':
            motion_label = 'dredgeLFP' + rigid_label
        else:
            raise Exception('Motion Estimation method "{}" not recognized\nAvailable options: {}'.format(motion_options['method'], ['decentralized', 'iterative_template', 'dredge_ap', 'dredge_lfp']))

        
        #####################################################################
        # Get interpolation labels:
        if interpolate_options['method']=='idw':
            interpolation_label = 'idw'
        elif interpolate_options['method']=='nearest':
            interpolation_label = 'near'
        elif interpolate_options['method']=='kriging':
            interpolation_label = 'krig'
        else:
            raise Exception('Interpolation method "{}" not recognized\nAvailable options: {}'.format(interpolate_options['method'], ['kriging', 'idw', 'nearest']))
        
        if interpolate_options['border_mode']=="remove_channels":
            border_label = 'Rmv'
        elif interpolate_options['border_mode']== "force_extrapolate":
            border_label = 'Extrap'
        elif interpolate_options['border_mode']=="force_zeros":
            border_label = 'Zeros'
        else:
            raise Exception('BorderMode method "{}" not recognized\nAvailable options: {}'.format(interpolate_options['border_mode'], ['remove_channels', 'force_extrapolate', 'force_zeros']))
        
        motion_interpolation_label = motion_label + '_' + interpolation_label + border_label

        si_recording_sufix = '_' + peaksLoc_label + '_' + motion_interpolation_label

        if os.path.isfile(os.path.join(elecGroupSessFolder, elecGroupSessName + si_recording_sufix + '_SIrecording' + '.pkl')) and not rewrite_prepro:

            print('\nMotion corrected recording was found... \nelectrodeGroup Session : {}\nPeaks {}: sign = {}, detectionTH = {}\nLocation Method: {}\nMotion Interpolation label : {}\n'.format(elecGroupSessName,
                                                                                    peaks_options['method'], peaks_options['peak_sign'], peaks_options['detect_threshold'], locations_options['location_method'], motion_interpolation_label))

            si_recording_motion = si.load_extractor(os.path.join(elecGroupSessFolder, elecGroupSessName + si_recording_sufix + '_SIrecording.pkl'))

            # It is probable that previous versions did not have these annotations
            update_recording_motion = False
            if 'elecGroupSessName' not in si_recording_motion._annotations:
                si_recording_motion.annotate(elecGroupSessName=elecGroupSessName)
                print('Updating annotations {} to the motion corrected recording...'.format('elecGroupSessName'))
                update_recording_motion = True
            if 'recording_sufix' not in si_recording_motion._annotations:
                si_recording_motion.annotate(recording_sufix=si_recording_sufix)
                print('Updating annotations {} to the motion corrected recording...'.format('recording_sufix'))
                update_recording_motion = True
            if 'savePath' not in si_recording_motion._annotations:
                si_recording_motion.annotate(savePath=elecGroupSessFolder)
                print('Updating annotations {} to the motion corrected recording...'.format('savePath'))
                update_recording_motion = True

            if update_recording_motion:
                si_recording_motion.dump_to_pickle(os.path.join(elecGroupSessFolder, elecGroupSessName  + si_recording_sufix + '_SIrecording'))
                del si_recording_motion
                si_recording_motion = si.load_extractor(os.path.join(elecGroupSessFolder, elecGroupSessName  + si_recording_sufix + '_SIrecording.pkl'))

        else:

            ##################################################################
            #  Load PEAKS 
            if not os.path.exists(os.path.join(folder_temporal, 'peaks_' + peaks_label + '.npy')):

                print('\nGetting Peaks from electrodeGroup: {}\n'.format(elecGroupSessName))

                noise_levels = si.get_noise_levels(si_recording, return_scaled=False)
                
                peaks = detect_peaks(recording=si_recording, 
                    noise_levels = noise_levels,
                    method=peaks_options['method'], 
                    gather_mode = 'memory', # gather_mode= 'npy', # 'npy'
                    folder = None, # folder = eg_dirs['motion'],
                    names = None, # names = ['peaks'],
                    peak_sign=peaks_options['peak_sign'], 
                    detect_threshold=peaks_options['detect_threshold'], 
                    radius_um=peaks_options['radius_um'],
                    **job_kwargs)
                        
                numpy.save(os.path.join(folder_temporal, 'peaks_' + peaks_label + '.npy'), peaks)

                del peaks, noise_levels

            peaks = numpy.load(os.path.join(folder_temporal, 'peaks_' + peaks_label + '.npy'))

            ##################################################################
            # Load all PEAKS LOCATIONS 
            if not os.path.exists(os.path.join(folder_temporal, 'peaks_' + peaks_label + '_locations_' + loc_label + '.npy')):

                print('\nGetting Peaks Locations from electrodeGroup: {}\n'.format(elecGroupSessName))

                peaks_locations = localize_peaks(
                    recording=si_recording,
                    peaks=peaks,
                    ms_before= locations_options['ms_before'],
                    ms_after= locations_options['ms_after'],
                    radius_um = peaks_options['radius_um'],
                    method= locations_options['location_method'],
                    **locations_options['location_kwargs'],
                    **job_kwargs
                    )

                numpy.save(os.path.join(folder_temporal, 'peaks_' + peaks_label + '_locations_' + loc_label + '.npy'), peaks_locations)

                del peaks_locations   

            peaks_locations = numpy.load(os.path.join(folder_temporal, 'peaks_' + peaks_label + '_locations_' + loc_label + '.npy'))

            ###############################################################################
            # MOTION ESTIMATION
            print('\nEstimating MOTION from electrodeGroup: {}\n'.format(elecGroupSessName))

            motion, extra_check = estimate_motion(
                # Parameters common to all motion Methods
                recording=si_recording,
                peaks=peaks,
                peak_locations=peaks_locations,
                direction="y", # **histogram section**
                rigid=motion_rigid, # **non-rigid section** # default False
                win_shape="gaussian", # default "gaussian"
                win_step_um = win_step_um, # default 50.0
                win_scale_um = win_step_um*3.0, # default 150.0
                win_margin_um = None, # default -win_scale_um/2
                method = motion_options['method'], # **method options**
                extra_outputs=True, # **extra options** 
                progress_bar=True, 
                verbose=False,

                # Parameters common to all motion Methods but defined as **method_kwargs
                bin_um = step_chan/4, # default 10.0
                bin_s = 10.0, # default 10.0 # bin_s=10.0, # default 10.0

                # Specific Method arguments
                **motion_options['method_kwargs'],
            )

            contact_locations = si_recording.get_property('location')
            minY_loc = -50
            maxY_loc = max(contact_locations[:, 1])+50

            plot_motion_outputs(
                peaks = peaks, 
                peaks_locations = peaks_locations, 
                sampling_frequency = si_recording.sampling_frequency, 
                motion = motion, 
                extra_check = extra_check,  
                peaks_label = peaks_label, 
                loc_label = loc_label,
                motion_label = motion_label,
                minY_loc = minY_loc, 
                maxY_loc = maxY_loc, 
                folderPlots = elecGroupSessFolder, 
                prefixRec = elecGroupSessName, 
                showPlots = False, 
                savePlots = True, 
                concatenationTimes = si_recording.get_annotation('concatenationTimes'), 
                verbose = True
            )

            ####################################################################################################################################
            #                                           Interpolate motion
            ####################################################################################################################################

            ##################################################################
            print('motion interpolaton : ...' + peaks_label + '-' + loc_label + '-' + motion_interpolation_label + '....')

            si_recording_motion = interpolate_motion(
                recording = si_recording,
                motion = motion,
                border_mode = interpolate_options['border_mode'],
                spatial_interpolation_method = interpolate_options['method'],
                sigma_um = step_chan, # Used in the "kriging" formula
                p=1, # Used in the "kriging" formula
                num_closest=3, # Number of closest channels used by "idw" method for interpolation
            )
            
            si_recording_motion.annotate(motion_corrected = True)
            si_recording_motion.annotate(motion_method_label = si_recording_sufix[1:])
            si_recording_motion.set_annotation(annotation_key='recording_sufix', value=si_recording_sufix, overwrite=True)

            ####################################################################################################################################
            #  If NWB files were kept in the original location: Save Recording Object (LAZY without traces)
            if keepOriginal_NWBpath:
                si_recording_motion.set_annotation(annotation_key='savePath', value=elecGroupSessFolder, overwrite=True)
                si_recording_motion.dump_to_pickle(os.path.join(elecGroupSessFolder, elecGroupSessName + si_recording_sufix + '_SIrecording'))
                del si_recording_motion
                si_recording_motion = si.load_extractor(os.path.join(elecGroupSessFolder, elecGroupSessName  + si_recording_sufix + '_SIrecording.pkl'))

            ###########################################################
            # PLOT NEW PEAKS LOCATIONS AS A FUNCTION OF TIME
            print('\nPlotting Motion Corrected Peaks Locations from electrodeGroup: {}\n'.format(elecGroupSessName))

            plot_peakLocations(
                si_recording = si_recording_motion, 
                folderPeaks = folder_temporal,
                peaks_options = peaks_options, 
                locations_options = locations_options, 
                rewrite = False, 
                locationsSubSampled = True,
                showPlots = False,
                savePlots = True, 
                folderPlots = elecGroupSessFolder
            )

    if return_recording:
        if si_recording.get_num_channels()>=nearest_chans and do_motion:
            return si_recording_motion
        else:
            return si_recording
    else:
        clear_tempdir(tempFolderPath=folder_temporal, verbose=False)
        


####################################################################################################################################################################################
#                                                              MAIN FUNCTION TO RUN SORTING (& PREPRO):
####################################################################################################################################################################################
#  It will run a sorting method (Kilosort4 or Mountainsort5 or WaveClus) 
#  In case there is no preprocessed SI-recording object to retrieve, it will run all the preprocessed steps (see "run_prepro" function for details about preprocessing steps)
####################################################################################################################################################################################
def run_prepro_and_sorting(parentRecordingFolder, parentPreproSortingFolder, sessionYear, sessionMonth, sessionDay, electrodeGroup_Name, session_index=None, parentSortingFolder = None, 
        local_radius_chans = (exclude_radius_chans_default, include_radius_chans_default), 
        noisy_freq = noisy_freq_default, 
        ms_before = ms_before_default, 
        ms_after = ms_after_default, 
        peak_sign = peak_sign_default,
        nearest_chans = nearest_chans_default, 
        peak_detect_threshold = peak_detect_threshold_default, 
        do_detect_bad_channels = True,
        do_motion = True,
        motion_rigid = motion_rigid_default, 
        motion_options = motion_options_default, 
        interpolate_options = interpolate_options_default,
        localProcess_NWB = False,
        rewrite_prepro = False,
        run_sorting = True,
        sorterName = sorterName_default,
        sorter_detect_sign = peak_sign_default,
        sorter_detect_threshold = None,
        sorter_nearest_chans = None,
        saveSortingAnalyzer = False,
        postprocessing_alingSorting = True,
        export2phy = True, 
        export2phy_with_recording = True
    ):    
    
    electrodeGroups = getUnique_electrodeGroups(parentRecordingFolder, sessionYear, sessionMonth, sessionDay)
    electrodeGroup_sessions = select_electrodeGroup_and_session_info(electrodeGroups, sessionYear, sessionMonth, sessionDay, electrodeGroup_Name, session_index=session_index)

    si_recording = run_prepro(parentRecordingFolder, parentPreproSortingFolder, sessionYear, sessionMonth, sessionDay, electrodeGroup_sessions=electrodeGroup_sessions,
                        local_radius_chans = local_radius_chans, 
                        noisy_freq = noisy_freq, 
                        ms_before = ms_before, 
                        ms_after = ms_after, 
                        peak_sign = peak_sign,
                        nearest_chans = nearest_chans, 
                        peak_detect_threshold = peak_detect_threshold, 
                        do_detect_bad_channels = do_detect_bad_channels,
                        do_motion = do_motion,
                        motion_rigid = motion_rigid, 
                        motion_options = motion_options, 
                        interpolate_options = interpolate_options,
                        localProcess_NWB = localProcess_NWB,
                        rewrite_prepro = rewrite_prepro,
                        return_recording = True     
                    )
    
    # Get SI temporal folder (it will be used to save sorting results. If export2phy is True, and saveSortingAnalyzer is False, then temporal folder will be used for sorting_analyzer temporary
    elecGroupSessName = '{}-{}'.format(electrodeGroup_sessions['sessID'], electrodeGroup_sessions['electrodeName'])
    folder_temporal = get_tempdir(processName='{}-{}'.format(processName_prefix, elecGroupSessName), resetDir=False)
    
    if run_sorting:

        """
        ######################################################################################################################################################
        # Check for compatibility with sorter & number of channels:
        # mountainSort is recommended for single channel
        if si_recording.get_num_channels()==1 and sorterName =='kilosort4':
            continueVal = input('\nWARNING¡¡¡ The recording only contains 1 channel, it is recommended to run mountainsort5 instead of kilosort4\n\tDo you want to continue? (Y/N)')
            if continueVal.upper()=='N':
                raise Exception('Aborted by user, incompatibility between kilosort4 and the number of channels in the recording (n={})'.format(si_recording.get_num_channels()))
        """

        recording_sufix = si_recording.get_annotation('recording_sufix')

        if si_recording.get_num_channels()<=4 and sorterName =='kilosort4':
            print('\nWARNING¡¡¡\n\tThe recording contains 1 channel, it is recommended to run waveclus or mountainsort5 instead of kilosort4\n')

        if si_recording.get_num_channels()>4 and sorterName =='waveclus':
            print('\nWARNING¡¡¡\n\tThe recording contains {} channels.\n\tAbove 4 channels it is recommended to run mountainsort5 or kilosort4 instead of waveclus (it may crash while writing binary files)\n'.format(si_recording.get_num_channels()))
        
        ###########################################################################################################################################################
        # # "parentSortingFolder" : folder to save PHY & "recording" (binary) & Optional: "sorting_analyzer"
        #   if "parentSortingFolder" is not provided, then "parentPreproSortingFolder" will be used as a default.
        #
        # "saveSortingAnalyzer":
        #   If TRUE: "sorting_analyzer" will ALWAYS be SAVED (useful to debug OR to explore/compare "sorting_analyzer" vs "PHY")
        #   If FALSE: 
        #       if "export2phy" is TRUE and VALID (i.e., more than 2 units were found by the sorter):
        #                       "sorting_analyzer" will be created in a temporary folder and deleted after exporting to PHY.
        #       if "export2phy" is FALSE or "export2phy" is INVALID (i.e., only 1 unit was found by the sorter):
        #                       "sorting_analyzer" will be created in "parentSortingFolder"
        ###########################################################################################################################################################

        if parentSortingFolder is None:
            parentSortingFolder = parentPreproSortingFolder

        if not os.path.isdir(parentSortingFolder):
            os.makedirs(parentSortingFolder)

        #######################################################################################################################
        # Get/Create FOLDER paths to save binary RECORDING & (SORTING ANALYZER and/or PHY)
        # SORTER RESULTS WILL BE SAVED IN THE TEMPORAL PATH 
        #######################################################################################################################
        # Create/Confirm TEMPORAL Session Folder exists
        sessionSortingFolder = os.path.join(os.path.abspath(parentSortingFolder), 'exp{}-{:02d}-{:02d}_sorting'.format(sessionYear, sessionMonth, sessionDay))
        elecGroupSessSortingFolder = os.path.join(sessionSortingFolder, elecGroupSessName)
        if not os.path.isdir(elecGroupSessSortingFolder):  
            os.makedirs(elecGroupSessSortingFolder)

        ###############################################################################################################################
        # BinaryRecording will be saved in the parent Sorting folder. This binary recording will be exported to NWBprepro
        ###############################################################################################################################        
        si_recording_folder = os.path.join(elecGroupSessSortingFolder, elecGroupSessName + recording_sufix + '_binary')
        
        ###############################################################################################################################
        # Check if BynaryRecording already exists:
        if not os.path.exists(os.path.join(si_recording_folder, 'binary.json')) or not os.path.exists(os.path.join(si_recording_folder, 'si_folder.json')) or rewrite_prepro:
            si_recording.set_annotation(annotation_key='savePath', value=elecGroupSessSortingFolder, overwrite=True)
            print('prepraring to write Recording into binary......\n')
            si_recording.save_to_folder(folder=si_recording_folder, overwrite=True, **job_kwargs)

        del si_recording
        
        si_recording_loaded = si.load_extractor(si_recording_folder)

        print('reading Recording from binary successful¡......\n')

        ###############################################################################################################################
        # GET SORTING PARAMS
        ###############################################################################################################################
        nChans = si_recording_loaded.get_num_channels()
        step_chan = si_recording_loaded.get_annotation('y_contacts_distance_um')
        sampling_frequency = si_recording_loaded.sampling_frequency
        recording_total_duration = si_recording_loaded.get_total_duration()

        if sorter_nearest_chans is None:
            sorter_nearest_chans = nearest_chans

        sorter_and_sortingAnalyzer_params = get_sorter_and_sortingAnalyzer_params(sorterName, nChans, step_chan, sampling_frequency, recording_total_duration, ms_before, ms_after, sorter_detect_sign, sorter_nearest_chans, sorter_whitening=False, 
                                        detect_threshold=sorter_detect_threshold)
        
        sorter_info = sorter_and_sortingAnalyzer_params['sorter_info']
        estimate_sparsity_params = sorter_and_sortingAnalyzer_params['estimate_sparsity_params']
        sorting_analyzer_params = sorter_and_sortingAnalyzer_params['sorting_analyzer_params']

        sorter_folder_results = os.path.join(folder_temporal, elecGroupSessName + recording_sufix + '_' + sorter_info['sorter_label'] + '_results')
        sorting_folder = os.path.join(folder_temporal, elecGroupSessName + recording_sufix + '_' + sorter_info['sorter_label'] + '_sorting')

        sortingAnalyzer_folder = os.path.join(elecGroupSessSortingFolder, elecGroupSessName + recording_sufix + '_' + sorter_info['sorter_label'])
        sortingAnalyzer_folder_temporal = os.path.join(folder_temporal, elecGroupSessName + recording_sufix + '_' + sorter_info['sorter_label'])

        ###############################################################################################################################
        # WHITEN THE RECORDING
        ###############################################################################################################################
        # Check if Whiten BynaryRecording already exists:
        si_recording_to_sort_folder = os.path.join(folder_temporal, elecGroupSessName + recording_sufix + '_whiten_binary')

        if not os.path.exists(os.path.join(si_recording_to_sort_folder, 'binary.json')) or not os.path.exists(os.path.join(si_recording_to_sort_folder, 'si_folder.json')) or rewrite_prepro:
            
            # Do whitening
            recording_to_sort = spre.whiten(recording=si_recording_loaded, mode='local', radius_um = step_chan*3, dtype="float32")
            recording_to_sort.annotate(is_whitened=True) 

            # Save whitened recording
            recording_to_sort.set_annotation(annotation_key='savePath', value=folder_temporal, overwrite=True)
            print('prepraring to write Whitened Recording into binary......\n')
            recording_to_sort.save_to_folder(folder=si_recording_to_sort_folder, overwrite=True, **job_kwargs)
            del recording_to_sort
        
        recording_to_sort = si.load_extractor(si_recording_to_sort_folder)
        print('reading Whitened Recording from binary successful¡......\n')
        

        ###############################################################################################################################
        # RUN SORTING
        ###############################################################################################################################        
        if sorterName=='mountainsort5' and mountainsort5_run_local:
            _run_mountainsort5(sorter_folder_results, recording_to_sort, sorter_info)
        else:
            run_sorter(
                sorter_name = sorter_info['sorter_name'],
                recording = recording_to_sort,
                output_folder = sorter_folder_results,
                with_output = False,
                remove_existing_folder = True,
                delete_output_folder = False,
                docker_image = sorter_info['useDocker'],
                verbose = True,
                **sorter_info['params2update']
            )
            
        del recording_to_sort

        sorter_loaded = read_sorter_folder(sorter_folder_results, register_recording=False)
        sorter_loaded.save_to_folder(folder= sorting_folder,  overwrite=True, **job_kwargs)
        del sorter_loaded
        shutil.rmtree(sorter_folder_results, ignore_errors=True) 

        ###############################################################################################################################
        # DO SOME SORTING POSTPROCESSING
        ###############################################################################################################################
        sorting_postProcessed_folder = create_postprocessed_sorting(sorting_folder, si_recording_loaded, sorting_analyzer_params, alingSorting = postprocessing_alingSorting, job_kwargs=job_kwargs)
        sorting_loaded = si.load_extractor(sorting_postProcessed_folder)
        shutil.rmtree(sorting_folder, ignore_errors=True)

        ###############################################################################################################################
        # Create final SORTING ANALYZER
        # When only one unit is detected, PHY can not create/load features and it will not launch
        # Therefore, the sorting analyzer and the recording object should not be deleted
        # Ensure saving SORTING ANALYZER when only one unit was found:
        ###############################################################################################################################

        print('\n{} UNITS WERE FOUND ¡¡¡'.format(sorting_loaded.get_num_units()))

        if sorting_loaded.get_num_units()==0:

            #######################################################################################################################
            # If no units were found DELETE SORTING RESULTS
            ######################################################################
            # explicitly try to close the spikes.npy 
            try:
                sorting_loaded.spikes._mmap.close()
            except:
                pass
            del sorting_loaded, si_recording_loaded

            print('\nDeleting temporal sorting files .............\n')

            shutil.rmtree(sorting_postProcessed_folder, ignore_errors=True)
            print('\nDone¡ .............\n')
            
            raise Exception('NO UNITS WERE FOUND ¡¡¡')

        elif sorting_loaded.get_num_units()<2:

            #######################################################################################################################
            # When only 1 unit is founded, PHY will not launch, therefore ensure sorting analyzer is saved
            print('WARINING¡ Only ONE UNIT was found¡¡\nONLY Sorting analyzer will be save¡¡')

            export2phy = False
            saveSortingAnalyzer = True

        ###############################################################################################################################
        # Define sortingAnalyzer folder
        if saveSortingAnalyzer:               
            folder_analyzer = sortingAnalyzer_folder
        else:
            folder_analyzer = sortingAnalyzer_folder_temporal      

        createFULL_sortingAnalyzer_folder(sorting_loaded, si_recording_loaded, folder_analyzer, estimate_sparsity_params, sorting_analyzer_params, job_kwargs)
        
        ######################################################################
        # explicitly try to close the spikes.npy 
        try:
            sorting_loaded.spikes._mmap.close()
        except:
            pass
        del sorting_loaded
        shutil.rmtree(sorting_postProcessed_folder, ignore_errors=True)

        ###############################################################################################################################
        #                                    EXPORT TO PHY
        ###############################################################################################################################
        if export2phy:

            print('\nExporting Sorting Analyzer to PHY......\n')

            #############################################################################
            # Create/Confirm Parent Folder of the PHY folder
            if not os.path.isdir(elecGroupSessSortingFolder):  
                os.makedirs(elecGroupSessSortingFolder)
            
            # "export2phy_with_recording" 
            if export2phy_with_recording and si_recording_loaded.is_binary_compatible():
                print('PHY raw traces will be link to the recording path : {}'.format(si_recording_folder))
                export2phy_with_recording = False
                
            folder_analyzer_phy = os.path.join(elecGroupSessSortingFolder, elecGroupSessName + recording_sufix + '_' + sorter_info['sorter_label'] + '_phy')

            sorting_analyzer = si.load_sorting_analyzer(folder=folder_analyzer, load_extensions=True)

            export_to_phy(
                sorting_analyzer, 
                output_folder= folder_analyzer_phy,
                remove_if_exists = True,
                copy_binary = export2phy_with_recording,
                use_relative_path = False,
                verbose = True, 
                **job_kwargs
            )

            del sorting_analyzer
                
        ###############################################################################################################################
        # DELETE SORTING RESULTS
        ###############################################################################################################################
        del si_recording_loaded

        if not saveSortingAnalyzer: 
            shutil.rmtree(folder_analyzer, ignore_errors=True)
        
        # Temporal binary Recording folders (preprocessed & whitened) are not deleted until user explicitly clear the entire folder_temporal
        # This will allow to run diferent sorter params without re-writing the recording traces 

    return folder_temporal
