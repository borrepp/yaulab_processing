import os
import numpy
import shutil
from tkinter.filedialog import askdirectory

from ..yaulab_extras import get_tempdir

from .spikeinterface_sorting import (
    ms_before_default,
    ms_after_default,
    get_sorter_and_sortingAnalyzer_params,
    create_postprocessed_sorting,
    createFULL_sortingAnalyzer_folder,
    validate_sortingAnalyzer    
)

from pynwb import NWBHDF5IO

from spikeinterface.core import load_extractor, load_sorting_analyzer, split_sorting, select_segment_sorting
from spikeinterface.extractors import read_phy

from warnings import warn


####################################################################################################################################################################################
#                                                                    SOME DEFAULT PARAMETERS
####################################################################################################################################################################################

processName_prefix = 'SIprepSortExport'

n_cpus = os.cpu_count()
n_jobs = n_cpus - 2

job_kwargs = dict(chunk_duration="1s", n_jobs=n_jobs, progress_bar=True)

# si.set_global_job_kwargs(**job_kwargs)


##############################################################################################################
# Spikeinterface LFP options
##############################################################################################################
lfp_params_spikeInterface_default = {
    
    'lfp_description': 'SpikeInterface Preprocessing: 1) BANDPASS (see scipy.signal.iirfilter): Butterworth filter using second-order sections. LowFreq: (order:5, freq-corner:0.1); HighFreq: (order:5, freq-corner:300.0). Margin ms: 5.0. Direction: forward-backward. See scipy.signal.iirfilter. 2) Resample (see scipy.signal.resample): Sampling rate 1KHz',

    # Spikeinterface "BANDPASS_FILTER" opts:
    #
    # freq_min : float
    #     The highpass cutoff frequency in Hz
    # freq_max : float
    #     The lowpass cutoff frequency in Hz
    # margin_ms : float, default: 5.0
    #     Margin in ms on border to avoid border effect
    # dtype : dtype or None, default: None
    #     The dtype of the returned traces. If None, the dtype of the parent recording is used
    # coeff : array | None, default: None
    #     Filter coefficients in the filter_mode form.
    # add_reflect_padding : Bool, default False
    #     If True, uses a left and right margin during calculation.
    # filter_order : float, default: 5.0
    #     The order of the filter for `scipy.signal.iirfilter`
    # filter_mode :  "sos" | "ba", default: "sos"
    #     Filter form of the filter coefficients for `scipy.signal.iirfilter`:
    #     - second-order sections ("sos")
    #     - numerator/denominator : ("ba")
    # ftype : str, default: "butter"
    #     Filter type for `scipy.signal.iirfilter` e.g. "butter", "cheby1".
    # direction : "forward" | "backward" | "forward-backward", default: "forward-backward"
    #     Direction of filtering:
    #     - "forward" - filter is applied to the timeseries in one direction, creating phase shifts
    #     - "backward" - the timeseries is reversed, the filter is applied and filtered timeseries reversed again. Creates phase shifts in the opposite direction to "forward"
    #     - "forward-backward" - Applies the filter in the forward and backward direction, resulting in zero-phase filtering. Note this doubles the effective filter order.
    'bandpass': {
        'freq_min': 0.1,
        'freq_max': 300.0,
        'margin_ms': 150.0, # Default 5.0. (5 ms at 30K = 150 samples). To match ~150 samples at 1K, set to 150
        'dtype': None,
        'coeff': None,
        'add_reflect_padding': False,
        'filter_order': 5,
        'filter_mode': 'sos',
        'ftype': 'butter',
        'direction': 'forward-backward'
    },

    # Spikeinterface "RESAMPLE" opts:
    #
    # resample_rate : int
    #     The resampling frequency
    # margin_ms : float, default: 100.0
    #     Margin in ms for computations, will be used to decrease edge effects.
    # dtype : dtype or None, default: None
    #     The dtype of the returned traces. If None, the dtype of the parent recording is used.
    # skip_checks : bool, default: False
    #     If True, checks on sampling frequencies and cutoff filter frequencies are skipped
    'resample': {
        'resample_rate': 1000,
        'margin_ms': 3000.0, # Default 100.0 (100 ms at 30K = 3000 samples), To match ~3000 samples at 1K, set to 3000.0
        'dtype': None,
        'skip_checks': False,
    }
}

####################################################################################################################################################################################
# GET DEFAULT sorted UNITS column names (it must match with "sorting_analyzer" extensions cumputed by spikeinterface)
####################################################################################################################################################################################
def get_default_unitsCols(waveform_samples):
    
    ####################################################################################################################################################################################
    # Get Unit table description
    ####################################################################################################################################################################################
    unit_table_description = 'Sorted units come from a custom made Yau-lab SpikeInterface pipeline. Units table includes results from a spileinterface-sorting_analyzer created after manual curation (PHY or Spikeinterface)'

    ####################################################################################################################################################################################
    # Get SORTING property descriptions that will be added as columns
    # Note, everyting that wish to be added, needs to be a property of the sorter if you use NEUROCONV
    ####################################################################################################################################################################################
    unit_columns_information = dict(

        ########################################################################
        # Column added by NEUROCONV
        unit_name = dict(
            default_value = "NaN",
            default_type = str,
            description="Unique reference for each unit."
            ), 
        # PHY sorting property 
        quality = dict(
            default_value = "notinspected",
            default_type = str,
            description="Quality of the unit as defined by phy or spikeiterface curation (good, mua, noise)."
            ),
        # PHY-Spikeinterface sorting property
        original_cluster_id = dict(
            default_value = "NaN",
            default_type = str,
            description="Cluster ID from PHY / spikeinterface"
            ), 

        ########################################################################
        # Properties per spike detected:
        spike_amplitudes = dict(
            default_value = numpy.array([numpy.nan]),
            default_type = numpy.ndarray,
            index = 1,
            description="Spikeinterface amplitude of each spike as the value of the traces on the extremum channel at the times of each spike."
            ),
        amplitude_scalings = dict(
            default_value = numpy.array([numpy.nan]),
            default_type = numpy.ndarray,
            index = 1,
            description=" Spikeinterface Amplitude scalings are the scaling factor to multiply the unit template to best match the waveform. Each waveform has an associated amplitude scaling."
            ),
        spike_rel_x = dict(
            default_value = numpy.array([numpy.nan]),
            default_type = numpy.ndarray,
            index = 1,
            description="x location in the probe plane. Spikeinterface location of each spike in the sorting output. Spike location estimates can be done with center of mass, a monopolar triangulation, or with the method of grid convolution."
            ),
        spike_rel_y = dict(
            default_value = numpy.array([numpy.nan]),
            default_type = numpy.ndarray,
            index = 1,
            description="y is the depth by convention. Spikeinterface location of each spike in the sorting output. Spike location estimates can be done with center of mass, a monopolar triangulation, or with the method of grid convolution."
            ),
        spike_rel_z = dict(
            default_value = numpy.array([numpy.nan]),
            default_type = numpy.ndarray,
            index = 1,
            description="z it the orthogonal axis to the probe plane. Spikeinterface location of each spike in the sorting output. Spike location estimates can be done with center of mass, a monopolar triangulation, or with the method of grid convolution."
            ),
        spike_rel_alpha = dict(
            default_value = numpy.array([numpy.nan]),
            default_type = numpy.ndarray,
            index = 1,
            description="Alpha is the amplitude at source estimation. Spikeinterface location of each spike in the sorting output. Spike location estimates can be done with center of mass, a monopolar triangulation, or with the method of grid convolution."
            ),
        
        ########################################################################
        # Properties default NWB:
        waveform_mean = dict(
            default_value = numpy.full((1, waveform_samples), numpy.nan),
            default_type = numpy.ndarray,
            index = 1,
            description="The extracellular average waveform."
            ), 
        waveform_sd = dict(
            default_value = numpy.full((1, waveform_samples), numpy.nan),
            default_type = numpy.ndarray,
            index = 1,
            description="The extracellular Standard Deviation of the waveform."
            ),

        ########################################################################
        # Extra UNIT properties
        unit_amplitude = dict(
            default_value = numpy.nan,
            default_type = float,
            description=" Spikeinterface average amplitude of peaks detected on the best channel"
            ),
        unit_rel_x = dict(
            default_value = numpy.nan,
            default_type = float,
            description='x location in the probe plane. Spikeinterface metric similar to the spike_locations, but instead of estimating a location for each spike based on individual waveforms, it calculates at the unit level using templates. The same localization methods (method="center_of_mass" | "monopolar_triangulation" | "grid_convolution") are available.'
            ),
        unit_rel_y = dict(
            default_value = numpy.nan,
            default_type = float,
            description='y is the depth by convention. Spikeinterface metric similar to the spike_locations, but instead of estimating a location for each spike based on individual waveforms, it calculates at the unit level using templates. The same localization methods (method="center_of_mass" | "monopolar_triangulation" | "grid_convolution") are available.'
            ),
        unit_rel_z = dict(
            default_value = numpy.nan,
            default_type = float,
            description='z it the orthogonal axis to the probe plane. Spikeinterface metric similar to the spike_locations, but instead of estimating a location for each spike based on individual waveforms, it calculates at the unit level using templates. The same localization methods (method="center_of_mass" | "monopolar_triangulation" | "grid_convolution") are available.'
            ),
        unit_rel_alpha = dict(
            default_value = numpy.nan,
            default_type = float,
            description='Alpha is the amplitude at source estimation. Spikeinterface metric similar to the spike_locations, but instead of estimating a location for each spike based on individual waveforms, it calculates at the unit level using templates. The same localization methods (method="center_of_mass" | "monopolar_triangulation" | "grid_convolution") are available.'
            ),
            
        ########################################################################
        # NEUROCONV will recognized this name value to add NWB Electrode-Table attributes
        max_electrode = dict(
            default_value = -1,
            default_type = int,
            description="The recording channel id with the largest amplitude."
            ), 

        ########################################################################
        # Template Metrics:
        peak_to_valley = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-TemplateMetric: The duration in secs between the negative and the positive peaks computed on the maximum channel."
            ),
        peak_trough_ratio = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-TemplateMetric: Ratio between negative and positive peaks."
            ),
        halfwidth = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-TemplateMetric: The full-width (in secs) half maximum of the negative peak computed on the maximum channel."
            ),
        repolarization_slope = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-TemplateMetric: Speed to repolarize from the positive peak to 0."
            ),
        recovery_slope = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-TemplateMetric: Speed to recover from the negative peak to 0."
            ),
        num_positive_peaks = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-TemplateMetric: The number of positive peaks."
            ),
        num_negative_peaks = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-TemplateMetric: The number of negative peaks."
            ),     
        velocity_above = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-TemplateMetric: the velocity (in microns per second) above the max channel of the template."
            ), 
        velocity_below = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-TemplateMetric: the velocity (in microns per second) below the max channel of the template."
            ), 
        exp_decay = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-TemplateMetric: the exponential decay (in microns) of the template amplitude over distance."
            ), 
        spread = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-TemplateMetric:. the spread (in microns) of the template amplitude over distance"
            ), 
        
        ########################################################################
        # Quality metrics:
        num_spikes = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: Total number of spike events from this unit."
            ),
        firing_rate = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: the average number of spikes within the recording per second. Both very high and very low firing rates can indicate errors."
            ),
        presence_ratio = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: Calculate the presence ratio, the proportion of discrete time bins in which at least one spike occurred. Complete units are expected to have a presence ratio of 90% or more."
            ),
        snr = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: The signal-to-noise ratio of the unit. A high SNR unit has a signal which is greater in amplitude than the background noise and is likely to correspond to a neuron. A low SNR value (close to 0) suggests that the unit is highly contaminated by noise."
            ),
        
        #isi_violation = dict(
        #    default_value = numpy.nan,
        #    default_type = float,
        #    description="Spikeinterface-QualityMetric: measures the InterSpikeInterval violation ratio as a proxy for the purity of the unit. A high value indicates a highly contaminated unit. Being a ratio, the contamination can exceed 1."
        #    ),
        
        isi_violations_ratio = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: the relative firing rate of the hypothetical neurons that are generating the ISI violations. A high value indicates a highly contaminated unit. Being a ratio, the contamination can exceed 1."
            ),
        isi_violations_count = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: number of ISI violations. A high value indicates a highly contaminated unit. Being a ratio, the contamination can exceed 1."
            ),
        rp_contamination = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: The refactory period contamination described in [Llobet]. A high value indicates a highly contaminated unit."
            ),
        rp_violations = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: Calculate the number of refractory period violations. A high value indicates a highly contaminated unit."
            ),
        sliding_rp_violation = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: Compute sliding refractory period violations, a metric developed by IBL which computes contamination by using a sliding refractory period. This metric computes the minimum contamination with at least 90% confidence. A high number of violations indicates contamination, so a low value is expected for high quality units."
            ),
        amplitude_cutoff = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: Calculate approximate fraction of spikes missing from a Gaussian distribution of amplitudes. smaller amplitude cutoff values tend to indicate higher quality units."
            ),
        amplitude_median = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: Compute median of the amplitude distributions (in absolute value). A larger value (larger signal) indicates a better unit."
            ),
        
        #amplitude_cv = dict(
        #    default_value = numpy.nan,
        #    default_type = float,
        #    description="Spikeinterface-QualityMetric: Calculate coefficient of variation of spike amplitudes within defined temporal bins. From the distribution of coefficient of variations, both the median and the “range” (the distance between the percentiles defined by percentiles parameter) are returned. The amplitude CV median is expected to be relatively low for well-isolated units. The amplitude CV range can be high in the presence of noise contamination."
        #    ),
        
        amplitude_cv_median = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: MEDIAN from the distribution of coefficient of variations. Calculate coefficient of variation of spike amplitudes within defined temporal bins. The amplitude CV median is expected to be relatively low for well-isolated units."
            ),
        amplitude_cv_range = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: The “range” (the distance between the percentiles defined by percentiles parameter) from the distribution of coefficient of variations. Calculate coefficient of variation of spike amplitudes within defined temporal bins. The amplitude CV range can be high in the presence of noise contamination."
            ),
        
        #synchrony = dict(
        #    default_value = numpy.nan,
        #    default_type = float,
        #    description="Spikeinterface-QualityMetric: Synchrony metrics represent the rate of occurrences of spikes at the exact same sample index, with synchrony sizes 2, 4 and 8. A larger value indicates a higher synchrony of the respective spike train with the other spike trains. Larger values, especially for larger sizes, indicate a higher probability of noisy spikes in spike trains."
        #    ),
        
        sync_spike_2 = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: Synchrony metrics represent the rate of occurrences of spikes at the exact same sample index, with synchrony sizes 2. A larger value indicates a higher synchrony of the respective spike train with the other spike trains. Larger values, especially for larger sizes, indicate a higher probability of noisy spikes in spike trains."
            ),
        sync_spike_4 = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: Synchrony metrics represent the rate of occurrences of spikes at the exact same sample index, with synchrony sizes 4. A larger value indicates a higher synchrony of the respective spike train with the other spike trains. Larger values, especially for larger sizes, indicate a higher probability of noisy spikes in spike trains."
            ),
        sync_spike_8 = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: Synchrony metrics represent the rate of occurrences of spikes at the exact same sample index, with synchrony sizes 8. A larger value indicates a higher synchrony of the respective spike train with the other spike trains. Larger values, especially for larger sizes, indicate a higher probability of noisy spikes in spike trains."
            ),
        firing_range = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: Calculate firing range, the range between the 5th and 95th percentiles of the firing rates distribution computed in non-overlapping time bins. Very high levels of firing ranges, outside of a physiological range, might indicate noise contamination."
            ),
        
        #drift = dict(
        #    default_value = numpy.nan,
        #    default_type = float,
        #    description="Spikeinterface-QualityMetric: Compute drifts metrics using estimated spike locations. Over the duration of the recording, the drift signal for each unit is calculated as the median position in an interval with respect to the overall median positions over the entire duration (reference position). Larger values indicate more “drifty” units, possibly of lower quality."
        #    ),
        
        drift_ptp = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: The peak-to-peak of the drift signal for each unit using estimated spike locations. Over the duration of the recording, the drift signal for each unit is calculated as the median position in an interval with respect to the overall median positions over the entire duration (reference position). Larger values indicate more “drifty” units, possibly of lower quality."
            ),
        drift_std = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: The standard deviation of the drift signal for each unit using estimated spike locations. Over the duration of the recording, the drift signal for each unit is calculated as the median position in an interval with respect to the overall median positions over the entire duration (reference position). Larger values indicate more “drifty” units, possibly of lower quality."
            ),
        drift_mad = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: The median absolute deviation of the drift signal for each uni using estimated spike locations. Over the duration of the recording, the drift signal for each unit is calculated as the median position in an interval with respect to the overall median positions over the entire duration (reference position). Larger values indicate more “drifty” units, possibly of lower quality."
            ),
        sd_ratio = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: Computes the SD (Standard Deviation) of each unit's spike amplitudes, and compare it to the SD of noise. In this case, noise refers to the global voltage trace on the same channel as the best channel of the unit. (ideally (not implemented yet), the noise would be computed outside of spikes from the unit itself). For a unit representing a single neuron, this metric should return a value close to one. However for units that are contaminated, the value can be significantly higher."
            ),
        isolation_distance = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: Calculate isolation distance (computed from Mahalanobis distance). Can be interpreted as a measure of distance from the cluster to the nearest other cluster. A well isolated unit should have a large isolation distance."
            ),
        l_ratio = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric:  L-ratio metric computed from Mahalanobis distance. Since this metric identifies unit separation, a high value indicates a highly contaminated unit."
            ),
        d_prime = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: Calculate d-prime based on Linear Discriminant Analysis to estimate the classification accuracy of the unit. Hit_rate: Fraction of neighbors for target cluster that are also in target cluster. Miss_rate:Fraction of neighbors outside target cluster that are in target cluster. D-prime is a measure of cluster separation, and will be larger in well separated clusters."
            ),
        silhouette = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: Calculate the simplified silhouette score for each cluster. The value ranges from -1 (bad clustering) to 1 (good clustering). The simplified silhoutte score utilizes the centroids for distance calculations rather than pairwise calculations."
            ),
        
        #nearest_neighbor = dict(
        #    default_value = numpy.nan,
        #    default_type = float,
        #    description="Spikeinterface-QualityMetric: Calculate unit contamination based on NearestNeighbors search in PCA space."
        #    ),

        nn_hit_rate = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: Calculate unit contamination based on NearestNeighbors (NN) search in PCA space. NN-hit rate gives an estimate of contamination (an uncontaminated unit should have a high NN-hit rate)"
            ), 
        nn_miss_rate = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: Calculate unit contamination based on NearestNeighbors search in PCA space. NN-miss rate gives an estimate of completeness. A more complete unit should have a low NN-miss rate"
            ), 
        nn_isolation = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: Calculate unit contamination based on NearestNeighbors search in PCA space. compute the pairwise isolation score between the chosen cluster and every other cluster. The isolation score is then the minimum of the pairwise scores (the worst case)"
            ), 
        nn_noise_overlap = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Spikeinterface-QualityMetric: Calculate unit contamination based on NearestNeighbors search in PCA space. A noise cluster is generated by randomly sampling voltage snippets from the recording. Following a similar procedure to that of the nn_isolation method, compute isolation between the cluster of interest and the generated noise cluster. This metric gives an indication of the contamination present in the unit cluster"
            ),

        ########################################################################
        # Information about sessions & sorting parameters
        sessions_shared = dict(
            default_value = numpy.array([""]),
            default_type = numpy.ndarray,
            index=1,
            description="A list of NWBfile names sharing this unit/cluster (It means the sorting was performed concatenating this sessions)."
            ),
        motion_corrected = dict(
            default_value = "unknown",
            default_type = str,
            description="Whether motion correction preprocessing was perform or not (Spikeinterface preprocessing)"
            ),
        recording_sufix = dict(
            default_value = "unknown",
            default_type = str,
            description="String made of peaksLoc_label + motion_label + interpolation_label parameters"
            ),
        curated_with_phy = dict(
            default_value = "unknown",
            default_type = str,
            description="Whether the curation was made using PHY or via Spikeinterface postprocessing & widgets"
            ),
        sorterName = dict(
            default_value = "unknown",
            default_type = str,
            description="Sorting package used to sort the unit"
            ),
        sorter_detect_sign = dict(
            default_value = "unknown",
            default_type = str,
            description="Sign of the peak to detect events to sort"
            ),
        sorter_detect_threshold = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Threshold used to detect events on the withened traces (units depend on the sorter: waveclus & kilosort4 = STD, mountainsort5 = ZCA)"
            ),
        sorter_nearest_chans = dict(
            default_value = numpy.nan,
            default_type = float,
            description="Number of NEIGHBOR CHANNELS to search for the same waveform during sorting & sorting_analyzer. If set to ZERO, it will perform similar to sorting single channels. KS do not accept 0, minumum 1"
            ),   
    )
    
    return unit_table_description, unit_columns_information


####################################################################################################################################################################################
####################################################################################################################################################################################
#                                                                   MAIN FUNCTIONS
####################################################################################################################################################################################
####################################################################################################################################################################################

#########################################################################################################################################
# FUNCTION TO EXTRACT RECORDING & SORTING INFORMATION FROM THE CURATION FOLDER PATH-NAME
#########################################################################################################################################
def get_params_from_curatedPath(curatedFolder_path=None, ms_before=ms_before_default, ms_after=ms_after_default):

    if curatedFolder_path is None:
        curatedFolder_path = askdirectory(title='Select curated "PHY" or "SortingAnalyzer" folder to export', mustexist=True)

    ###############################################################################
    # Get general parentFolders & expSession ID information
    ###############################################################################
    curatedFolder_path = os.path.abspath(curatedFolder_path)
    elecGroupSessSortingFolder, curation_folder = os.path.split(curatedFolder_path)
    sessionDaySortingFolder, elecGroupSessName = os.path.split(elecGroupSessSortingFolder)
    parentSortingFolder, sessionDayFolder = os.path.split(sessionDaySortingFolder)

    electrodeGroup_Name = elecGroupSessName[elecGroupSessName.index('-raw-')+1:]
    session_index_name = elecGroupSessName[elecGroupSessName.index('-sess-')+6:elecGroupSessName.index('-raw-')]

    # if '_' in "session_index_name" means it is a consecutive range of session index
    if '_' in session_index_name:
        start_stop_index = session_index_name.split('_')
        if len(start_stop_index)!=2:
            raise Exception('When detected a range of sessions two numbers are expected : {}'.format(start_stop_index))
        session_index = [i for i in range( int(start_stop_index[0]), int(start_stop_index[1])+1)]
        
    # if '-' in "session_index_name" means non-consecutive range of session index
    elif '-' in session_index_name:
        session_index = [int(i) for i in session_index_name.split('-')]
    else:
        raise Exception('Session-index syntax not recognized: {}'.format(session_index_name))


    sessionExpDay = sessionDayFolder.removesuffix('_sorting').removeprefix('exp').split('-')
    sessionYear = int(sessionExpDay[0])
    sessionMonth = int(sessionExpDay[1])
    sessionDay = int(sessionExpDay[2])


    ###############################################################################
    # Get Sorting Info : 
    # sorterName, sorter_detect_sign, sorter_detect_threshold, sorter_nearest_chans
    ###############################################################################

    is_phy = False
    stop_sorting_keyName = len(curation_folder)

    # Search first for sufix '_curated in case was added to the '_phy' ending
    # Otherwise, sufix '_curated' might exist only when sorting_analyzer objects was used for curation
    if curation_folder.endswith('_curated'):
        stop_sorting_keyName -= 8

    if curation_folder.endswith('_phy'):
        is_phy = True
        stop_sorting_keyName -= 4

    # By default ALL sorters will have the sign_label in the name
    # But the nearest_chan is not added to WaveClus naming
    sorter_nearest_chans = None
    if '_WCLUS' in curation_folder:
        sorter_prefix = '_WCLUS'
        sorterName ='waveclus'
        sorter_nearest_chans = 0
    elif '_MS5' in curation_folder:
        sorter_prefix = '_MS5'
        sorterName ='mountainsort5'
    elif '_KS4' in curation_folder:
        sorter_prefix= '_KS4'
        sorterName = 'kilosort4'
    else:
        raise Exception('Curation folder : \n"{}"\nis not supported by this pipeline version\nSupported sorters contains Prefixes:\n\t{}\n\t{}\n\t{}'.format(
            curation_folder, 'waveclus: "_WCLUS"', 'mountainsort5: "_MS5"', 'kilosort4: "_KS4"'))

    start_sorting_keyName = curation_folder.index(sorter_prefix)

    sorter_label = curation_folder[start_sorting_keyName+1:stop_sorting_keyName]

    if sorter_nearest_chans is None:
        sorter_nearest_chans = int(sorter_label[sorter_label.index('ch')+2:sorter_label.index('th')])

    threshold_key = sorter_label[sorter_label.index('th')+2:stop_sorting_keyName]
    if threshold_key[0].isnumeric():
        sorter_detect_sign_str = ''
        sorter_detect_threshold_str = threshold_key.replace('_', '.')
    else:
        sorter_detect_sign_str = threshold_key[0].upper()
        sorter_detect_threshold_str = threshold_key[1::].replace('_', '.')

    if not sorter_detect_threshold_str.isnumeric:
        raise Exception('Sorter Threshold = {} was not a valid value from sorting folder name: {}'.format(
            sorter_detect_threshold_str, sorter_label))

    sorter_detect_threshold = float(sorter_detect_threshold_str)

    if sorter_detect_sign_str=='':
        sorter_detect_sign = 'both'
    elif sorter_detect_sign_str=='N':
        sorter_detect_sign = 'neg'
    elif sorter_detect_sign_str=='P':
        sorter_detect_sign = 'pos'
    else:
        raise Exception('Sorter detect sign = {} was not a valid value from sorting folder name: {}'.format(
            sorter_detect_sign_str, sorter_label))

    ###############################################################################
    # Get Recording Info
    ###############################################################################
    recordingName = curation_folder[0:start_sorting_keyName] 
    recording_sufix = recordingName.removeprefix(elecGroupSessName)
    if recording_sufix == '':
        motion_corrected = False
    else:
        motion_corrected = True

    # binary Recording will exists in the folder that contains the curated folder
    recording_binary_name = recordingName + '_binary'

    si_recording_folder = os.path.join(elecGroupSessSortingFolder, recording_binary_name)

    if not os.path.exists(os.path.join(si_recording_folder, 'binary.json')) or not os.path.exists(os.path.join(si_recording_folder, 'si_folder.json')):
        raise Exception('Recording Binary folder must contain "binary.json" & "si_folder.json" files')
    else:
        si_recording_loaded = load_extractor(si_recording_folder)
        print('reading Recording from binary successful¡......\n')
        print(si_recording_loaded, '\n\n')

        nChans = si_recording_loaded.get_num_channels()
        step_chan = si_recording_loaded.get_annotation('y_contacts_distance_um')
        sampling_frequency = si_recording_loaded.sampling_frequency
        recording_total_duration = si_recording_loaded.get_total_duration()

        del si_recording_loaded
    
    snippet_T1 = int(numpy.ceil(ms_before * sampling_frequency / 1000.0))
    snippet_T2 = int(numpy.ceil(ms_after * sampling_frequency / 1000.0))
    
    return {
        'phy_or_sorter_analyzer_path': curatedFolder_path,
        'parentSortingFolder': parentSortingFolder,
        'sessionYear': sessionYear,
        'sessionMonth': sessionMonth,
        'sessionDay': sessionDay,
        'electrodeGroup_Name': electrodeGroup_Name,
        'session_index': session_index,
        'elecGroupSessName' : elecGroupSessName,
        'si_recording_folder': si_recording_folder,
        'recording_binary_name': recording_binary_name,
        'recording_sufix': recording_sufix,
        'motion_corrected': motion_corrected,
        'nChans': nChans,
        'step_chan': step_chan,
        'sampling_frequency': sampling_frequency,
        'recording_total_duration': recording_total_duration,
        'ms_before': ms_before,
        'ms_after': ms_after,
        'n_before': snippet_T1,
        'n_after': snippet_T2,
        'curated_with_phy': is_phy,
        'sorterName': sorterName,
        'sorter_detect_sign': sorter_detect_sign, 
        'sorter_detect_threshold': sorter_detect_threshold, 
        'sorter_nearest_chans': sorter_nearest_chans
    }


##############################################################################################################
# Find the "-prepro.nwb" filepath corresponding to a given recording session
##############################################################################################################
def find_peproNWBpath_from_si_recording_session(si_recording, parentFolder_preproNWB, verbose=True):

    # Check the recording object has the session_index property (a parameter added during exporting a curation folder)
    if si_recording.get_annotation('session_index') is None:
        raise Exception('Recording object must contain annotation = "session_index".\nCurrent annotations in the recording: {}'.format(list(si_recording.get_annotation_keys())))
    
    session_index = si_recording.get_annotation('session_index')
    
    nwbPrepro_path_list = []
    shared_prepro_fileNames = []
    for nwbRaw_path in si_recording.get_annotation('fileNamePaths'):

        _, nwbFileName = os.path.split(nwbRaw_path)
        nwbFileNameSplit = os.path.splitext(nwbFileName)
        nwbFilePrefix = nwbFileNameSplit[0]
        del nwbFileName, nwbFileNameSplit

        nwbPrepro_list = []
        parentFolder = os.path.abspath(parentFolder_preproNWB)
        for root, _, files in os.walk(parentFolder):
            for name in files:
                nameSplit = os.path.splitext(name)
                if nameSplit[1]=='.nwb' and nwbFilePrefix in nameSplit[0] and '_prepro' in nameSplit[0]:
                    nwbPrepro_list.append(os.path.join(root, name))

        if len(nwbPrepro_list)>1:
            raise Exception('There are {} nwbFiles with the name : {}.\nConfirm the parentFolder_preproNWB is correct (current path: {})'.format(nwbFilePrefix+'_prepro.nwb', parentFolder))
        elif len(nwbPrepro_list)==0:
            raise Exception('Preprocessed NWB has NOT been found for raw NWB: {}\nIt requires to create NWBprepro first¡'.format(nwbFilePrefix))
        
        nwbPrepro_path_list.append(nwbPrepro_list[0])
        shared_prepro_fileNames.append(nwbFilePrefix)

    nwbPrepro_path = nwbPrepro_path_list[session_index]

    # Remove the current preproFile to get the remaning session Names (variable use in the UNITS-table)
    current_prepro_fileName = shared_prepro_fileNames.pop(session_index)
    
    if verbose:
        print('\nPreprocessed NWB file {} ({}/{} concatenated session) has been found at:\n\t{}\n'.format(current_prepro_fileName, session_index+1, si_recording.get_annotation('nSessions'), nwbPrepro_path))
    
    
    return nwbPrepro_path, current_prepro_fileName, shared_prepro_fileNames

##############################################################################################################
# Update recording properties to match NWB electrode table:
##############################################################################################################
def udpdate_si_recording_with_nwb_electrodes_table(si_recording, nwbFile_path, electrodeGroupName=None, verbose=False):
    
    nwbFile_io = NWBHDF5IO(nwbFile_path, mode="r+")
    nwbFile = nwbFile_io.read()


    # REMOVE properties not found in the NWB.electrode Table, except those that NEUROCONV requires
    special_cases_from_neuroconv = [
        "offset_to_uV",  # Use to written in the ElectricalSeries
        "gain_to_uV",  # Use to written in the ElectricalSeries
        "contact_vector",  # Structured array representing the probe # It will be deleted automatically by NEUROCONV
        "channel_labels", # if the channel was label as "good"/"dead"/"noise"/"bad" during "spikeinterface.preprocessing.detect_bad_channels"
    ]
    properties_to_remove = [p for p in si_recording.get_property_keys() if p not in nwbFile.electrodes.colnames and p not in special_cases_from_neuroconv]

    for p in properties_to_remove:
        if verbose:
            print('Removing property "{}" from recording'.format(p))
        si_recording.delete_property(p)
        
    # ADD electrode group names:
    if si_recording.get_property("group_name") is None and electrodeGroupName is not None:
        if verbose:
            print('Adding to the recording property: "group_name"')
        si_recording.set_property(key = "group_name", values=[electrodeGroupName]*si_recording.get_num_channels())
    
    # ADD the remaining Electrode columns as properties to avoid refilling the remaining columns with default values 
    nwb_electrodes_ids = list(nwbFile.electrodes.to_dataframe().index.values)
    channel_indexes_in_NWB = [nwb_electrodes_ids.index(e) for e in si_recording.channel_ids]

    properties_to_add = [p for p in nwbFile.electrodes.colnames if p not in si_recording.get_property_keys()]
    for p in properties_to_add:
        if verbose:
            print('Adding to the recording property: "{}"'.format(p))
        val_list = nwbFile.electrodes.to_dataframe().loc[:, p].to_list()
        val_sorted = []
        for ch in range(si_recording.get_num_channels()):
            val_sorted.append(val_list[channel_indexes_in_NWB[ch]])
        
        si_recording.set_property(key = p, values = val_sorted)
    
    # If "channel_labels" already exists in the NWBfile, rewrite its value
    # otherwise only the first probe will have values and the rest will be set to empty string
    if "channel_labels" in nwbFile.electrodes.colnames and si_recording.get_property("channel_labels") is not None:
        channel_labels = si_recording.get_property("channel_labels")
        if verbose:
            print('Updating Electrode-column = "channel_labels" in the NWBfile')
        for ch in range(si_recording.get_num_channels()):
            nwbFile.electrodes["channel_labels"].data[channel_indexes_in_NWB[ch]] = str(channel_labels[ch])

    nwbFile_io.close()
    del nwbFile_io, nwbFile

    return si_recording

####################################################################################################################################################################################
#                                                      MAIN FUNCTION TO CREATE SORTING ANALYZERS FOR NWB EXPORTING:
####################################################################################################################################################################################
#  It will use the path of a curated sorting and will split it into their respective Recording & Sorting_Analyzer folder per session
####################################################################################################################################################################################
def export_curatedFolder_to_sortingAnalyzer_sessions(curatedFolder_path, start_unit_id=None, ms_before=ms_before_default, ms_after=ms_after_default, rewriteAnalyzer=False, verbose=False):

    ###############################################################################################################################
    # GET General information about the recording session and preprocessing
    curated_params = get_params_from_curatedPath(curatedFolder_path, ms_before=ms_before, ms_after=ms_after)
    
    if verbose:
        print('\nInformation about curated sorting:')
        for k, v in curated_params.items():
            print('\t{} : {}'.format(k, v))

    ###############################################################################################################################
    # GET Sorter & Sorting_analyzer params
    sorter_and_sortingAnalyzer_params = get_sorter_and_sortingAnalyzer_params(
        sorterName = curated_params['sorterName'], 
        nChans = curated_params['nChans'],
        step_chan = curated_params['step_chan'],
        sampling_frequency = curated_params['sampling_frequency'],
        recording_total_duration = curated_params['recording_total_duration'],
        ms_before = ms_before, 
        ms_after = ms_after, 
        peak_sign = curated_params['sorter_detect_sign'],
        nearest_chans = curated_params['sorter_nearest_chans'],
        sorter_whitening=False, 
        detect_threshold = curated_params['sorter_detect_threshold']
        )

    # sorter_info = sorter_and_sortingAnalyzer_params['sorter_info']
    estimate_sparsity_params = sorter_and_sortingAnalyzer_params['estimate_sparsity_params']
    sorting_analyzer_params = sorter_and_sortingAnalyzer_params['sorting_analyzer_params']
    """
    if verbose:
        print('\nsorter_info')
        for k, v in sorter_info.items():
            print('\t{} : {}'.format(k, v))
        print('\n')

        print('estimate_sparsity_params')
        for k, v in estimate_sparsity_params.items():
            print('\t{} : {}'.format(k, v))
        print('\n')

        print('sorting_analyzer_params')
        for k, v in sorting_analyzer_params.items():
            print('\t{} : {}'.format(k, v))
    """
    ####################################################################################################################################################################################
    # Create "Recording" & "Sorting" & "Sorting_Analyzer" for each session
    ####################################################################################################################################################################################

    folder_temporal = get_tempdir(processName='{}-{}'.format(processName_prefix, curated_params['elecGroupSessName'] ), resetDir=rewriteAnalyzer)

    ######################################################################################################
    # SPLIT RECORDING
    si_recording_loaded = load_extractor(curated_params['si_recording_folder']) 
    start_frame = 0
    recording_list = []
    for n in range(si_recording_loaded.get_annotation('nSessions')):
        end_frame = start_frame + si_recording_loaded.get_annotation('sessionSamples')[n]
        recording_list.append(si_recording_loaded.frame_slice(start_frame=start_frame, end_frame=end_frame))
        start_frame = end_frame

    ######################################################################################################
    # SPLIT SORTING
    # GET & SAVE curated sorting in a temporal folder:
    sorting_folder = os.path.join(folder_temporal, curated_params['elecGroupSessName'] + curated_params['recording_sufix'] + '_curatedSorting')

    if curated_params['curated_with_phy']:

        sorter_loaded = read_phy(curated_params['phy_or_sorter_analyzer_path'], exclude_cluster_groups=["noise", "unsorted"], load_all_cluster_properties=False)
        if start_unit_id is None:
            sorter_loaded.save_to_folder(folder= sorting_folder,  overwrite=True, **job_kwargs)
        else:
            if verbose:
                print('Updating UNIT-IDs........')
            sorter_loaded_renamed = sorter_loaded.rename_units([i for i in range(start_unit_id, start_unit_id+len(sorter_loaded.unit_ids))])
            sorter_loaded_renamed.save_to_folder(folder= sorting_folder,  overwrite=True, **job_kwargs)
            del sorter_loaded_renamed

        del sorter_loaded

    else:

        sorting_analyzer = load_sorting_analyzer(folder=curated_params['phy_or_sorter_analyzer_path'], load_extensions=False)
        sorting_loaded = sorting_analyzer.sorting
        # TODO: Remove unsorted noise
        # remove_units(remove_unit_ids)
        
        if start_unit_id is None:
            sorting_loaded.save_to_folder(folder= sorting_folder,  overwrite=True, **job_kwargs)
        else:
            if verbose:
                print('Updating UNIT-IDs........')
            sorting_loaded_renamed = sorting_loaded.rename_units([i for i in range(start_unit_id, start_unit_id+len(sorting_loaded.unit_ids))])
            sorting_loaded_renamed.save_to_folder(folder= sorting_folder,  overwrite=True, **job_kwargs)
            del sorting_loaded_renamed
        
        del sorting_loaded, sorting_analyzer

    sorting_loaded = load_extractor(sorting_folder)
    num_units = len(sorting_loaded.unit_ids)

    # Get segmented sorting into a list
    sorting_multi = split_sorting(sorting_loaded, recording_list)
    sorting_list = []
    for n in range(si_recording_loaded.get_annotation('nSessions')):
        sorting_list.append(select_segment_sorting(sorting_multi, segment_indices=n))

    ######################################################################################################
    # Get temporal folder prefixes
    recordingPrefix_folder_temporal = os.path.join(folder_temporal, curated_params['elecGroupSessName'] + curated_params['recording_sufix']+ '_rec')
    sortingPrefix_folder_temporal = os.path.join(folder_temporal, curated_params['elecGroupSessName'] + curated_params['recording_sufix']+ '_sort')
    sortingAnalyzerPrefix_folder_temporal = os.path.join(folder_temporal, curated_params['elecGroupSessName'] + curated_params['recording_sufix']+ '_analyzer')

    recordingFolder_list = []
    sortingAnalyzerFolder_list = []

    for n in range(si_recording_loaded.get_annotation('nSessions')):

        ######################################################################################################
        # SAVE temporal RECORDING SEGMENT:
        recFolder = os.path.abspath('{}{:02d}'.format(recordingPrefix_folder_temporal, n))

        ###############################################################################################################################
        # Check if BynaryRecording already exists:
        if not os.path.exists(os.path.join(recFolder, 'binary.json')) or not os.path.exists(os.path.join(recFolder, 'si_folder.json')):

            if not os.path.isdir(recFolder):  
                os.makedirs(recFolder)

            recording_list[n].set_annotation(annotation_key='savePath', value=recFolder, overwrite=True)
            recording_list[n].set_annotation(annotation_key='session_index', value=n, overwrite=True)
            if verbose:
                print('\nprepraring to write Recording {} out of {} into binary......\n'.format(n+1, si_recording_loaded.get_annotation('nSessions')))
            recording_list[n].save_to_folder(folder=recFolder, overwrite=True, **job_kwargs)

        else:
            if verbose:
                print('\nBinary Recording {} out of {} has been found......\n'.format(n+1, si_recording_loaded.get_annotation('nSessions')))

        recordingFolder_list.append(recFolder)
        del recFolder

        ###############################################################################################################################
        # SAVE sortingAnalyzer 
        createAnalyzer_segment = True
        analyzerFolder = os.path.abspath('{}{:02d}'.format(sortingAnalyzerPrefix_folder_temporal, n))

        if os.path.isdir(analyzerFolder) and not rewriteAnalyzer:
            if verbose:
                print('Sorting Analyzer from segment {} out of {} has been found......\n'.format(n+1, si_recording_loaded.get_annotation('nSessions')))

            analyzerFolder_loaded = load_sorting_analyzer(folder=analyzerFolder, load_extensions=True)

            if validate_sortingAnalyzer(sorting_analyzer=analyzerFolder_loaded, sorting_analyzer_params=sorting_analyzer_params):
                createAnalyzer_segment=False
            
            del analyzerFolder_loaded

        else: 
            os.makedirs(analyzerFolder)
        
        if createAnalyzer_segment:
            ######################################################################################################
            # SAVE temporal SORTING SEGMENT:
            sortFolder = os.path.abspath('{}{:02d}'.format(sortingPrefix_folder_temporal, n))
            if not os.path.isdir(sortFolder):  
                os.makedirs(sortFolder)
            sorting_list[n].save_to_folder(folder= sortFolder,  overwrite=True, **job_kwargs)

            ###############################################################################################################################
            # DO SOME SORTING POSTPROCESSING
            ###############################################################################################################################
            si_recording_segment = load_extractor(recordingFolder_list[n]) 

            sorting_postProcessed_folder = create_postprocessed_sorting(sortFolder, si_recording_segment, sorting_analyzer_params, alingSorting = True, job_kwargs=job_kwargs)
            sorting_segment = load_extractor(sorting_postProcessed_folder)

            ###############################################################################################################################
            # Create final SORTING ANALYZER
            # When only one unit is detected, PHY can not create/load features and it will not launch
            # Therefore, the sorting analyzer and the recording object should not be deleted
            # Ensure saving SORTING ANALYZER when only one unit was found:
            ###############################################################################################################################
            if verbose:
                print('\n{} UNITS WERE FOUND ¡¡¡'.format(sorting_segment.get_num_units()))

            if sorting_segment.get_num_units()==0:

                #######################################################################################################################
                # If no units were found DELETE SORTING RESULTS
                ######################################################################
                # explicitly try to close the spikes.npy 
                try:
                    sorting_segment.spikes._mmap.close()
                except:
                    pass
                del sorting_segment, si_recording_segment

                print('\nDeleting temporal sorting files .............\n')
                shutil.rmtree(sorting_postProcessed_folder, ignore_errors=True)
                print('\nDone¡ .............\n')
                
                raise Exception('NO UNITS WERE FOUND ¡¡¡')

            elif sorting_segment.get_num_units()<2:

                #######################################################################################################################
                # When only 1 unit is founded
                warn('WARINING¡ Only ONE UNIT was found¡¡')

            createFULL_sortingAnalyzer_folder(sorting_segment, si_recording_segment, analyzerFolder, estimate_sparsity_params, sorting_analyzer_params, job_kwargs)

            ######################################################################
            # explicitly try to close the spikes.npy 
            try:
                sorting_segment.spikes._mmap.close()
            except:
                pass
            del sorting_segment, si_recording_segment
            if verbose:
                print('\nDeleting temporal sorting files .............\n')
            shutil.rmtree(sorting_postProcessed_folder, ignore_errors=True)
            shutil.rmtree(sortFolder, ignore_errors=True)

            del sorting_postProcessed_folder 
        
        sortingAnalyzerFolder_list.append(analyzerFolder)
        del analyzerFolder

    ######################################################################
    # explicitly try to close the spikes.npy 
    try:
        sorting_segment.spikes._mmap.close()
    except:
        pass
    del sorting_loaded
    shutil.rmtree(sorting_folder, ignore_errors=True)

    return dict(
        recordingFolder_list = recordingFolder_list, 
        sortingAnalyzerFolder_list=sortingAnalyzerFolder_list, 
        num_units=num_units, 
        curated_params=curated_params, 
        **sorter_and_sortingAnalyzer_params
        )
