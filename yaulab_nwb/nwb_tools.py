import os
import shutil
import numpy
import h5py
import time
import sys
import pandas

from tkinter.filedialog import askopenfilename

from ..yaulab_extras import (
    Unbuffered,
    removeRipple_stimulus_list_default,
    resample_stimulus_list_default,
    labInfo, 
    supported_probes_manufacturer,
    get_tempdir,
    clear_tempdir,
    check_temperature_date,
    check_folderSession_process
    )
from .yaml_tools import yaml2dict, getEyeStartTime, getEyeData, expYAML
from .ripple_tools import AnalogIOchannel, SegmentStimChannel, get_stimElectrodeInfo, get_nsFile, getNS_StartDateTime
from .constructor_tools import (
    get_eyePC_offset, 
    temp_TimeSeries_hdf5,
    temp_TimeSeries_hdf5_analog_cls,
    temp_hdf5_unitWaveforms_from_electrical_series,
    temp_hdf5_from_numpy_with_DataChunkIterator,
    temp_hdf5_from_HDMFGenericDataChunkIterator,
    temp_resample_hdf5_dataset,
    get_feetEvents, 
    get_rewardEvents,
    getNWB_rawElectrodes,
    create_rawElectrodeGroup_hdf5,
    getNWB_trials
)
from ..yaulab_si import (
    ms_before_default,
    ms_after_default,
    lfp_params_spikeInterface_default, 
    get_default_unitsCols,
    find_peproNWBpath_from_si_recording_session,
    udpdate_si_recording_with_nwb_electrodes_table,
    export_curatedFolder_to_sortingAnalyzer_sessions 
)

from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.common import DynamicTableRegion

from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.file import Subject
from pynwb.device import Device
from pynwb.epoch import TimeIntervals
from pynwb.behavior import SpatialSeries, EyeTracking, PupilTracking, Position
from pynwb.ecephys import ElectricalSeries, SpikeEventSeries, FilteredEphys #, LFP
from pynwb.misc import Units
from nwbinspector import inspect_nwbfile

from spikeinterface.core import load_extractor, load_sorting_analyzer, get_template_extremum_channel, get_template_extremum_amplitude
from spikeinterface.extractors import read_nwb as se_read_nwb
from spikeinterface.extractors import NumpyRecording
from spikeinterface.preprocessing import bandpass_filter, resample

from neuroconv.tools.spikeinterface.spikeinterface import add_electrical_series_to_nwbfile as neuroconv_add_electrical_series_to_nwbfile
from neuroconv.tools.spikeinterface.spikeinterfacerecordingdatachunkiterator import (
    SpikeInterfaceRecordingDataChunkIterator,
)

n_cpus = os.cpu_count()
n_jobs = n_cpus - 2

lfp_job_kwargs = dict(chunk_duration="30s", n_jobs=n_jobs, progress_bar=True)

description_channel_labels = 'If the channel was label as "good"/"dead"/"noise"/"bad" during spikeinterface preprocessing (see: spikeinterface.preprocessing.detect_bad_channels)'
description_ecephys = "Intermediate data from extracellular electrophysiology recordings, e.g., LFP."
comments_ecephys_processed ='This ElectricalSeries corresponds to the Processed signal sampled at 30KHz. The signal was preprocessed using Yaulab processing pipeline'

############################################################################################################################################################################################################################
############################################################################################################################################################################################################################
# Some complementary functions to export NWB

def _validate_si_recording_with_preproNWB(si_recording_folder, parentFolder_preproNWB, verbose=False):

    si_recording = load_extractor(si_recording_folder) 
    if not si_recording.is_binary_compatible():
        # Close all recording segments
        for segment_rec in si_recording._recording_segments:
                segment_rec.file.close()
        del si_recording
        raise Exception('It is required "si_recording" binary compatible')

    ##############################################################################################################
    # NOTE: It is assume that "nwbPrepro" is a "copy" of the "nwbRaw" except for the raw-aquisition-ephys,
    #       which might be already deleted from the Prepro. Therefore, the electrode table should match
    ##############################################################################################################
    nwbPrepro_path, _, _ = find_peproNWBpath_from_si_recording_session(si_recording, parentFolder_preproNWB, verbose=verbose)
    nwbPrepro_io = NWBHDF5IO(nwbPrepro_path, mode="r")
    nwbPrepro = nwbPrepro_io.read()

    if nwbPrepro.electrodes is None:
        # Close NWB
        nwbPrepro_io.close()
        del nwbPrepro_io, nwbPrepro
        # Close all recording segments
        for segment_rec in si_recording._recording_segments:
            segment_rec.file.close()
        del si_recording
        raise Exception('NWB file must contain electrode Table')
    elif "group_name" not in nwbPrepro.electrodes.colnames:
        # Close NWB
        nwbPrepro_io.close()
        del nwbPrepro_io, nwbPrepro
        # Close all recording segments
        for segment_rec in si_recording._recording_segments:
            segment_rec.file.close()
        del si_recording
        raise Exception('prepro NWB file must contain "group name" in the electrode Table')
    
    if "channel_name" not in nwbPrepro.electrodes.colnames:
        preproNWB_channel_names = numpy.array(nwbPrepro.electrodes.id[:]).astype("int")
    else:
        preproNWB_channel_names = nwbPrepro.electrodes["channel_name"][:]
    preproNWB_channels = [f"{ch_name}_{gr_name}" for ch_name, gr_name in zip(preproNWB_channel_names, nwbPrepro.electrodes["group_name"][:])]
        
    ##############################################################################################################
    # Confirm Electrode Group and Channel-IDs match between "si_recording" and "preproNWB"
    elecGroupSessName = si_recording.get_annotation('elecGroupSessName')
    electrodeGroup_Name = elecGroupSessName[elecGroupSessName.index('-raw-')+1:]

    electrodeGroup_Name_in_NWB = [name for name in nwbPrepro.electrode_groups.keys() if name in electrodeGroup_Name]
    if len(electrodeGroup_Name_in_NWB) != 1:
        print('Electrode Group {} was detected {} times. It must match with a unique group name from the available electrode_groups in the NWB file : \n{}'.format(
            electrodeGroup_Name, len(electrodeGroup_Name_in_NWB), nwbPrepro.electrode_groups.keys()
        ))
        # Close NWB
        nwbPrepro_io.close()
        del nwbPrepro_io, nwbPrepro
        # Close all recording segments
        for segment_rec in si_recording._recording_segments:
            segment_rec.file.close()
        del si_recording
        raise Exception('Multiple electrode groups with the same name')
    
    electrodeGroup_Name_in_NWB = electrodeGroup_Name_in_NWB[0]

    channel_names = si_recording.get_channel_ids().astype("str", copy=False)
    recording_channels = [f"{ch_}_{electrodeGroup_Name_in_NWB}"for ch_ in channel_names]

    channel_indices_extra = [index for index, key in enumerate(recording_channels) if key not in preproNWB_channels]

    if len(channel_indices_extra)>0:
        print('Recording electrode-group IDs:\n\t{}\nPREPRO electrode-group IDs:\n\t{}'.format(recording_channels, preproNWB_channels))
        for i in channel_indices_extra:
            print('si_recording.channel_name = {} NOT FOUND in preproNWB.electrodes ¡¡¡'.format(recording_channels[i]))
        # Close NWB
        nwbPrepro_io.close()
        del nwbPrepro_io, nwbPrepro
        # Close all recording segments
        for segment_rec in si_recording._recording_segments:
            segment_rec.file.close()
        del si_recording
        raise Exception('Unmatch number of channels')
    
    ##############################################################################################################
    # Check if Device is a valid name:
    nameDevice = nwbPrepro.electrode_groups[electrodeGroup_Name_in_NWB].device.name.upper()
    validDevice = any([prefix in nameDevice for prefix in supported_probes_manufacturer])

    if not validDevice:
        if verbose:
            print('WARNING:\nElectrode Group: {} has an INVALID device name: {}\nvalid Device must start with: {}\nProcessed FilteredEphys will NOT be included'.format(
                    electrodeGroup_Name_in_NWB, nameDevice, supported_probes_manufacturer))
    
    deviceDescription = nwbPrepro.electrode_groups[electrodeGroup_Name_in_NWB].description

    # Close NWB
    nwbPrepro_io.close()
    del nwbPrepro_io, nwbPrepro
    # Close all recording segments
    for segment_rec in si_recording._recording_segments:
        segment_rec.file.close()
    del si_recording

    return nwbPrepro_path, electrodeGroup_Name_in_NWB, validDevice, deviceDescription # preproNWB_channels, recording_channels

##############################################################################################################
# Check whether a electrica series (FilteredEphys or LFP) from the "processing/ecephys/(LFP | Processed)" module exists:
# NOTE: module names match NEUROCONV naming
def _exists_electrical_series_in_nwb_ecephys_module(nwb_path, processing_module, electrical_series_name):

    assert processing_module.lower() in [
        "processed",
        "lfp",
    ], f"'processing_module' should be'processed' or 'lfp', but instead received value {processing_module.lower()}"

    if processing_module.lower()=='lpf':
        processing_module_name = "LFP"
    else:
        processing_module_name = "Processed"
    
    es_exists = False

    ######################################################################################################################
    # Check if electrical series exists
    nwb_io = NWBHDF5IO(nwb_path, mode="r")
    nwbfile = nwb_io.read()

    # Check if default "ecephys" processing module exists
    if "ecephys" not in nwbfile.processing:
        message_exists = '"ecephys" module do NOT exist in the /processing module¡'
    else:
        # Check if requested processing module exists
        if processing_module_name not in nwbfile.processing["ecephys"].data_interfaces:
            message_exists = '"{}" module do NOT exist in "/processing/ecephys/ module path"¡'.format(processing_module_name)
        else:
            if electrical_series_name not in nwbfile.processing["ecephys"][processing_module_name].electrical_series:
                message_exists = '"{}" electrical series do NOT exist in "/processing/ecephys/{} module path"¡'.format(electrical_series_name, processing_module_name)
            else:
                es_exists = True
                message_exists = '"{}" electrical series founded in: "/processing/ecephys/{}"¡'.format(electrical_series_name, processing_module_name)

    nwb_io.close()
    del nwb_io, nwbfile

    return es_exists, message_exists

##############################################################################################################
# Remove a electrica series (FilteredEphys or LFP) from the "processing/ecephys" module:
# STEPS:
# 1) Create a temporal subfolder where the NWBfile exists: ..\nwbfileName_copy\"
# 2) Whithin the temporal folder create a copy of the preproFile without the electrical series of interest
# 3) Delete the original Pepro
# 4) Move the prepro-copy to the original folder
# NOTE: module names match NEUROCONV naming
def _remove_electrical_series_from_nwb_ecephys_module(nwb_path, processing_module, electrical_series_name, verbose=True):

    ######################################################################################################################
    # Check if electrical series exists
    es_exists, message_exists = _exists_electrical_series_in_nwb_ecephys_module(nwb_path, processing_module, electrical_series_name)
    
    # Format proceesing module name:
    if processing_module.lower()=='lpf':
        processing_module_name = "LFP"
    else:
        processing_module_name = "Processed"

    
    if not es_exists:
        raise Exception(message_exists)
    else:
        if verbose:
            print('Removing {} from nwbPrepro.........'.format(electrical_series_name))

        ##############################################################################################################
        # Create temporal folder
        nwb_parentFoder, nwb_fileName = os.path.split(nwb_path)
        nwb_fileNameSplit = os.path.splitext(nwb_fileName)
        nwb_fileNamePrefix = nwb_fileNameSplit[0]
        del nwb_fileNameSplit

        temporal_folder = os.path.join(nwb_parentFoder, nwb_fileNamePrefix + '_copy')
        os.makedirs(temporal_folder)
        if verbose:
            print('Temporal folder {} was created....'.format(temporal_folder))
        
        del nwb_parentFoder

        # Remove item
        nwb_io = NWBHDF5IO(nwb_path, mode="r")
        nwbfile = nwb_io.read()

        nwbfile.processing["ecephys"][processing_module_name].electrical_series.pop(electrical_series_name)

        # generate a new set of object IDs
        nwbfile.generate_new_id()

        ######################################################################################################################
        # Export PREPRO file
        temporal_nwb_path = os.path.join(temporal_folder, nwb_fileNamePrefix + '.nwb')

        with NWBHDF5IO(temporal_nwb_path, "w") as io:
            if verbose:
                print('\nRewriting prepro NWB file: {}\n..........\n\n'.format(nwb_fileNamePrefix))
            io.export(src_io = nwb_io, nwbfile = nwbfile)

        nwb_io.close()
        del nwb_io, nwbfile
        
        # Delete Original nwbfile
        os.remove(nwb_path)
        # Move exported nwb to the original path
        shutil.move(temporal_nwb_path, nwb_path, copy_function = shutil.copy2)
        # Delete temporal folder
        os.rmdir(temporal_folder)

        del nwb_fileNamePrefix, temporal_folder, temporal_nwb_path

##############################################################################################################
# Check whether a given UNITS table exists in the "processing/ecephys/units" :
# NOTE: module names match NEUROCONV naming
def _exists_units_table_in_nwb_ecephys_module(nwb_path, units_table_name):
    
    units_exists = False

    ######################################################################################################################
    # Check if electrical series exists
    nwb_io = NWBHDF5IO(nwb_path, mode="r")
    nwbfile = nwb_io.read()

    # Check if default "units" processing module exists
    if "units" not in nwbfile.processing:
        message_exists = '"units" module do NOT exist in "/processing/ module path"¡'
    else:
        if units_table_name not in nwbfile.processing["units"].data_interfaces:
            message_exists = '"{}" table do NOT exist in "/processing/units module path"¡'.format(units_table_name)
        else:
            units_exists = True
            message_exists = 'UNITS table = "{}" was found in "/processing/units/"'.format(units_table_name)

    nwb_io.close()
    del nwb_io, nwbfile

    return units_exists, message_exists

##############################################################################################################
# Remove a units table from the "processing/ecephys/units" module:
# STEPS:
# 1) Create a temporal subfolder where the NWBfile exists: ..\nwbfileName_copy\"
# 2) Whithin the temporal folder create a copy of the preproFile without the electrical series of interest
# 3) Delete the original Pepro
# 4) Move the prepro-copy to the original folder
# NOTE: "units" module is set within preprocessing container by default (method adapted from NEUROCONV)
def _remove_units_table_from_nwb_ecephys_module(nwb_path, units_table_name, verbose=True):

    ######################################################################################################################
    # Check if electrical series exists
    units_exists, message_exists = _exists_units_table_in_nwb_ecephys_module(nwb_path, units_table_name)

    if not units_exists:
        raise Exception(message_exists)
    
    else:
        if verbose:
            print('Removing {} table from nwbPrepro.........'.format(units_table_name))

        ##############################################################################################################
        # Create temporal folder
        nwb_parentFoder, nwb_fileName = os.path.split(nwb_path)
        nwb_fileNameSplit = os.path.splitext(nwb_fileName)
        nwb_fileNamePrefix = nwb_fileNameSplit[0]
        del nwb_fileNameSplit

        temporal_folder = os.path.join(nwb_parentFoder, nwb_fileNamePrefix + '_copy')
        os.makedirs(temporal_folder)
        if verbose:
            print('Temporal folder {} was created....'.format(temporal_folder))
        
        del nwb_parentFoder

        # Remove item
        nwb_io = NWBHDF5IO(nwb_path, mode="r")
        nwbfile = nwb_io.read()

        nwbfile.processing["units"].data_interfaces.pop(units_table_name)

        # generate a new set of object IDs
        nwbfile.generate_new_id()

        ######################################################################################################################
        # Export PREPRO file
        temporal_nwb_path = os.path.join(temporal_folder, nwb_fileNamePrefix + '.nwb')

        with NWBHDF5IO(temporal_nwb_path, "w") as io:
            if verbose:
                print('\nRewriting prepro NWB file: {}\n..........\n\n'.format(nwb_fileNamePrefix))
            io.export(src_io = nwb_io, nwbfile = nwbfile)

        nwb_io.close()
        del nwb_io, nwbfile
        
        # Delete Original nwbfile
        os.remove(nwb_path)
        # Move exported nwb to the original path
        shutil.move(temporal_nwb_path, nwb_path, copy_function = shutil.copy2)
        # Delete temporal folder
        os.rmdir(temporal_folder)

        del nwb_fileNamePrefix, temporal_folder, temporal_nwb_path


############################################################################################################################################################################################################################
############################################################################################################################################################################################################################
# Combine Behavioral Data from *.nev (markers) & *.YAML files into a NWB file 
# file format follow YAULAB convention of the task : visual cues + 2 shakers + footbar + eye tracking + pupil diameter
############################################################################################################################################################################################################################
############################################################################################################################################################################################################################

##############################################################################################################
##############################################################################################################
# Add eye tracking data into the NWBfile using the behaviour module.
def nwb_add_eyeData(nwb_behavior_module, filePath, dictYAML, 
                    nsFile=None,
                    tempFolderPath=None,
                    analogEye = None,
                    eyeTrackID=None,
                    eyeXYComments=None,                    
                    pupilDiameterComments = None,
                    verbose=False
                    ):
    
    if eyeTrackID is None:
        eyeTrackID = labInfo['EyeTrackingInfo']['EyeTracked']
    
    if eyeXYComments is None:
        eyeXYComments = labInfo['EyeTrackingInfo']['XYComments']
    
    if pupilDiameterComments is None:
        pupilDiameterComments = labInfo['EyeTrackingInfo']['PupilDiameterComments']
    
    ###############################################################
    #               Add EYE-PC data -eyeTracking
    ############################################################### 
    exists_PCEYE_containers = False

    eyeData_h5paths = []
    eyeData_h5objs = []

    if os.path.isfile(filePath + '.eye'):

        eyePCstartTime = getEyeStartTime(filePathEYE = filePath + '.eye', verbose=False)

        if eyePCstartTime is not None:

            eyePC_offsetNEV = get_eyePC_offset(dictYAML, eyePCstartTime, nsFile)
            eyePC_aligned2nev = getEyeData(filePathEYE = filePath + '.eye', offsetSecs = eyePC_offsetNEV)

            ###############################################################
            #                      Add PC-eyeTracking
            ############################################################### 
            if verbose:
                print('\nextracting [X, Y] eye Data from PC file.....')
            eyeTrackingPC = SpatialSeries(
                name= eyeTrackID + '_eyePC_XY',
                description = '(horizontal, vertical) {} eye position'.format(eyeTrackID),
                comments = eyeXYComments + ' Data were recorded on the behavioral-PC',
                data = H5DataIO(
                    data = numpy.array([eyePC_aligned2nev['x'], eyePC_aligned2nev['y']]).transpose(), 
                    compression = True
                    ),
                reference_frame = "(0, 0) is the center of animal's eye gaze",
                timestamps = eyePC_aligned2nev['time'],
                unit = 'degrees',
                )

            eyeTracking = EyeTracking(spatial_series=eyeTrackingPC)
            del eyeTrackingPC

            ###############################################################
            #                   Add PC-pupilDiamater
            ###############################################################
            if verbose:
                print('\nextracting pupil diameter from PC file.....')
            pupilPC = TimeSeries(
                name = eyeTrackID + '_pupilPC',
                description = 'pupil diameter in pixels units. Obtained by video recording of the {} eye'.format(eyeTrackID),
                comments = pupilDiameterComments + ' Data were recorded on the behavioral-PC',
                data = H5DataIO(
                    data = eyePC_aligned2nev['pupil'], 
                    compression = True
                    ),
                timestamps = eyePC_aligned2nev['time'],
                unit = 'pixels',
                continuity = 'continuous',
                )

            pupildiameter = PupilTracking(time_series=pupilPC)
            del pupilPC

            exists_PCEYE_containers = True

    ###############################################################
    #                  Add RIPPLE-eyeTracking 
    ###############################################################
    exists_nsEYE_tracking = False
    if nsFile is not None:
        extractAnalogEye_horizontal = False
        if analogEye is None:
            extractAnalogEye_horizontal = True
        else:
            extractAnalogEye_horizontal = analogEye['exists_eyeHorizontal']

        if extractAnalogEye_horizontal:

            nsEye_horizontal = AnalogIOchannel(nsFile=nsFile, chanName='eyeHorizontal')
            nsEye_horizontal_INFO = nsEye_horizontal.get_info()

            if verbose:
                print('\nextracting [X] eyeData from Ripple.....')

            eye_h5path = temp_TimeSeries_hdf5(nsFile, 
                            entityIndexes=[nsEye_horizontal_INFO['index']], 
                            itemCount=nsEye_horizontal_INFO['item_count'], 
                            tempFolderPath = tempFolderPath,
                            tempName='eyeHorizontal', 
                            verbose=verbose)
            
            eye_h5 = h5py.File(name=eye_h5path, mode="r")

            eyeTrackingNS = SpatialSeries(
                name= eyeTrackID + '_eyeRipple_X',
                description = nsEye_horizontal_INFO['description'] + '({} eye)'.format(eyeTrackID),
                comments = eyeXYComments + ' Data were recorded on Ripple System (chanName:' + nsEye_horizontal_INFO['chanName'] + ')',
                data = H5DataIO(eye_h5["dataSet"]),
                reference_frame = "(0) is the horizontal-center of animal's eye gaze",
                starting_time = 0.0,
                rate = nsEye_horizontal_INFO['samplingRate'],
                unit = nsEye_horizontal_INFO['units'],
                conversion = nsEye_horizontal_INFO['convertionFactor'],
                )
            
            eyeData_h5paths.append(eye_h5path)
            eyeData_h5objs.append(eye_h5)

            if exists_PCEYE_containers:
                eyeTracking.add_spatial_series(spatial_series=eyeTrackingNS)
            else:
                eyeTracking = EyeTracking(spatial_series=eyeTrackingNS)

            del eyeTrackingNS

            exists_nsEYE_tracking = True

    if exists_PCEYE_containers or exists_nsEYE_tracking:
        nwb_behavior_module.add(eyeTracking)

        del eyeTracking

    ###############################################################
    #                    Add Ripple-pupilDiamater
    ###############################################################
    exists_nsPUPIL_tracking = False
    if nsFile is not None:
        extractAnalogEye_pupil = False
        if analogEye is None:
            extractAnalogEye_pupil = True
        else:
            extractAnalogEye_pupil = analogEye['exists_eyePupil']

        if extractAnalogEye_pupil:

            nsPupilDiameter = AnalogIOchannel(nsFile=nsFile, chanName='pupilDiameter')
            nsPupilDiameter_INFO = nsPupilDiameter.get_info()

            if verbose:
                print('\nextracting pupil diameter from Ripple.....')
            
            pupil_h5path = temp_TimeSeries_hdf5(nsFile, 
                            entityIndexes=[nsPupilDiameter_INFO['index']],  
                            itemCount=nsPupilDiameter_INFO['item_count'], 
                            tempFolderPath = tempFolderPath,
                            tempName='pupilDiameter',
                            verbose=verbose)
            
            pupil_h5 = h5py.File(name=pupil_h5path, mode="r")

            pupilNS = TimeSeries(
                name = eyeTrackID + '_pupilRipple',
                description = nsPupilDiameter_INFO['description'] +'({} eye'.format(eyeTrackID),
                comments = pupilDiameterComments + ' Data were recorded on Ripple System (chanName:' + nsPupilDiameter_INFO['chanName'] + ')',
                data = H5DataIO(pupil_h5["dataSet"]),
                starting_time = 0.0,
                rate = nsPupilDiameter_INFO['samplingRate'],
                unit = nsPupilDiameter_INFO['units'],
                conversion = nsPupilDiameter_INFO['convertionFactor'],
                )
            
            eyeData_h5paths.append(pupil_h5path)
            eyeData_h5objs.append(pupil_h5)

            if exists_PCEYE_containers:
                pupildiameter.add_timeseries(time_series=pupilNS)
            else:
                pupildiameter = PupilTracking(time_series=pupilNS)

            del pupilNS

            exists_nsPUPIL_tracking = True

    if exists_nsPUPIL_tracking or exists_PCEYE_containers:
        nwb_behavior_module.add(pupildiameter)

        del pupildiameter
    
    return eyeData_h5objs, eyeData_h5paths


##############################################################################################################
##############################################################################################################
# Add ALL the analog INPUTS related to Stimuli (Except reward)
def nwb_add_nsAnalog_stimuli(nwbFile, dictYAML, nsFile, tempFolderPath,
                analogAccl=None, 
                analogFix=None, 
                analogVisualEvents=None, 
                analogTemp = None,
                verbose=False):
    
    analogStim_h5objs = []
    analogStim_h5paths = []

    # Extract Shaker Command and Accelerometer
    if analogAccl is not None:
        analogAccl['shakerInfo'] = expYAML.getTactorInfo(dictYAML)
        if analogAccl['exists']:
            shakerDict = {
                'leftCommand' : None, 
                'rightCommand' : None, 
                'leftAccelerometer' : analogAccl['shakerInfo']['leftAcclSensitivity'], 
                'rightAccelerometer' : analogAccl['shakerInfo']['rightAcclSensitivity']
                }
            for chanName, sensitivity in shakerDict.items():

                if verbose:
                    print('\nextracting Analog "{}" signal.............'.format(chanName))

                nsCommand = AnalogIOchannel(nsFile=nsFile, chanName=chanName,
                                                        acclSensitivity=sensitivity)
                nsCommand_INFO = nsCommand.get_info()

                nsCommand_h5path = temp_TimeSeries_hdf5(nsFile, 
                            entityIndexes=[nsCommand_INFO['index']], 
                            itemCount=nsCommand_INFO['item_count'], 
                            tempFolderPath = tempFolderPath,
                            tempName=chanName, 
                            verbose=verbose)
            
                nsCommand_h5 = h5py.File(name=nsCommand_h5path, mode="r")

                nwbFile.add_stimulus(TimeSeries(
                        name = nsCommand_INFO['chanName'],
                        description = nsCommand_INFO['description'],
                        data =  H5DataIO(nsCommand_h5["dataSet"]),
                        starting_time = 0.0,
                        rate = nsCommand_INFO['samplingRate'],
                        unit = nsCommand_INFO['units'],
                        conversion = nsCommand_INFO['convertionFactor'],
                        ))
                del nsCommand, nsCommand_INFO

                analogStim_h5objs.append(nsCommand_h5)
                analogStim_h5paths.append(nsCommand_h5path)

    if analogFix is not None:
        if analogFix['exists']:

            if verbose:
                    print('\nextracting Analog "fixON" signal............')

            nsCommand = AnalogIOchannel(nsFile=nsFile, chanName='fixON')
            nsCommand_INFO = nsCommand.get_info()

            nsCommand_h5path = temp_TimeSeries_hdf5(nsFile, 
                            entityIndexes=[nsCommand_INFO['index']], 
                            itemCount=nsCommand_INFO['item_count'], 
                            tempFolderPath = tempFolderPath,
                            tempName='fixON',
                            verbose=verbose)
            
            nsCommand_h5 = h5py.File(name=nsCommand_h5path, mode="r")
            
            nwbFile.add_stimulus(TimeSeries(
                name = nsCommand_INFO['chanName'],
                description = nsCommand_INFO['description'],
                data =  H5DataIO(nsCommand_h5["dataSet"]),
                starting_time = 0.0,
                rate = nsCommand_INFO['samplingRate'],
                unit = nsCommand_INFO['units'],
                conversion = nsCommand_INFO['convertionFactor'],
                ))
            
            del nsCommand, nsCommand_INFO

            analogStim_h5objs.append(nsCommand_h5)
            analogStim_h5paths.append(nsCommand_h5path)
    
    if analogVisualEvents is not None:
        if analogVisualEvents['exists']:

            if verbose:
                    print('\nextracting Analog "visualON" signal............')
            
            nsCommand = AnalogIOchannel(nsFile=nsFile, chanName='visualON')
            nsCommand_INFO = nsCommand.get_info()

            nsCommand_h5path = temp_TimeSeries_hdf5(nsFile, 
                            entityIndexes=[nsCommand_INFO['index']],  
                            itemCount=nsCommand_INFO['item_count'], 
                            tempFolderPath = tempFolderPath,
                            tempName='visualON',
                            verbose=verbose)
            
            nsCommand_h5 = h5py.File(name=nsCommand_h5path, mode="r")

            nwbFile.add_stimulus(TimeSeries(
                name = nsCommand_INFO['chanName'],
                description = nsCommand_INFO['description'],
                data =  H5DataIO(nsCommand_h5["dataSet"]),
                starting_time = 0.0,
                rate = nsCommand_INFO['samplingRate'],
                unit = nsCommand_INFO['units'],
                conversion = nsCommand_INFO['convertionFactor'],
                ))
            
            del nsCommand, nsCommand_INFO

            analogStim_h5objs.append(nsCommand_h5)
            analogStim_h5paths.append(nsCommand_h5path)

    if analogTemp is not None:

        if analogTemp['exists']:

            if verbose:

                print('\nextracting Thermistor signals............')

            temp_chanNames = []
            descriptionTemp = 'Thermistor descriptions:'
            itemsTempCount = 0
            tempInfoSet = False

            for tempID in analogTemp['thermistorIDs']:

                temp_chanNames.append(tempID)

                analogTemp_info = AnalogIOchannel(nsFile=nsFile, chanName=tempID).get_info()

                if not tempInfoSet:

                    itemsTempCount = analogTemp_info['item_count']
                    rateTemp = analogTemp_info['samplingRate']
                    unitsTemp = analogTemp_info['units']
                    convFactorTemp = analogTemp_info['convertionFactor']

                    tempInfoSet = True
                
                descriptionTemp += '\n{}'.format(analogTemp_info['description'])

                del analogTemp_info

            
            nsCommand_h5path = temp_TimeSeries_hdf5_analog_cls(
                nsFile, 
                analog_chanNames = temp_chanNames, 
                itemCount = itemsTempCount, 
                tempFolderPath = tempFolderPath,
                tempName = 'thermistors',
                verbose=True)
                
            nsCommand_h5 = h5py.File(name=nsCommand_h5path, mode="r")

            nwbFile.add_stimulus(
                TimeSeries(
                    name = 'thermistors',
                    description = descriptionTemp,
                    data =  H5DataIO(nsCommand_h5["dataSet"]),
                    starting_time = 0.0,
                    rate = rateTemp,
                    unit = unitsTemp,
                    conversion = convFactorTemp,
                )
            )

            analogStim_h5objs.append(nsCommand_h5)
            analogStim_h5paths.append(nsCommand_h5path)
            

    return analogStim_h5objs, analogStim_h5paths


##############################################################################################################
##############################################################################################################
# Add FootBar Aanalog signals and FootbarEvents.
def nwb_add_footData(nwbFile, nsFile, tempFolderPath,
                    analogFeet=None,
                    verbose = True):
    
    feet_h5objs = []
    feet_h5paths = []

    if analogFeet is not None:

        if analogFeet['exists']:
            if verbose:
                print('\nextracting Analog "FootBar" signal............')
            
            nsLeftFeet = AnalogIOchannel(nsFile=nsFile, chanName='leftFoot')
            nsLeftFeet_INFO = nsLeftFeet.get_info()
            nsRightFeet = AnalogIOchannel(nsFile=nsFile, chanName='rightFoot')
            nsRightFeet_INFO = nsRightFeet.get_info()

            feet_h5path = temp_TimeSeries_hdf5(nsFile, 
                            entityIndexes=[nsLeftFeet_INFO['index'], nsRightFeet_INFO['index']], 
                            itemCount=nsLeftFeet_INFO['item_count'], 
                            tempFolderPath = tempFolderPath,
                            tempName='FootBar', 
                            verbose=verbose)
            
            feet_h5 = h5py.File(name=feet_h5path, mode="r")

            footBarPosition = Position(
                name='FootPosition',
                spatial_series = SpatialSeries(
                    name= 'LeftRight_footBar',
                    description = "(Left, Right) 5V signal wich indicates that subject's feet are not holding the footbar",
                    comments = nsLeftFeet_INFO['description'] + ' ' + nsRightFeet_INFO['description'],
                    data = H5DataIO(feet_h5["dataSet"]),
                    reference_frame = "0V indicates holding, 5V indicates release",
                    starting_time = 0.0,
                    rate = nsLeftFeet_INFO['samplingRate'],
                    unit = nsLeftFeet_INFO['units'],
                    conversion = nsLeftFeet_INFO['convertionFactor'],
                    )
                )
            
            nwbFile.processing['behavior'].add(footBarPosition)

            feet_h5objs.append(feet_h5)
            feet_h5paths.append(feet_h5path)

            ###############################################################
            # ADD FootBar Epochs/Intervals
            ###############################################################
            if verbose:
                print('\nextracting "FootBar" INTERVALS ............')
            
            dictFootEvents = get_feetEvents(nsFile, 
                            chunkSizeSecs = analogFeet['chunkSizeSecs'], 
                            showPlot = analogFeet['showPlot'])
            
            feet_events = TimeIntervals(
                name="feetEvents",
                description="intervals for each foot response: (release Left, release Right, release Both, hold Both)",
            )
            
            ###############################################################
            # first: add colum names
            feet_events.add_column(name='feetResponseID', description="Numerical-ID of the foot Response: [0, 1, 2, 3]")
            feet_events.add_column(name='feetResponse', description="Description of the foot Response: ['holdBoth', 'releaseLeft', 'releaseRight', 'releaseBoth']")
            
            ###############################################################
            # second: ADD FootEvents
            for n in range(len(dictFootEvents)):
                feet_events.add_row(**dictFootEvents[n])
            
            nwbFile.add_time_intervals(feet_events)
    
    return feet_h5objs, feet_h5paths

##############################################################################################################
##############################################################################################################
def nwb_add_rewardData(nwbFile, nsFile, tempFolderPath,
                analogReward=None,
                verbose = True):
    
    reward_h5objs = [] 
    reward_h5paths = []

    if analogReward is not None:
        if analogReward['exists']:
            if verbose:
                    print('\nextracting Analog "rewardON" signal............')

            analogReward_cls = AnalogIOchannel(nsFile=nsFile, chanName='rewardON')
            analogReward_cls_INFO = analogReward_cls.get_info()

            rw_h5path = temp_TimeSeries_hdf5(nsFile, 
                            entityIndexes=[analogReward_cls_INFO['index']], 
                            itemCount=analogReward_cls_INFO['item_count'], 
                            tempFolderPath = tempFolderPath,
                            tempName='rewardON',
                            verbose=verbose)
            
            rw_h5 = h5py.File(name=rw_h5path, mode="r")

            nwbFile.add_stimulus(TimeSeries(
                name = analogReward_cls_INFO['chanName'],
                description = analogReward_cls_INFO['description'],
                data = H5DataIO(rw_h5["dataSet"]),
                starting_time = 0.0,
                rate = analogReward_cls_INFO['samplingRate'],
                unit = analogReward_cls_INFO['units'],
                conversion = analogReward_cls_INFO['convertionFactor'],
                ))
            
            reward_h5objs.append(rw_h5) 
            reward_h5paths.append(rw_h5path)

            ###############################################################
            # ADD REWARD Epochs/Intervals
            ###############################################################
            if verbose:
                print('\nextracting "Reward" INTERVALS ............')

            analogReward_cls = AnalogIOchannel(nsFile=nsFile, chanName='rewardON')

            dictRewardEvents = get_rewardEvents(analogReward_cls, 
                            chunkSizeSecs = analogReward['chunkSizeSecs'], 
                            showPlot = analogReward['showPlot'])
            
            if len(dictRewardEvents)>0:
                            
                reward_events = TimeIntervals(
                    name="rewardEvents",
                    description=  "intervals when the reward was ON: (It is to include those times when the reward was delivered manually)",
                )
                
                ###############################################################
                # first: add colum names
                reward_events.add_column(name='label', description="Description of the reward event: rewardON")
                reward_events.add_column(name='labelID', description="Numerical-ID of the reward event: 1")
                
                
                ###############################################################
                # second: ADD FootEvents
                for n in range(len(dictRewardEvents)):
                    reward_events.add_row(**dictRewardEvents[n])
                
                nwbFile.add_time_intervals(reward_events)
    
    return reward_h5objs, reward_h5paths


##############################################################################################################
##############################################################################################################
def nwb_add_electrodeTable(nwbFile, dictYAML, nsFile, expDay_log=None, verbose=False):

    electrodesDict, electrodeGroups, devices, receptiveFieldsInfo = getNWB_rawElectrodes(dictYAML, nsFile, expDay_log=expDay_log, verbose=verbose)

    if verbose:
        print('\ncreating Electrode Devices ......')

    # create devices
    devicesList = []
    devicesNames = []
    for d in devices:
        devicesList.append(
            nwbFile.create_device(
                name= d['name'],
                description = d['description'],
                manufacturer = d['manufacturer']
                )
            )
        devicesNames.append(d['name'])

    if verbose:
        print('\ncreating Electrode Groups ......')
    
    elecGroupList=[]
    elecGroupList_id = []
    for g in electrodeGroups:
        elecGroupList.append(nwbFile.create_electrode_group(
            name = g['name'],
            description = g['description'],
            location = g['location'],
            device = devicesList[devicesNames.index(g['deviceName'])],
            position = g['position']
        ))
        elecGroupList_id.append(g['group_id'])

    # Create new column Names for ElectrodeTable
    #
    # nsKeys2keep = ['entity_type', 'entity_index', 'id', 'port_id', 'frontEnd_id', 'frontEnd_electrode_id',
    #     'units', 'item_count', 'sample_rate', 'resolution', 'probe_info',
    #     'high_freq_corner', 'high_freq_order', 'high_filter_type',
    #     'low_freq_corner', 'low_freq_order', 'low_filter_type',
    #     ]
    # ymlKeys2add = ['deviceName', 'device_id', 'location', 'rel_id',
    #     'x', 'y', 'z', 'rel_x', 'rel_y', 'rel_z',
    #     ]
    # default_NWBelectrodeColumnNames = ['x', 'y', 'z', 'imp', 'location', 'filtering', 'group', 'id',
    #     'rel_x', 'rel_y', 'rel_z', 'reference']
    
    columns2add = {
        'rel_id': 'Electrode ID within the electrode group / device',
        'frontEnd_electrode_id': 'Electrode ID within the FrontEnd where the device was connected',
        'frontEnd_id': "FrontEnd ID where the electrode's device was connected",
        'port_id': "Port ID where the electrode's FrontEnd was connected",
        'high_freq_corner': 'High pass filter corner frequency in Hz', 
        'high_freq_order': 'High pass filter order.', 
        'high_filter_type': "High pass filter type: None, Butterworth, Chebyshev",
        'low_freq_corner': 'Low pass filter corner frequency in Hz',
        'low_freq_order': 'Low pass filter order.', 
        'low_filter_type': "Low pass filter type: None, Butterworth, Chebyshev"
    }
    
    for key, value in columns2add.items():
        nwbFile.add_electrode_column(
            name = key,
            description = value
        )
    
    # UPDATE Receptive Field Columns
    if receptiveFieldsInfo is not None:
        for key, value in receptiveFieldsInfo.items():
            nwbFile.add_electrode_column(
                name = key,
                description = value['description']
            )

    if verbose:
        print('\ncreating Electrode Table ......')

    for e in electrodesDict:

        # UPDATE Receptive Field Columns
        e_cols2add = dict()
        if receptiveFieldsInfo is not None:
            for key in receptiveFieldsInfo.keys():
                e_cols2add.update({key: e[key]})

        nwbFile.add_electrode(
            group = elecGroupList[elecGroupList_id.index(e['group_id'])], 
            id = e['id'], 
            rel_id = e['rel_id'],
            frontEnd_electrode_id = e['frontEnd_electrode_id'],
            frontEnd_id = e['frontEnd_id'],
            port_id = e['port_id'],
            x = e['ap'], 
            y = e['dv'], 
            z = e['ml'], 
            imp = None, 
            location = e['location'], 
            filtering = None, 
            rel_x = e['rel_ap'], 
            rel_y = e['rel_dv'], 
            rel_z = e['rel_ml'], 
            reference = None,
            high_freq_corner = e['high_freq_corner'], 
            high_freq_order = e['high_freq_order'], 
            high_filter_type = e['high_filter_type'],
            low_freq_corner = e['low_freq_corner'],
            low_freq_order = e['low_freq_order'], 
            low_filter_type = e['low_filter_type'],
            **e_cols2add
        )

######################################################################################################################################################################
######################################################################################################################################################################
def nwb_update_electrodeTable_ReceptiveFields(nwbFile, dictYAML, expDay_log, updateRF = False, skipMissing_RF=False, verbose=True):
    
    # YAML file is required to use the function "getElectrodeList_with_ReceptiveField" which matches the electrodes' Receptive Rield information
    # Be sure to use the original dicYAML from which the nwbFile came.
    electrodeInfo_YAML_updated = expYAML.getElectrodeList_with_ReceptiveField(dictYAML=dictYAML, expDay_log= expDay_log, skipMissing_RF=skipMissing_RF)

    # Get electrodes Dictionay
    electrodesYAML = electrodeInfo_YAML_updated['electrodes']

    # Get ReceptiveField's column names & descriptions
    receptiveFieldsInfo = electrodeInfo_YAML_updated['receptiveFieldsInfo']

    ##############################################################################################################
    # If update Receptive field information (possible update of the expDAY_log.xlsx)
    # All the columns will be revisited regardless if they exist
    if updateRF:
        cols2add = list(receptiveFieldsInfo.keys())
    else:
        cols2add = [key for key in receptiveFieldsInfo.keys() if nwbFile.electrodes.colnames.count(key)==0]

    ###############################################################################################################################################################
    # Loop Per Column to create a dictionary containing all the keys needed for calling "DynamicTable.add_colum" function
    cols2add_list = []
    nChans_range = range(len(electrodesYAML))
    for col_name in cols2add:

        # Fields to add_column:
        # name, description, data=[], table=False, index=False, enum=False, col_cls=<class 'hdmf.common.table.VectorData'>, check_ragged=True
        # Default values apply, because it will be search channel by channel to match order
        col_dict = {'name': col_name, 'description': receptiveFieldsInfo[col_name]['description'], 'data': [], 'table': False, 'index': False, 'enum': False, 'check_ragged': True}

        # Loop to match order from NWBfile.electrodes table
        for id in nwbFile.electrodes.id.data[:]:

            e_index = [e_i for e_i in nChans_range if electrodesYAML[e_i]['id']==id]

            if len(e_index)==1:
                col_dict['data'].append(electrodesYAML[e_index[0]][col_name])
            elif len(e_index)==0:
                print('WARNING¡¡\n{} Receptive Field information from Electrode {} was not found.\ninformation will be set to default values:{}'.format(
                    col_name, id, receptiveFieldsInfo[col_name]['defaultvalue']))
                
                col_dict['data'].append(receptiveFieldsInfo[col_name]['defaultvalue'])
            else:
                raise Exception('Multiple indexes for Electrode {} were found (index={}). Must be unique index or zero'.format(id, e_index))
        
        cols2add_list.append(col_dict)
    
    ##############################################################################################################
    # Loop to add DATA (dict) per column
    for col in cols2add_list:

        # If column already exist, check previous data.type matches new data.type
        if col['name'] in nwbFile.electrodes.colnames:

            # Check the number of elements match the number of electrodes (receptive field values is a list of length = numChans)
            # Ragged arrays have different lengths. Receptive field attibutes are single values per channel:
            if nwbFile.electrodes[col['name']].data.shape[0]!=nwbFile.electrodes.id.shape[0]:
                raise Exception('column: "{}" (shape={}) should match the number of electrodes (chans={}).\n\tNOTE: ragged arrays are NOT supported to update Receptive Fields)'.format(
                    col['name'], nwbFile.electrodes[col['name']].data.shape, nwbFile.electrodes.id.shape[0]))
            else:
                if verbose:
                    print('Column: "{}" already exists¡¡\nUPDATING their values¡¡\n'.format(col['name']))

                # ReWrite data per channel
                for i in range(nwbFile.electrodes.id.shape[0]):

                    # Check the element is the same type: Either Numerical or String:
                    # receptiveField_n = float
                    # BodySide_"n" = str
                    # BodyPart_"n" str
                    # Modality_"n" str
                    type_match = False
                    if col['name'].lower()=="receptivefield_n":
                        if isinstance(nwbFile.electrodes[col['name']].data[0], numpy.floating) and isinstance(col['data'][i], float):
                            type_match = True
                    else:
                        if isinstance(nwbFile.electrodes[col['name']].data[i], str) and isinstance(col['data'][i], str):
                            type_match = True
                    
                    if type_match:
                            nwbFile.electrodes[col['name']].data[i] = col['data'][i]
                    else:
                        raise Exception('To update Receptive fields, data must be integer or string inputs:\n\tColumn: {}\n\tOriginal type: {}\n\tSustitucion type: {}'.format(
                        col['name'],  type(nwbFile.electrodes[col['name']].data[i]), type(col['data'][i])))
                
        else:
            if verbose:
                print('ADDING NEW column: {} ¡¡'.format(col['name']))
            nwbFile.add_electrode_column(**col)



##############################################################################################################
##############################################################################################################
def nwb_add_rawElectrode(nwbFile, nsFile, dictYAML, tempFolderPath, expDay_log=None, verbose=False):

    if nwbFile.electrodes is None:
        nwb_add_electrodeTable(nwbFile, dictYAML, nsFile, expDay_log=expDay_log, verbose=verbose)
    
    electrodesDict, _, _, _ = getNWB_rawElectrodes(dictYAML, nsFile, expDay_log=expDay_log, verbose=False)

    print('\nAdding Neural Data...\n')

    electrodes_h5objs = []
    electrodes_h5paths = []

    for i in range(len(electrodesDict)):

        e = electrodesDict[i]
        e_h5path = temp_TimeSeries_hdf5(
                                nsFile = nsFile, 
                                entityIndexes= [e['entity_index']],  
                                itemCount= int(e['item_count']),
                                tempFolderPath = tempFolderPath,
                                tempName= '{}{}-{}-raw{}'.format(
                                    e['port_id'], 
                                    e['frontEnd_id'], 
                                    e['frontEnd_electrode_id'], 
                                    e['id']
                                    ),
                                verbose = verbose,
                    )

        e_h5 = h5py.File(name=e_h5path, mode="r")

        electrodes_h5objs.append(e_h5)
        electrodes_h5paths.append(e_h5path)
        
        if e['units']=='mV':
            convertion2V = 1/1000
        elif e['units']=='V':
            convertion2V = 1.0
        elif e['units']=='uV':
            convertion2V = 1/1000000


        nwbFile.add_acquisition(
            ElectricalSeries(
                name = 'raw-{}{}-{}-{}'.format(e['port_id'], e['frontEnd_id'], e['frontEnd_electrode_id'], e['id']),
                data = H5DataIO(e_h5["dataSet"]),
                electrodes = DynamicTableRegion(name='electrodes', data=[i], description=e['probe_info'], table=nwbFile.electrodes),  
                filtering = '{} HighFreq Filter (order:{}, freq-corner:{}); {} LowFreq Filter (order:{}, freq-corner:{})'.format(
                        e['high_filter_type'], e['high_freq_order'], e['high_freq_corner'],
                        e['low_filter_type'], e['low_freq_order'], e['low_freq_corner']
                        ), 
                resolution=e['resolution']*convertion2V, 
                conversion=convertion2V,  
                starting_time=0.0, 
                rate=e['sample_rate'], 
                comments='This ElectricalSeries corresponds to the raw neural data collected with Trellis software (Ripple)', 
                description=e['probe_info']
            ))
        
    return electrodes_h5objs, electrodes_h5paths


##############################################################################################################
##############################################################################################################
def nwb_add_rawElectrodeGroup(nwbFile, nsFile, dictYAML, tempFolderPath, expDay_log=None, verbose=False):

    if nwbFile.electrodes is None:
        nwb_add_electrodeTable(nwbFile, dictYAML, nsFile, expDay_log=expDay_log, verbose=verbose)
    
    electrodesDict, electrodeGroups, _, _ = getNWB_rawElectrodes(dictYAML, nsFile, expDay_log=expDay_log, verbose=False)

    print('\nAdding Neural Data...\n')

    electrodeGroups_h5objs = []
    electrodeGroups_h5paths = []

    for eg in electrodeGroups:

        group_h5path, groupInfo = create_rawElectrodeGroup_hdf5(electrodesDict, nsFile, 
                                                                groupID=eg['group_id'],
                                                                tempFolderPath = tempFolderPath, 
                                                                verbose=True)

        if verbose:
            print('Importing..... : raw signal from {}\n'.format(eg['name']))

        group_h5 = h5py.File(name=group_h5path, mode="r")

        if groupInfo['units']=='mV':
            convertion2V = 1/1000
        elif groupInfo['units']=='V':
            convertion2V = 1.0
        elif groupInfo['units']=='uV':
            convertion2V = 1/1000000

        nwbFile.add_acquisition(
            ElectricalSeries(
                name = 'raw-{}'.format(eg['name']),
                data = H5DataIO(group_h5["dataSet"]), 
                electrodes = DynamicTableRegion(name='electrodes', data=groupInfo['electrode_index'], 
                                                description=eg['name'], table=nwbFile.electrodes),  
                filtering = '{} HighFreq Filter (order:{}, freq-corner:{}); {} LowFreq Filter (order:{}, freq-corner:{})'.format(
                        groupInfo['high_filter_type'], groupInfo['high_freq_order'], groupInfo['high_freq_corner'],
                        groupInfo['low_filter_type'], groupInfo['low_freq_order'], groupInfo['low_freq_corner']
                        ), 
                resolution=groupInfo['resolution']*convertion2V, 
                conversion=convertion2V,  
                starting_time=0.0, 
                rate=groupInfo['sample_rate'], 
                comments='This ElectricalSeries corresponds to the raw neural data collected with Trellis software (Ripple)', 
                description=eg['description']
            ))
        
        electrodeGroups_h5objs.append(group_h5)
        electrodeGroups_h5paths.append(group_h5path)
    
    return electrodeGroups_h5objs, electrodeGroups_h5paths

    
##############################################################################################################
##############################################################################################################
def nwb_add_stimElectrodesWaveForms(nwbFile, nsFile, dictYAML, expDay_log=None, verbose=False):

    if nwbFile.electrodes is None:
        nwb_add_electrodeTable(nwbFile, dictYAML, nsFile, expDay_log=expDay_log, verbose=verbose)
    
    electrodesDict, _, _, _ = getNWB_rawElectrodes(dictYAML, nsFile, expDay_log=expDay_log, verbose=False)
    stimElectrodes = get_stimElectrodeInfo(nsFile)

    for i in range(len(stimElectrodes)):

        indexElec = [e for e in range(len(electrodesDict)) if electrodesDict[e]['id']==stimElectrodes[i]['id']]

        if len(indexElec)==0:
            raise Exception('Electrode {} was not found in ElectrodeTable, \nElectrodes ID in Table: \n'.format(
                stimElectrodes[i]['id'], [e['id'] for e in electrodesDict]
                ))
        elif len(indexElec)>1:
            raise Exception('Electrode {} was found more than once in ElectrodeTable, \n Verify Electrodes ID in Table: \n'.format(
                stimElectrodes[i]['id'], [e['id'] for e in electrodesDict]
                ))
            
        # print(electrodesDict[indexElec[0]], '\n')

        chanStim = SegmentStimChannel(nsFile=nsFile, electrode_id=stimElectrodes[i]['id'])
        chanStimInfo = chanStim.get_info()

        if verbose:
            print('\nExtracting microStimulation Waveforms..... : stim-{}{}-{}-{}'.format(
                chanStimInfo['port_id'], chanStimInfo['frontEnd_id'], chanStimInfo['frontEnd_electrode_id'], chanStimInfo['id']
                ))
            
        if chanStimInfo['units']=='mV':
            convertion2V = 1/1000
        elif chanStimInfo['units']=='V':
            convertion2V = 1.0
        elif chanStimInfo['units']=='uV':
            convertion2V = 1/1000000

        # print(chanStimInfo, '\n')

        data = chanStim.get_data(index=range(chanStimInfo['item_count']), verbose=verbose)

        nwbFile.add_stimulus(SpikeEventSeries(
            name='stim-{}{}-{}-{}'.format(chanStimInfo['port_id'], chanStimInfo['frontEnd_id'], chanStimInfo['frontEnd_electrode_id'], chanStimInfo['id']),
            # data = data['waveForms'], 
            # timestamps = data['timeStamps'], 
            data = H5DataIO( data=data['waveForms'], compression = True), 
            timestamps = H5DataIO( data=data['timeStamps'], compression = True), 
            electrodes = DynamicTableRegion(name='electrodes', 
                                            data=indexElec, 
                                            description='This is channel {} ({})'.format(chanStimInfo['id'], chanStimInfo['label_id']), 
                                            table=nwbFile.electrodes),  
            resolution=chanStimInfo['resolution']*convertion2V, 
            conversion=convertion2V, 
            comments="This SpikeEventSeries corresponds to the electrical microstimulation Waveforms delivered through this electrode\nFurther details in Trellis software documentation from Ripple", 
            description= "Microstimulation waveforms from electrode {} ({}).\nSamples per Waveform: [min={}, max={}]\nOriginal amplitude units: {}\nAmplitude Range: [min={}, max={}]".format(
                chanStimInfo['id'], chanStimInfo['probe_info'], chanStimInfo['min_sample_count'], chanStimInfo['max_sample_count'], 
                chanStimInfo['units'], chanStimInfo['min_val'], chanStimInfo['max_val']
                ), 
            control=None, 
            control_description=None, 
            offset=0.0
            ))

    
##############################################################################################################
# Add a spikeinterface - recording object using SpikeInterfaceRecordingDataChunkIterator from neuroconv
# Recording must be already written into a binary folder
# It will run nwb-inspector, print message according to "above_nwbinspectorSeverity" (HIGH = 2, LOW = 1). Set to "0" it will print all
##############################################################################################################
def nwb_add_processed_si_recording_SIdataChunkIterator(si_recording_folder, parentFolder_preproNWB, rewrite=False, above_nwbinspectorSeverity=1, verbose=False):
    
    nwbPrepro_path, electrodeGroup_Name_in_NWB, validDevice, deviceDescription = _validate_si_recording_with_preproNWB(si_recording_folder, parentFolder_preproNWB, verbose=verbose)

    if validDevice:
        
        ##############################################################################################################
        # Check if the "electrical_series" with the name of the electrode group already exists in the module: "processing/ecephys/Processed" 
        validWriting = True
        es_name = 'ephys-{}'.format(electrodeGroup_Name_in_NWB)
        es_exists, message_exists = _exists_electrical_series_in_nwb_ecephys_module(nwb_path=nwbPrepro_path, processing_module="Processed", electrical_series_name=es_name)

        if es_exists and not rewrite:
            validWriting = False
            print('WARNING: {}\nskipping {} ....¡¡'.format(message_exists, es_name))

        ##############################################################################################################
        # If the processed recording already exists and rewrite=True, remove the electrical series 
        elif es_exists and rewrite:
            _remove_electrical_series_from_nwb_ecephys_module(nwb_path=nwbPrepro_path, processing_module="Processed", electrical_series_name=es_name, verbose=verbose)

        ######################################################################################################################
        # Export electrical series
        if validWriting:

            si_recording = load_extractor(si_recording_folder) 

            nwbPrepro_io = NWBHDF5IO(nwbPrepro_path, mode="r+")
            nwbPrepro = nwbPrepro_io.read()

            ##############################################################################################################
            # Create recording processing description :    
            if si_recording.get_annotation('is_referenced'):
                reference_desc = si_recording.get_annotation('reference') + '-' + si_recording.get_annotation('reference_mode') + ' referenced'
            else:
                reference_desc = 'no referenced'

            if si_recording.get_annotation('is_filtered'):
                filtered_desc = 'hig-pass filtered'
            else:
                filtered_desc = 'no filtered'

            if si_recording.get_annotation('is_centered'):
                centered_desc = si_recording.get_annotation('centered_mode') + '-centered'
            else:
                centered_desc = 'no centered'

            if si_recording.get_annotation('recording_sufix'):
                motion_desc = 'Motion estimation and interpolation abbreviation : ' + si_recording.get_annotation('recording_sufix')
            else:
                motion_desc = 'No motion correction was performed'

            es_description = 'Processed {}. {}'.format(electrodeGroup_Name_in_NWB, deviceDescription)
            es_filtering = 'Processed "raw-{}" electrophysiological traces with spiketinterface. Recording is {}, {} and {}. {}.'.format(electrodeGroup_Name_in_NWB, reference_desc, filtered_desc, centered_desc, motion_desc)

            ##############################################################################################################
            # UPDATE THE COLUMN "channel_labes" from the NWB.ELECTRODES TABLE 
            ##############################################################################################################
            nwb_electrodes_ids = list(nwbPrepro.electrodes.to_dataframe().index.values)
            channel_indexes_in_NWB = [nwb_electrodes_ids.index(e) for e in si_recording.channel_ids]

            if si_recording.get_property("channel_labels") is not None:
                # If "channel_labels" does not exists in the NWBfile, add the column with default "not_inspected" values
                if "channel_labels" not in nwbPrepro.electrodes.colnames:
                    nwbPrepro.add_electrode_column(
                        name = "channel_labels", 
                        description = description_channel_labels, 
                        data = ['not_inspected']*len(nwb_electrodes_ids), 
                        table = False, 
                        index = False, 
                        enum = False, 
                        check_ragged = True
                    )
                    
                # Update the current values:
                channel_labels = si_recording.get_property("channel_labels")
                if verbose:
                    print('Updating Electrode-column = "channel_labels" in the NWBfile')
                for ch in range(si_recording.get_num_channels()):
                    nwbPrepro.electrodes["channel_labels"].data[channel_indexes_in_NWB[ch]] = str(channel_labels[ch])

            ##############################################################################################################
            # CREATE THE HDF5 temporal file
            ##############################################################################################################
            # For now only one segment is supported
            if si_recording.get_num_segments()>1:
                raise Exception('Sorting object has more than one segments (n={})\nExporting multisegment sorting is not supported yet¡¡'.format(si_recording.get_num_segments()))
            else:
                segment_index = 0

            # Create Create "nwbPrepro-recording" temporal folder to save hdf5 files with data from units
            # It will use the parent folder of the si_recording folder 
            parentFolder, _ = os.path.split(si_recording_folder)
            _, nwbFileName = os.path.split(nwbPrepro_path)
            nwbFileNameSplit = os.path.splitext(nwbFileName)
            es_folder = os.path.join(parentFolder, '{}-recording'.format(nwbFileNameSplit[0]))
            del parentFolder, nwbFileName, nwbFileNameSplit

            if os.path.isdir(es_folder):  
                if verbose:
                    print('\nTemporal folder for the electrical series = {} already exists. It will be deleted.. \n{}.............\n'.format(es_name, es_folder))
                shutil.rmtree(es_folder, ignore_errors=False)
            
            os.makedirs(es_folder)
            if verbose:
                print('\nTemporal folder for the electrical series = {} was successfully created¡ .............\n'.format(es_name))

            traces_as_iterator = SpikeInterfaceRecordingDataChunkIterator(
                recording = si_recording,
                segment_index = segment_index,
                return_scaled = True, # Whether to return the trace data in scaled units (uV, if True)
                display_progress= False
            )

            tempTraces_hdf5_path = temp_hdf5_from_HDMFGenericDataChunkIterator(
                HDMFGenericDataChunkIterator = traces_as_iterator, 
                tempFolderPath = es_folder, 
                tempName = es_name,
                verbose=verbose
                )
            
            del traces_as_iterator

            ##############################################################################################################
            # ADD ELECTRICAL SERIES:
            ##############################################################################################################
            if verbose:
                print('\nAdding Electrical series into NWB...........\n')
            if "ecephys" not in nwbPrepro.processing:
                nwbPrepro.create_processing_module(name="ecephys", description = description_ecephys)

            if "Processed" not in nwbPrepro.processing["ecephys"].data_interfaces:
                nwbPrepro.processing["ecephys"].add(FilteredEphys(name="Processed"))

            # ADD ElectricalSeries to nwbfile
            tempTraces_h5 = h5py.File(name=tempTraces_hdf5_path, mode="r")
            nwbPrepro.processing["ecephys"]["Processed"].add_electrical_series(
                ElectricalSeries(
                    name = es_name,
                    data = H5DataIO(tempTraces_h5["dataSet"]), 
                    electrodes = DynamicTableRegion(
                        name='electrodes', 
                        data = channel_indexes_in_NWB, 
                        description='electrode_table_region corresponding to the electrical series = {}'.format(es_name), 
                        table=nwbPrepro.electrodes
                        ),  
                    filtering = es_filtering,  
                    conversion = 1e-6,  # micro_to_volts_conversion_factor = 1e-6
                    starting_time = 0.0, 
                    rate = si_recording.get_sampling_frequency(), 
                    comments = comments_ecephys_processed, 
                    description = es_description
                )
            )

            ##############################################################################################################
            # Write-close PREPRO-NWB
            if verbose:
                print('\nWriting Recording into NWB...........\n')
            nwbPrepro_io.write(nwbPrepro)
            nwbPrepro_io.close()

            del nwbPrepro_io, nwbPrepro

            ######################################################################################################################
            # REMOVE TEMPORAL WAVEFORMS FILES
            ######################################################################################################################
            if verbose:
                print('\ndeleting Temprary HDF5 files..........')
            tempTraces_h5.close()
            if verbose:
                print('removing {}'.format(tempTraces_hdf5_path))
            os.remove(tempTraces_hdf5_path)

            if os.path.isdir(es_folder):
                shutil.rmtree(es_folder, ignore_errors=True)

            ###############################################################
            #                 RUN nwbInspector
            ###############################################################
            resultsInspector = list(inspect_nwbfile(nwbfile_path=nwbPrepro_path))
            
            if len(resultsInspector)==0:
                if verbose:
                    print("congrats¡ no NWB inspector comments\n")
            else:
                print('\nNWB inspector comments:\n')
                for r in resultsInspector:
                    if r.severity.value>above_nwbinspectorSeverity:
                        print('Message : ', r.message)
                        print('Object Name : ',r.object_name)
                        print('Object Type : ',r.object_type)
                        print('Severity : ',r.severity)
                        print('Importance : ', r.importance, '\n') 

            # Close all recording segments
            for segment_rec in si_recording._recording_segments:
                segment_rec.file.close()
            del si_recording


##############################################################################################################
# Add a spikeinterface - recording object using Neuroconv 
# Recording must be already written into a binary folder
# It will run nwb-inspector, print message according to "above_nwbinspectorSeverity" (HIGH = 2, LOW = 1). Set to "0" it will print all
# NOTE: module names match must match NEUROCONV naming
##############################################################################################################
def nwb_add_processed_si_recording_neuroconv(si_recording_folder, parentFolder_preproNWB, rewrite=False, verbose=False, above_nwbinspectorSeverity=1):

    nwbPrepro_path, electrodeGroup_Name_in_NWB, validDevice, deviceDescription  = _validate_si_recording_with_preproNWB(si_recording_folder, parentFolder_preproNWB, verbose=False)

    if validDevice:
        
        ##############################################################################################################
        # Check if the "electrical_series" with the name of the electrode group already exists in the module: "processing/ecephys/Processed" 
        validWriting = True
        es_name = 'ephys-{}'.format(electrodeGroup_Name_in_NWB)
        es_exists, message_exists = _exists_electrical_series_in_nwb_ecephys_module(nwb_path=nwbPrepro_path, processing_module="Processed", electrical_series_name=es_name)

        if es_exists and not rewrite:
            validWriting = False
            print('WARNING: {}\nskipping {} ....¡¡'.format(message_exists, es_name))

        ##############################################################################################################
        # If the processed recording already exists and rewrite=True, remove the electrical series 
        elif es_exists and rewrite:
            _remove_electrical_series_from_nwb_ecephys_module(nwb_path=nwbPrepro_path, processing_module="Processed", electrical_series_name=es_name, verbose=verbose)

        ######################################################################################################################
        # Export electrical series
        if validWriting:

            si_recording = load_extractor(si_recording_folder) 

            ##############################################################################################################
            # Create recording processing description :    
            if si_recording.get_annotation('is_referenced'):
                reference_desc = si_recording.get_annotation('reference') + '-' + si_recording.get_annotation('reference_mode') + ' referenced'
            else:
                reference_desc = 'no referenced'

            if si_recording.get_annotation('is_filtered'):
                filtered_desc = 'hig-pass filtered'
            else:
                filtered_desc = 'no filtered'

            if si_recording.get_annotation('is_centered'):
                centered_desc = si_recording.get_annotation('centered_mode') + '-centered'
            else:
                centered_desc = 'no centered'

            if si_recording.get_annotation('recording_sufix'):
                motion_desc = 'Motion estimation and interpolation abbreviation : ' + si_recording.get_annotation('recording_sufix')
            else:
                motion_desc = 'No motion correction was performed'

            metadata = {'Ecephys': dict()}
            metadata = {
                'Ecephys': {
                    es_name : {
                        'name' : es_name,
                        'description': 'Processed {}. {}'.format(electrodeGroup_Name_in_NWB, deviceDescription),
                        'filtering': 'Processed "raw-{}" electrophysiological traces with spiketinterface. Recording is {}, {} and {}. {}.'.format(electrodeGroup_Name_in_NWB, reference_desc, filtered_desc, centered_desc, motion_desc),
                        'comments': comments_ecephys_processed
                    },
                    'Electrodes': [{
                        'name' : "channel_labels",
                        'description': 'If the channel was label as "good"/"dead"/"noise"/"bad" during spikeinterface preprocessing (see: spikeinterface.preprocessing.detect_bad_channels)'
                    }]
                }
            }

            ##############################################################################################################
            # Update recording properties to match NWB electrode table:
            si_recording = udpdate_si_recording_with_nwb_electrodes_table(si_recording, nwbFile_path=nwbPrepro_path, electrodeGroupName=electrodeGroup_Name_in_NWB, verbose=verbose)

            ##############################################################################################################
            # It might always be a single segment, but in this way we follow NEUROCONV pipeline
            number_of_segments = si_recording.get_num_segments()

            for segment_index in range(number_of_segments):

                if verbose:
                    print('adding Processed Ephys to NWB (neuroconv AbstractChunkIterator)...... segement: {}/{}'.format(segment_index+1, number_of_segments))

                ##############################################################################################################
                # Open-read PREPRO-NWB
                nwbPrepro_io = NWBHDF5IO(nwbPrepro_path, mode="r+")
                nwbPrepro = nwbPrepro_io.read()

                ##############################################################################################################
                # USE NEUROCONV
                # Ensure the start time = 0.0, Recording object has been split into sessions, otherwise it will keep the start time relative to concatened sessions
                # It is a way to manually reset to 0.0 based on how the pipeline from NEUROCONV set the starting_time
                recording_t_start = si_recording._recording_segments[segment_index].t_start or 0
                if recording_t_start>0:
                    recording_t_start *= -1

                neuroconv_add_electrical_series_to_nwbfile(
                    recording = si_recording,
                    nwbfile = nwbPrepro,
                    metadata = metadata,
                    segment_index = segment_index,
                    starting_time = recording_t_start,
                    write_as = "processed",
                    es_key = es_name,
                    write_scaled = False,
                    iterator_type = 'v2',
                    always_write_timestamps = False
                    )
                
                nwb_starting_time = nwbPrepro.processing['ecephys']['Processed'].electrical_series[es_name].starting_time
                if nwb_starting_time>0:
                    raise Exception('"starting time" MUST BE ZERO ({}) for the current electrical_series = {}'.format(nwb_starting_time, es_name))
                
                ##############################################################################################################
                # Write-close PREPRO-NWB
                if verbose:
                    print('\nWriting Recording into NWB...........\n')
                nwbPrepro_io.write(nwbPrepro)
                nwbPrepro_io.close()

                del nwbPrepro_io, nwbPrepro, recording_t_start

                ###############################################################
                #                 RUN nwbInspector
                ###############################################################
                resultsInspector = list(inspect_nwbfile(nwbfile_path=nwbPrepro_path))
                
                if len(resultsInspector)==0:
                    if verbose:
                        print("congrats¡ no NWB inspector comments\n")
                else:
                    print('\nNWB inspector comments:\n')
                    for r in resultsInspector:
                        if r.severity.value>above_nwbinspectorSeverity:
                            print('Message : ', r.message)
                            print('Object Name : ',r.object_name)
                            print('Object Type : ',r.object_type)
                            print('Severity : ',r.severity)
                            print('Importance : ', r.importance, '\n')    

            
            # Close all recording segments
            for segment_rec in si_recording._recording_segments:
                    segment_rec.file.close()
            del si_recording


##############################################################################################################
# Filter & Resample RAW data to create LFPs using SpikeInterface 
# Add LFPs using Neuroconv 
# NOTE: module names MUST match NEUROCONV naming
##############################################################################################################
def nwb_add_lfp_from_nwbRaw_neuroconv(nwbRaw, nwbPrepro_path, tempFolderPath, lfp_params_spikeInterface=None, rewrite=False, verbose=False):

    ##############################################################################################################
    # NOTE: It is assume that "nwbPrepro" is a "copy" of the "nwbRaw" except for the raw-aquisition-ephys,
    #       which might be already deleted from the Prepro. Therefore, the electrode table should match
    ##############################################################################################################
    nwbPrepro_io = NWBHDF5IO(nwbPrepro_path, mode="r")
    nwbPrepro = nwbPrepro_io.read()

    if nwbRaw.electrodes is None or nwbPrepro.electrodes is None:
        raise Exception('both NWB files must contain electrode Table')
    elif "group_name" not in nwbRaw.electrodes.colnames or "group_name" not in nwbPrepro.electrodes.colnames:
        raise Exception('both NWB files must contain "group name" in the electrode Table')
    else:
        if "channel_name" not in nwbRaw.electrodes.colnames:
            raw_channel_names = numpy.array(nwbRaw.electrodes.id[:]).astype("int")
        else:
            raw_channel_names = nwbRaw.electrodes["channel_name"][:]
        raw_channels = [f"{ch_name}_{gr_name}" for ch_name, gr_name in zip(raw_channel_names, nwbRaw.electrodes["group_name"][:])]

        if "channel_name" not in nwbPrepro.electrodes.colnames:
            print('prepro NWB files must contain "channel_name" in the electrode Table to be compatible with NEUROCONV, otherwise it will not be able to compare electrodeIDs with spikeinterface')
            raise Exception('NEUROCONV missing argument')
        
        prepro_channel_names = nwbPrepro.electrodes["channel_name"][:]
        prepro_channels = [f"{ch_name}_{gr_name}" for ch_name, gr_name in zip(prepro_channel_names, nwbPrepro.electrodes["group_name"][:])]

        onlyRAW = set(raw_channels).difference(set(prepro_channels))
        onlyPrepro = set(prepro_channels).difference(set(raw_channels))

        invalid_channels = False

        if any(onlyRAW):
            print('There were RAW-electrodes not found in the PREPRO : \n{}'.format(onlyRAW))
            invalid_channels = True
        elif any(onlyPrepro): 
            print('There were PREPRO-electrodes not found in RAW : \n{}'.format(onlyPrepro))
            invalid_channels = True
        
        if invalid_channels:
            print('RAW electrode-group IDs:\n\t{}\nPREPRO electrode-group IDs:\n\t{}'.format(raw_channels, prepro_channels))
            raise Exception('Unmatch number of channels')
        
    nwbPrepro_io.close()
    del nwbPrepro_io, nwbPrepro

    ##############################################################################################################
    # Get LFP params
    ##############################################################################################################
    if lfp_params_spikeInterface is None:
        bandpass_params = lfp_params_spikeInterface_default['bandpass']
        resample_params = lfp_params_spikeInterface_default['resample']
        lfp_description = lfp_params_spikeInterface_default['lfp_description']
    else:
        if 'bandpass' in lfp_params_spikeInterface.keys():
            bandpass_params = lfp_params_spikeInterface['bandpass']
        else:
            bandpass_params = lfp_params_spikeInterface_default['bandpass']

        if 'resample' in lfp_params_spikeInterface.keys():
            resample_params = lfp_params_spikeInterface['resample']
        else:
            resample_params = lfp_params_spikeInterface_default['resample']

        if 'lfp_description' in lfp_params_spikeInterface.keys():
            lfp_description = lfp_params_spikeInterface['lfp_description']
        else:
            lfp_description = lfp_params_spikeInterface_default['lfp_description']

    # Complete "bandpass" parameters
    for key, value in lfp_params_spikeInterface_default['bandpass'].items():
        if key not in bandpass_params.keys():
            bandpass_params.update({key:value})
    # Complete "resample" parameters
    for key, value in lfp_params_spikeInterface_default['resample'].items():
        if key not in resample_params.keys():
            resample_params.update({key:value})

    ##############################################################################################################
    # Extract all "RAW" electrode groups in a LOOP
    # It will open-read-write-close PREPRO-NWB for each electrode to avoid Memory issues (?)
    ##############################################################################################################
    electrode_groups_names = [k for k in nwbRaw.acquisition.keys() if 'raw-' in k]
    metadata = {'Ecephys': dict()}

    for eg in electrode_groups_names:

        ##############################################################################################################
        # Check Device is a valid name:
        # Similar procedure as "yaulab_si.spikeinterface_tools.getProbeInfo"
        electrodeGroup_dataFrame = nwbRaw.acquisition[eg].electrodes.to_dataframe()
        groupName_u = electrodeGroup_dataFrame.group_name.unique()

        # Double check if these electrodes were recorded with the same probe:
        if groupName_u.size>1:
            raise Exception('ElectrodeGroup should have a unique name, but more than one name was found {}'.format(
            groupName_u 
            ))
        
        groupName = groupName_u[0]
        nameDevice = nwbRaw.electrode_groups[groupName].device.name.upper()
        validDevice = any([prefix in nameDevice for prefix in supported_probes_manufacturer])

        ##############################################################################################################
        if not validDevice:
            print(('Warning:\nElectrode Group: {} has an INVALID device name: {}\nvalid Device must start with: {}\nProcessed LFP will NOT be included'.format(
                groupName, nameDevice, supported_probes_manufacturer)))
        
        else:

            ##############################################################################################################
            # Check if the "electrical_series" with the name of the electrode group already exists in the module: "processing/ecephys/LFP" 
            validWriting = True
            es_name = 'lfp-{}'.format(eg[4::])
            lfp_exists, message_exists = _exists_electrical_series_in_nwb_ecephys_module(nwb_path=nwbPrepro_path, processing_module="LFP", electrical_series_name=es_name)

            if lfp_exists and not rewrite:
                validWriting = False
                print('WARNING: {}\nskipping {} ....¡¡'.format(message_exists, es_name))
                
            ##############################################################################################################
            # If the processed LFP already exists and rewrite=True, remove the electrical series 
            elif lfp_exists and rewrite:
                _remove_electrical_series_from_nwb_ecephys_module(nwb_path=nwbPrepro_path, processing_module="LFP", electrical_series_name=es_name, verbose=verbose)
            

            if validWriting:

                metadata['Ecephys'].update(
                    {es_name : {
                            'name' : es_name,
                            'description': 'Local field potentials from {}'.format(nwbRaw.acquisition[eg].description),
                            'filtering': lfp_description,
                            'comments': 'This ElectricalSeries corresponds to the LFP signal sampled at 1KHz. The signal was preprocessed using spikeinterface.preprocessing : bandpass_filter & resample.'
                            }
                    }
                    )
                
                ##############################################################################################################
                # It will use SpikeInterface preprocessing "filter" and "resample"
                if verbose:
                    print('\nSpikeInterface LFP-Preprocessing of electrode group: {}.............'.format(es_name))

                # LOAD recording
                si_recording_raw = se_read_nwb(
                        file_path=nwbRaw.container_source, 
                        electrical_series_path ='acquisition/' + eg,
                        load_recording=True,
                        load_sorting=False
                )

                ##############################################################################################################
                # preprocessing "filter" and "resample"
                si_recording_lfp = bandpass_filter(si_recording_raw, **bandpass_params)
                si_recording_lfp1k = resample(si_recording_lfp, **resample_params)

                ##############################################################################################################
                # Create BINARY recording folder
                binary_temporal_folder = os.path.join(tempFolderPath, 'lfp_' + eg)

                si_recording_lfp1k.save_to_folder(folder=binary_temporal_folder, overwrite=True, **lfp_job_kwargs)

                del si_recording_raw, si_recording_lfp, si_recording_lfp1k, 

                si_recording_lfp1k = load_extractor(binary_temporal_folder)

                if verbose:
                    print('reading lfp from binary successful¡......')
                
                ##############################################################################################################
                # Update recording properties to match NWB electrode table
                si_recording_lfp1k = udpdate_si_recording_with_nwb_electrodes_table(si_recording_lfp1k, nwbFile_path=nwbPrepro_path, electrodeGroupName=eg[4::], verbose=verbose)

                ##############################################################################################################
                # It might always be a single segment, but in this way we follow NEUROCONV pipeline
                number_of_segments = si_recording_lfp1k.get_num_segments()

                for segment_index in range(number_of_segments):

                    if verbose:
                        print('adding lfp to NWB (neuroconv AbstractChunkIterator)...... segement: {}/{}'.format(segment_index+1, number_of_segments))

                    ##############################################################################################################
                    # Open-read-write PREPRO-NWB
                    nwbPrepro_io = NWBHDF5IO(nwbPrepro_path, mode="r+")
                    nwbPrepro = nwbPrepro_io.read()

                    ##############################################################################################################
                    # USE NEUROCONV
                    neuroconv_add_electrical_series_to_nwbfile(
                        recording = si_recording_lfp1k,
                        nwbfile = nwbPrepro,
                        metadata = metadata,
                        segment_index = segment_index,
                        starting_time = None,
                        write_as = 'lfp',
                        es_key = es_name,
                        write_scaled = False,
                        iterator_type = 'v2',
                        always_write_timestamps = False
                        )
                    
                    ##############################################################################################################
                    # Write-close PREPRO-NWB
                    nwbPrepro_io.write(nwbPrepro)
                    nwbPrepro_io.close()

                    del nwbPrepro_io, nwbPrepro

                ##############################################################################################################
                # Remove LFP DATA
                if verbose:
                    print('removing {}'.format(binary_temporal_folder))

                # Close all recording segments
                for segment_lfp in si_recording_lfp1k._recording_segments:
                    segment_lfp.file.close()
                del si_recording_lfp1k

                shutil.rmtree(binary_temporal_folder, ignore_errors=True)

##############################################################################################################
# Resample ripple - RAW EYE data using SpikeInterface preprocessing:
# data is transformed into a SI recording object using NumpyRecording
# resampled data is saved into a temporal HDF5 file
##############################################################################################################
def nwb_resample_nsEyeData(nwbRaw_behavior_module, nwbPrepro_behavior_module, resample_params, tempFolderPath, use_spikeInterface=False, verbose=False):
    
    ##############################################################################################################################
    # It will use behaviour Modules as inputs. 
    # behaviour Module is a multicontainer with Data interfaces that can be accessed as dictionary
    eyeResampleData_h5paths = []
    eyeResampleData_h5objs = []
    
    ########################################################
    # EYE traking DATA : spatial_series
    if 'EyeTracking' in nwbRaw_behavior_module.data_interfaces.keys():

        eyeTracking_raw = nwbRaw_behavior_module.get_container('EyeTracking')

        createPrepro_EyeTracking = False
        if 'EyeTracking' not in nwbPrepro_behavior_module.data_interfaces.keys():
            createPrepro_EyeTracking = True
            prepro_EyeTacking_series_names = []
        else:
            prepro_EyeTacking_series_names = list(nwbPrepro_behavior_module['EyeTracking'].spatial_series.keys())

        # Eye Tracking is a container of spatial_series (eye position x, y, z)
        # Get those related to Ripple:
        raw_eyeRipple_spatial_series_list = [val for key, val in eyeTracking_raw.spatial_series.items() if 'eyeRipple' in key]

        for raw_eyeRipple_spatial_series in raw_eyeRipple_spatial_series_list:
            # TODO: check if RAW module is already at the desired sampling_frequency, then only copy the container
            # TODO: check if PREPRO module is already at the desired sampling_frequency, then don't do anything

            # Check if this spatial series already exists in the PREPRO module.
            if raw_eyeRipple_spatial_series.name in prepro_EyeTacking_series_names:
                if verbose:
                    print('{} spatial series already exist in the PREPRO nwbFile. It will be removed and replace it'.format(raw_eyeRipple_spatial_series.name))
                nwbPrepro_behavior_module['EyeTracking'].spatial_series.pop(raw_eyeRipple_spatial_series.name)

            if use_spikeInterface:
                if verbose:
                    print('\n\nloading {} spatial_series (shape = [{}]) into memory to create spikeinterface Recording object\nLazy resampling....'.format(
                        raw_eyeRipple_spatial_series.name, raw_eyeRipple_spatial_series.data.shape
                    ))

                if raw_eyeRipple_spatial_series.data.ndim==1:
                    eye_recording_raw = NumpyRecording(traces_list=numpy.array([raw_eyeRipple_spatial_series.data[:]]).T, sampling_frequency=raw_eyeRipple_spatial_series.rate)
                elif raw_eyeRipple_spatial_series.data.ndim>1:
                    if verbose:
                        print('\nWARNING¡¡¡¡ \n{} has more than one channel (nChans = {})\nIt can create memory issues'.format(
                            raw_eyeRipple_spatial_series.name, raw_eyeRipple_spatial_series.data.shape[1])
                            )
                    eye_recording_raw = NumpyRecording(traces_list=numpy.array(raw_eyeRipple_spatial_series.data[:]), sampling_frequency=raw_eyeRipple_spatial_series.rate)
                
                eye_recording_resampled = resample(eye_recording_raw, **resample_params)

                eyeResample_temp = temp_hdf5_from_numpy_with_DataChunkIterator(
                    array = (eye_recording_resampled.get_traces(return_scaled=False).squeeze() * raw_eyeRipple_spatial_series.conversion) + raw_eyeRipple_spatial_series.offset, 
                    tempFolderPath = tempFolderPath, 
                    tempName = raw_eyeRipple_spatial_series.name+'_resample',
                    verbose=verbose)
                
                del eye_recording_raw, eye_recording_resampled

            else:
                if verbose:
                    print('\n\nloading {} spatial timeseries (shape = [{}]) into memory to resample data ....'.format(
                        raw_eyeRipple_spatial_series.name, raw_eyeRipple_spatial_series.data.shape
                    ))

                eyeResample_temp = temp_resample_hdf5_dataset(
                    hdf5dataset = raw_eyeRipple_spatial_series.data,
                    dataset_rate = raw_eyeRipple_spatial_series.rate, 
                    resample_rate = resample_params['resample_rate'], 
                    tempFolderPath = tempFolderPath, 
                    tempName = raw_eyeRipple_spatial_series.name + '_resample', 
                    margin_ms = resample_params['margin_ms'], 
                    add_reflect_padding = True, 
                    add_zeros = False,
                    conversion =  raw_eyeRipple_spatial_series.conversion,
                    offset = raw_eyeRipple_spatial_series.offset,
                    verbose=verbose)
            
            eyeResample_h5 = h5py.File(name=eyeResample_temp, mode="r")
        
            eyeTrackingNS_resampled = SpatialSeries(
                name= raw_eyeRipple_spatial_series.name,
                description = raw_eyeRipple_spatial_series.description + '(resampled at : {}Hz; using SpikeInterface)'.format(resample_params['resample_rate']),
                comments = raw_eyeRipple_spatial_series.comments + '(resampled at : {}Hz; using SpikeInterface)'.format(resample_params['resample_rate']),
                data = H5DataIO(eyeResample_h5["dataSet"]),
                reference_frame = raw_eyeRipple_spatial_series.reference_frame,
                starting_time = raw_eyeRipple_spatial_series.starting_time,
                rate = float(resample_params['resample_rate']),
                unit = raw_eyeRipple_spatial_series.unit,
                conversion = 1.0,
                offset = 0.0
                )
            
            if createPrepro_EyeTracking:
                if verbose:
                    print('EyeTracking container was NOT found in the PREPRO nwbFile, it will be created...')
                nwbPrepro_behavior_module.add(EyeTracking(spatial_series=eyeTrackingNS_resampled))
                createPrepro_EyeTracking = False
            else:
                nwbPrepro_behavior_module['EyeTracking'].add_spatial_series(spatial_series=eyeTrackingNS_resampled)
            
            eyeResampleData_h5paths.append(eyeResample_temp)
            eyeResampleData_h5objs.append(eyeResample_h5)
    else:
        print('\nNOTHING TO PREPROCESS¡¡\nEyeTracking container was NOT found in the RAW nwbFile.\nAvailable containers in RAW-"behaviour": {}\n'.format(nwbRaw_behavior_module.data_interfaces.keys()))
    

    ########################################################
    # PUPIL tracking DATA: time_series
    if 'PupilTracking' in nwbRaw_behavior_module.data_interfaces.keys():

        pupilTracking_raw = nwbRaw_behavior_module.get_container('PupilTracking')

        createPrepro_PupilTracking = False
        if 'PupilTracking' not in nwbPrepro_behavior_module.data_interfaces.keys():
            createPrepro_PupilTracking = True
            prepro_PupilTacking_series_names = []
        else:
            prepro_PupilTacking_series_names = list(nwbPrepro_behavior_module['PupilTracking'].time_series.keys())

        # Eye Tracking is a container of time_series (pupil diameter)
        # Get those related to Ripple:
        raw_pupilRipple_time_series_list = [val for key, val in pupilTracking_raw.time_series.items() if 'pupilRipple' in key]

        for raw_pupilRipple_time_series in raw_pupilRipple_time_series_list:
             # TODO: check if RAW module is already at the desired sampling_frequency, then only copy the container
             # TODO: check if PREPRO module is already at the desired sampling_frequency, then don't do anything

            # Check if this time series already exists in the PREPRO module.
            if raw_pupilRipple_time_series.name in prepro_PupilTacking_series_names:
                if verbose:
                    print('{} time series already exist in the PREPRO nwbFile. It will be removed and replace it'.format(raw_pupilRipple_time_series.name))
                nwbPrepro_behavior_module['PupilTracking'].time_series.pop(raw_pupilRipple_time_series.name)

            if use_spikeInterface:
                if verbose:
                    print('\n\nloading {} timeseries (shape = [{}]) into memory to create spikeinterface Recording object\nLazy resampling....'.format(
                        raw_pupilRipple_time_series.name, raw_pupilRipple_time_series.data.shape
                    ))

                if raw_pupilRipple_time_series.data.ndim==1:
                    pupil_recording_raw = NumpyRecording(traces_list=numpy.array([raw_pupilRipple_time_series.data[:]]).T, sampling_frequency=raw_pupilRipple_time_series.rate)
                elif raw_pupilRipple_time_series.data.ndim>1:
                    if verbose:
                        print('\nWARNING¡¡¡¡ \n{} has more than one channel (nChans = {})\nIt can create memory issues'.format(
                            raw_pupilRipple_time_series.name, raw_pupilRipple_time_series.data.shape[1])
                            )
                    pupil_recording_raw = NumpyRecording(traces_list=numpy.array(raw_pupilRipple_time_series.data[:]), sampling_frequency=raw_pupilRipple_time_series.rate)
                
                pupil_recording_resampled = resample(pupil_recording_raw, **resample_params)

                pupilResample_temp = temp_hdf5_from_numpy_with_DataChunkIterator(
                    array = (pupil_recording_resampled.get_traces(return_scaled=False).squeeze() * raw_pupilRipple_time_series.conversion) + raw_pupilRipple_time_series.offset, 
                    tempFolderPath = tempFolderPath, 
                    tempName = raw_pupilRipple_time_series.name + '_resample',
                    verbose=verbose)
                
                del pupil_recording_raw, pupil_recording_resampled

            else:
                if verbose:
                    print('\n\nloading {} timeseries (shape = [{}]) into memory to resample data ....'.format(
                        raw_pupilRipple_time_series.name, raw_pupilRipple_time_series.data.shape
                    ))

                pupilResample_temp = temp_resample_hdf5_dataset(
                    hdf5dataset = raw_pupilRipple_time_series.data,
                    dataset_rate = raw_pupilRipple_time_series.rate, 
                    resample_rate = resample_params['resample_rate'], 
                    tempFolderPath = tempFolderPath, 
                    tempName = raw_pupilRipple_time_series.name + '_resample', 
                    margin_ms = resample_params['margin_ms'], 
                    add_reflect_padding = True, 
                    add_zeros = False,
                    conversion =  raw_pupilRipple_time_series.conversion,
                    offset = raw_pupilRipple_time_series.offset,
                    verbose=verbose)
            
            pupilResample_h5 = h5py.File(name=pupilResample_temp, mode="r")
        
            pupilTrackingNS_resampled = TimeSeries(
                name= raw_pupilRipple_time_series.name,
                description = raw_pupilRipple_time_series.description + '(resampled at : {}Hz; using SpikeInterface)'.format(resample_params['resample_rate']),
                comments = raw_pupilRipple_time_series.comments + '(resampled at : {}Hz; using SpikeInterface)'.format(resample_params['resample_rate']),
                data = H5DataIO(pupilResample_h5["dataSet"]),
                starting_time = raw_pupilRipple_time_series.starting_time,
                rate = float(resample_params['resample_rate']),
                unit = raw_pupilRipple_time_series.unit,
                conversion = 1.0,
                offset = 0.0
                )
            
            if createPrepro_PupilTracking:
                if verbose:
                    print('PupilTracking container was NOT found in the PREPRO nwbFile, it will be created...')
                nwbPrepro_behavior_module.add(PupilTracking(time_series=pupilTrackingNS_resampled))
                createPrepro_PupilTracking = False
            else:
                nwbPrepro_behavior_module['PupilTracking'].add_timeseries(time_series=pupilTrackingNS_resampled)
            
            eyeResampleData_h5paths.append(pupilResample_temp)
            eyeResampleData_h5objs.append(pupilResample_h5)
    else:
        print('\nNOTHING TO PREPROCESS¡¡\nPupilTracking container was NOT found in the RAW nwbFile.\nAvailable containers in RAW-"behaviour": {}\n'.format(nwbRaw_behavior_module.data_interfaces.keys()))

    
    return eyeResampleData_h5objs, eyeResampleData_h5paths

def nwb_resample_nsAnalog_stimuli(nwbRaw, nwbPrepro, stimulusName, resample_params, tempFolderPath, use_spikeInterface=False, verbose=False):

    ##############################################################################################################################
    # stimulus Module is a multicontainer with Data interfaces that can be accessed as dictionary
    # Up to 2024 these are the Stim Options: ['fixON', 'visualON', 'rewardON', 'leftAccelerometer', 'leftCommand', 'rightAccelerometer', 'rightCommand', 'thermistors']

    if stimulusName in nwbRaw.stimulus:

        if type(nwbRaw.stimulus[stimulusName])==TimeSeries:

            rawRipple_time_series = nwbRaw.stimulus[stimulusName]
            # TODO: check if RAW module is already at the desired sampling_frequency, then only copy the container
            # TODO: check if PREPRO module is already at the desired sampling_frequency, then don't do anything

            # Check if this time series already exists in the PREPRO module.
            if rawRipple_time_series.name in nwbPrepro.stimulus.keys():
                if verbose:
                    print('\n{} time series already exist in the PREPRO nwbFile. It will be removed and replace it'.format(rawRipple_time_series.name))
                nwbPrepro.stimulus.pop(rawRipple_time_series.name)
            
            if use_spikeInterface:

                if verbose:
                    print('\n\nloading {} timeseries (shape = [{}]) into memory to create spikeinterface Recording object\nLazy resampling....'.format(
                        rawRipple_time_series.name, rawRipple_time_series.data.shape
                    ))

                if rawRipple_time_series.data.ndim==1:
                    si_recording_raw = NumpyRecording(traces_list=numpy.array([rawRipple_time_series.data[:]]).T, sampling_frequency=rawRipple_time_series.rate)
                elif rawRipple_time_series.data.ndim>1:
                    if verbose:
                        print('\nWARNING¡¡¡¡ \n{} has more than one channel (nChans = {})\nIt can create memory issues'.format(
                            rawRipple_time_series.name, rawRipple_time_series.data.shape[1])
                            )
                    si_recording_raw = NumpyRecording(traces_list=numpy.array(rawRipple_time_series.data[:]), sampling_frequency=rawRipple_time_series.rate)
                
                si_recording_resampled = resample(si_recording_raw, **resample_params)

                resample_temp_path = temp_hdf5_from_numpy_with_DataChunkIterator(
                    array = (si_recording_resampled.get_traces(return_scaled=False).squeeze() * rawRipple_time_series.conversion) + rawRipple_time_series.offset, 
                    tempFolderPath = tempFolderPath, 
                    tempName = rawRipple_time_series.name + '_resample',
                    verbose=verbose)
                
                del si_recording_raw, si_recording_resampled

            else:
                if verbose:
                    print('\n\nloading {} timeseries (shape = [{}]) into memory to resample data ....'.format(
                        rawRipple_time_series.name, rawRipple_time_series.data.shape
                    ))

                resample_temp_path = temp_resample_hdf5_dataset(
                    hdf5dataset = rawRipple_time_series.data,
                    dataset_rate = rawRipple_time_series.rate, 
                    resample_rate = resample_params['resample_rate'], 
                    tempFolderPath = tempFolderPath, 
                    tempName = rawRipple_time_series.name + '_resample', 
                    margin_ms = resample_params['margin_ms'], 
                    add_reflect_padding = True, 
                    add_zeros = False,
                    conversion =  rawRipple_time_series.conversion,
                    offset = rawRipple_time_series.offset,
                    verbose=verbose)
            
            resample_h5 = h5py.File(name=resample_temp_path, mode="r")
        
            nwbPrepro.add_stimulus(
                TimeSeries(
                    name= rawRipple_time_series.name,
                    description = rawRipple_time_series.description + '(resampled at : {}Hz; using SpikeInterface)'.format(resample_params['resample_rate']),
                    comments = rawRipple_time_series.comments + '(resampled at : {}Hz; using SpikeInterface)'.format(resample_params['resample_rate']),
                    data = H5DataIO(resample_h5["dataSet"]),
                    starting_time = rawRipple_time_series.starting_time,
                    rate = float(resample_params['resample_rate']),
                    unit = rawRipple_time_series.unit,
                    conversion = 1.0,
                    offset = 0.0)
                )
            
            if verbose:
                print('Stimulus: {} has been added to PREPRO nwbFile...'.format(stimulusName))

            return resample_h5, resample_temp_path

        else:
            raise Exception('\nStimulus: "{}" is type={}. Resampling function works with pynwb "TimeSeries" object only'.format(stimulusName, type(nwbRaw.stimulus[stimulusName])))
    else:
        print('\nStimulus: "{}" do not exists in nwbRAW. Valid Stimulus names: {}'.format(stimulusName, list(nwbPrepro.stimulus.keys())))
        return None, None


###############################################################################################################################################################
# Add UNITS from a single-session "sorting_analyzer" into its corresponding "-prepro.nwb"
# It will read & write & close the NWB file 
# It will run nwb-inspector, print message according to "above_nwbinspectorSeverity" (HIGH = 2, LOW = 1). Set to "0" it will print all
###############################################################################################################################################################
def nwb_add_units_from_sorting_analyzer_session(sorting_analyzer_folder, parentFolder_preproNWB, curated_params, sorting_analyzer_params, add_waveforms=True, rewrite=False, verbose=True, above_nwbinspectorSeverity=1):

    units_module_description = 'Sorting results from extracellular electrophysiology recordings. One "units" table per electrode group'

    sorting_analyzer = load_sorting_analyzer(folder=sorting_analyzer_folder, load_extensions=True)

    ######################################################################################################################
    # Search for all prepro sessions that were sorted together:
    ######################################################################################################################
    nwbPrepro_path, _, shared_prepro_fileNames = find_peproNWBpath_from_si_recording_session(si_recording = sorting_analyzer.recording, parentFolder_preproNWB=parentFolder_preproNWB, verbose=verbose)

    ######################################################################################################################
    # Get electrode_indices from the corresponding electrodeGroup
    ######################################################################################################################
    nwbPrepro_io = NWBHDF5IO(nwbPrepro_path, mode="r")
    nwbPrepro = nwbPrepro_io.read()

    elecGroupSessName = sorting_analyzer.recording.get_annotation('elecGroupSessName')
    electrodeGroup_Name = elecGroupSessName[elecGroupSessName.index('-raw-')+1:]

    electrodeGroup_Name_in_NWB = [name for name in nwbPrepro.electrode_groups.keys() if name in electrodeGroup_Name]
    if len(electrodeGroup_Name_in_NWB) != 1:
        raise Exception('Electrode Group {} was detected {} times. It must match with a unique group name from the available electrode_groups in the NWB file : \n{}'.format(
            electrodeGroup_Name, len(electrodeGroup_Name_in_NWB), nwbPrepro.electrode_groups.keys()
        ))
    electrodeGroup_Name_in_NWB = electrodeGroup_Name_in_NWB[0]

    # Adapted from "neuroconv"
    nwb_electrode_group_indices = nwbPrepro.electrodes.to_dataframe().query(f"group_name in '{electrodeGroup_Name_in_NWB}'").index.values
    # Electrode Indices must match (recording was extracted from an NWB)
    if set(sorting_analyzer.channel_ids) - set(nwb_electrode_group_indices):
        raise Exception('There are sorting analyzer electrode IDs ({}) that do NOT match with the IDs from the NWB eletrodeGroup {}.\nelectrodeIDs from sorting_analyzer:{}\nelectrodeIDs from NWB:{}'.format(
            set(sorting_analyzer.channel_ids) - set(nwb_electrode_group_indices), electrodeGroup_Name_in_NWB, sorting_analyzer.channel_ids, nwb_electrode_group_indices
        ))

    nwb_electrodes_ids = list(nwbPrepro.electrodes.to_dataframe().index.values)
    channel_indexes_in_NWB = [nwb_electrodes_ids.index(e) for e in sorting_analyzer.channel_ids]

    nwbPrepro_io.close()
    del nwbPrepro_io, nwbPrepro

    es_name = 'ephys-{}'.format(electrodeGroup_Name_in_NWB)
    waveforms_exists, waveforms_message_exists = _exists_electrical_series_in_nwb_ecephys_module(nwb_path=nwbPrepro_path, processing_module="Processed", electrical_series_name=es_name)
    if add_waveforms and not waveforms_exists:
        print('"add_waveforms" was set to "True", but {}\nBe sure to export "preocessed_si_recording" first, otherwise set "add_waveforms" = False'.format(waveforms_message_exists))
        raise Exception('ADD waveforms error¡')

    ######################################################################################################################
    # Get electrode_indices from the corresponding electrodeGroup
    ######################################################################################################################
    units_table_name ='units-{}'.format(electrodeGroup_Name_in_NWB)
    units_exists, message_exists = _exists_units_table_in_nwb_ecephys_module(nwb_path=nwbPrepro_path, units_table_name=units_table_name)

    ##############################################################################################################
    # If the UNITS table already exists and rewrite=True, remove the table 
    valid_writing = True
    if units_exists and not rewrite:
        valid_writing = False
        print('WARNING: {}\nskipping {} ....¡¡'.format(message_exists, units_table_name))

    ##############################################################################################################
    # If the UNITS table already exists and rewrite=True, remove the table 
    elif units_exists and rewrite:
        _remove_units_table_from_nwb_ecephys_module(nwb_path=nwbPrepro_path, units_table_name=units_table_name, verbose=verbose)
    

    if valid_writing:

        ######################################################################################################################
        # PREPARE DICTIONARY WITH SORTING PROPERTIES TO EXPORT
        ######################################################################################################################
        if sorting_analyzer.sorting.get_num_segments()>1:
            raise Exception('Sorting object has more than one segments (n={})\nExporting multisegment sorting is not supported yet¡¡'.format(sorting_analyzer.sorting.get_num_segments()))
        else:
            segment_index = 0

        unit_ids = sorting_analyzer.unit_ids
        channel_ids = sorting_analyzer.channel_ids
        num_units = len(unit_ids)
        num_channels = len(channel_ids)

        if sorting_analyzer.is_sparse():
            # Already a dictionary {unitID:value}
            units_sparse_channel_ids = sorting_analyzer.sparsity.unit_id_to_channel_ids
            units_sparse_channel_indices = sorting_analyzer.sparsity.unit_id_to_channel_indices
        else:
            # Create a dictionary {unitID:value}
            units_sparse_channel_ids = {}
            units_sparse_channel_indices = {}
            for u in unit_ids:
                units_sparse_channel_ids.update({u:sorting_analyzer.channel_ids})
                units_sparse_channel_indices.update({u:numpy.arange(num_channels, dtype='int64')})

        ######################################################################################################
        # retrieve templates means and stds
        template_extension = sorting_analyzer.get_extension("templates")
        if template_extension is None:
            raise ValueError("No templates found in the sorting analyzer.")
        template_means = template_extension.get_templates()
        template_stds = template_extension.get_templates(operator="std")
        quality = sorting_analyzer.sorting.get_property('quality')
        if quality is None:
            raise Exception('Sorting must contain "quality" property (indicates it was not curated¡¡)')
        original_cluster_id = sorting_analyzer.sorting.get_property('original_cluster_id')
        if original_cluster_id is None:
            original_cluster_id = unit_ids

        # Create/Update a dictionary {unitID:value}
        units_template_means = {}
        units_template_stds = {}
        units_quality = {}
        units_original_cluster_id = {}

        for i in range(num_units):
            units_template_means.update({unit_ids[i]: template_means[i, :, units_sparse_channel_indices[unit_ids[i]]]})
            units_template_stds.update({unit_ids[i]: template_stds[i, :, units_sparse_channel_indices[unit_ids[i]]]})
            units_quality.update({unit_ids[i]: quality[i]})
            units_original_cluster_id.update({unit_ids[i]: original_cluster_id[i]})
        
        ######################################################################################################
        # Retrive the channel with the max amplitude per unit the max amplitude
        # Neuroconv naming of extremnum_channel os: "max_channel" or "max_electrode"
        # Already a dictionary {unitID:value}
        units_extrem_channel = get_template_extremum_channel(sorting_analyzer, peak_sign=sorting_analyzer_params['template_metrics']['peak_sign'], mode="extremum", outputs="index")
        units_extrem_amplitude = get_template_extremum_amplitude(sorting_analyzer, peak_sign=sorting_analyzer_params['template_metrics']['peak_sign'], mode="extremum", abs_value=False)

        ######################################################################################################
        # retrieve spike related properties
        spike_unit_index = sorting_analyzer.sorting.to_spike_vector()['unit_index']
        spike_unit_ids = numpy.array([unit_ids[i] for i in spike_unit_index])
        spike_amplitudes = sorting_analyzer.get_extension('spike_amplitudes').get_data(outputs = "by_unit")[segment_index]
        amplitude_scalings_extension = sorting_analyzer.get_extension('amplitude_scalings').data['amplitude_scalings']
        spike_locations = sorting_analyzer.get_extension('spike_locations').get_data(outputs = "by_unit")[segment_index]

        unit_locations = sorting_analyzer.get_extension('unit_locations').get_data(outputs = "by_unit")

        template_metrics = sorting_analyzer.get_extension('template_metrics').get_data()
        quality_metrics = sorting_analyzer.get_extension('quality_metrics').get_data()
        
        ######################################################################################################
        # Create a dictionary {unitID:value}
        units_electrodes = {}
        units_extrem_electrode = {}

        units_spike_times = {}
        units_amplitude_scalings = {}
        units_spike_rel_x = {}
        units_spike_rel_y = {}
        units_spike_rel_z = {}
        units_spike_rel_alpha = {}

        units_unit_rel_x = {}
        units_unit_rel_y = {}
        units_unit_rel_z = {}
        units_unit_rel_alpha = {}

        units_template_metrics = {}
        units_quality_metrics = {}

        for u in unit_ids:

            units_electrodes.update({u: numpy.array([channel_indexes_in_NWB[e] for e in units_sparse_channel_indices[u]])})
            units_extrem_electrode.update({u: channel_indexes_in_NWB[units_extrem_channel[u]]})

            units_spike_times.update(
                {u: sorting_analyzer.sorting.get_unit_spike_train(unit_id=u, segment_index=segment_index, return_times=True)}
                )
            
            units_amplitude_scalings.update({u: amplitude_scalings_extension[spike_unit_ids==u]})

            units_spike_rel_x.update({u: spike_locations[u]['x']})
            units_spike_rel_y.update({u: spike_locations[u]['y']})
            if 'z' in spike_locations[u].dtype.fields:
                units_spike_rel_z.update({u: spike_locations[u]['z']})
            else:
                units_spike_rel_z.update({u: numpy.full_like(spike_locations[u]['x'], numpy.nan, dtype=numpy.double)})
            if 'alpha' in spike_locations[u].dtype.fields:
                units_spike_rel_alpha.update({u: spike_locations[u]['alpha']})
            else:                 
                units_spike_rel_alpha.update({u: numpy.full_like(spike_locations[u]['x'], numpy.nan, dtype=numpy.double)})

            units_unit_rel_x.update({u: unit_locations[u][0]})
            units_unit_rel_y.update({u: unit_locations[u][1]})
            if len(unit_locations[u])>2:
                units_unit_rel_z.update({u: unit_locations[u][2]})
            else:
                units_unit_rel_z.update({u: numpy.nan})
            if len(unit_locations[u])>3:
                units_unit_rel_alpha.update({u: unit_locations[u][3]})
            else:
                units_unit_rel_alpha.update({u: numpy.nan})

            units_template_metrics.update({u:{}})
            for prop in template_metrics.columns:
                units_template_metrics[u].update({prop:template_metrics.loc[u, prop]})

            units_quality_metrics.update({u:{}})
            for prop in quality_metrics.columns:
                units_quality_metrics[u].update({prop:quality_metrics.loc[u, prop]})

        del spike_unit_index, spike_unit_ids, amplitude_scalings_extension, spike_locations, unit_locations, template_metrics, quality_metrics
        
        ######################################################################################################################
        # Create dictionary per unit with all properties (aka columns) 
        nwbPrepro_io = NWBHDF5IO(nwbPrepro_path, mode="r+")
        nwbPrepro = nwbPrepro_io.read()

        units_dict = {}
        for u in unit_ids:
            unit_row = dict()
            unit_row.update({
                'unit_name': str(u),
                'quality': units_quality[u],
                'original_cluster_id': units_original_cluster_id[u],
                'spike_times': units_spike_times[u],
                'electrodes':  units_electrodes[u], # units_electrodes[u], units_sparse_channel_ids[u],
                'electrode_group': nwbPrepro.get_electrode_group(electrodeGroup_Name_in_NWB),
                'spike_amplitudes': spike_amplitudes[u],
                'amplitude_scalings': units_amplitude_scalings[u],
                'spike_rel_x': units_spike_rel_x[u],
                'spike_rel_y': units_spike_rel_y[u],
                'spike_rel_z': units_spike_rel_z[u],
                'spike_rel_alpha': units_spike_rel_alpha[u],
                'waveform_mean': units_template_means[u],
                'waveform_sd': units_template_stds[u],
                'unit_amplitude': units_extrem_amplitude[u],
                'unit_rel_x': units_unit_rel_x[u],
                'unit_rel_y': units_unit_rel_y[u],
                'unit_rel_z': units_unit_rel_z[u],
                'unit_rel_alpha': units_unit_rel_alpha[u],
                'max_electrode': units_extrem_electrode[u]
            })

            unit_row.update(units_template_metrics[u])
            unit_row.update(units_quality_metrics[u])

            unit_row.update({
                'sessions_shared': shared_prepro_fileNames,
                'motion_corrected': curated_params['motion_corrected'],
                'recording_sufix': curated_params['recording_sufix'],
                'curated_with_phy': curated_params['curated_with_phy'],
                'sorterName': curated_params['sorterName'],
                'sorter_detect_sign': curated_params['sorter_detect_sign'],
                'sorter_detect_threshold': curated_params['sorter_detect_threshold'],
                'sorter_nearest_chans': curated_params['sorter_nearest_chans']
            })
            
            units_dict.update({u:unit_row})

        ######################################################################################################################
        # MATCH PROPERTIES WITH DEFAULT UNITS-COLUMNS
        ######################################################################################################################
        waveform_samples = curated_params['n_before']+curated_params['n_after']

        unit_table_description_default, unit_columns_information = get_default_unitsCols(waveform_samples = waveform_samples)

        # Validate data type and add default values for non-existing properties
        required_properties = ['spike_times', 'electrodes', 'electrode_group'] # , 'waveform_mean', 'waveform_sd'

        for u in unit_ids:
            if verbose:
                print('Checking columns from UNIT={}'.format(u))
            
            # Check all default columns
            for colName, colVal in unit_columns_information.items():
                if colName not in required_properties:
                    # Check if that parameter exists:
                    if colName in units_dict[u].keys():
                        # Check type and Try to change it 
                        origType = type(units_dict[u][colName])
                        if colVal['default_type'] != numpy.ndarray and origType != colVal['default_type']:
                            newType = colVal['default_type'](units_dict[u][colName])
                            units_dict[u][colName] = newType
                        elif colVal['default_type'] == numpy.ndarray:
                            # Update if it is a list
                            if origType != colVal['default_type']:
                                if origType ==list:
                                    newType = numpy.array(units_dict[u][colName])
                                    units_dict[u][colName] = newType
                                else:
                                    raise Exception('property {} from Unit={} was expected to be numpy.ndarray or list but it was {}\nvalues:{}'.format(colName, u, origType, units_dict[u][colName]))
                            # Check number of dimensions match the expected value
                            if colVal['index']!=units_dict[u][colName].ndim:
                                if 'waveform_'not in colName:
                                    raise Exception('property {} from Unit={} was expected to be numpy.ndarray with ndim={} but it has ndim={}\noriginal array:{}'.format(
                                        colName, u, colVal['index'], units_dict[u][colName].ndim, units_dict[u][colName]))
                    
                    # Otherwise add default values:
                    else:
                        print('\tWarning: NOT found property = "{}"\tIt will have default value = {}'.format(colName, colVal['default_value']))
                        units_dict[u].update({colName: colVal['default_value']})
            
        # Check if there is unit-keys NOT present in the the unit_column_information:
        invalid_keys = []
        invalid_keys = [unit_key for unit_key in units_dict[u].keys() if unit_key not in unit_columns_information.keys() and unit_key not in required_properties]
        if len(invalid_keys)> 0:
            raise Exception('key(s)={} from Unit={} do NOT exists in the unit_columns_information (check & update "yaulab_processing.yaulab_extras.get_default_unitsCols()")'.format(invalid_keys, u))
        

        ######################################################################################################################
        # Write UNITS table in the "units" processing module 
        ######################################################################################################################
        
        ######################################################################################################################
        # Check if "units" processing module exists. If not, create it. Then return module.
        # Method copied from NEUROCONV (nwb_helpers.get_module)
        if "units" in nwbPrepro.processing:
            units_module = nwbPrepro.processing["units"]
        else:
            units_module = nwbPrepro.create_processing_module(name="units", description=units_module_description)

        ######################################################################################################################
        # Create UNITS table.
        ######################################################################################################################
        # columns_to_remove is mainly use for testing variables that might create an error
        columns_to_remove = [] # ['waveform_mean', 'waveform_sd'] # ['spike_amplitudes', 'amplitude_scalings', 'spike_rel_x', 'spike_rel_y', 'spike_rel_z', 'spike_rel_alpha']

        unit_table_description = 'UNITS table from electrode group: {}. {}'.format(electrodeGroup_Name_in_NWB, unit_table_description_default)

        units_table = Units(name=units_table_name, description=unit_table_description, electrode_table=nwbPrepro.electrodes, 
                            waveform_rate=sorting_analyzer.sampling_frequency, 
                            waveform_unit='uV')
        
        units_module.add(units_table)

        # Update Electrode Table in max_electrode column
        unit_columns_information['max_electrode'].update({'table':nwbPrepro.electrodes})

        # Create UNITS columns
        for colName, colVal in unit_columns_information.items():
            if colName not in columns_to_remove and colName not in required_properties:
                
                if 'table' in colVal.keys():
                    table = colVal['table']
                else:
                    table = False
                
                if 'index' in colVal.keys():
                    index = colVal['index']
                else:
                    index = False
                # if verbose:
                #    print('adding column {} (table={}, index={})'.format(colName, table, index))

                units_table.add_column(name=colName, description=colVal['description'], table=table, index=index)
        
        ######################################################################################################################
        # Add UNITS 
        # It will extract Waveforms per units (it is required to add si_recording first)
        ######################################################################################################################
        if add_waveforms:

            # Create a temporal folder to save hdf5 files with data from units
            # It will use the parent folder of the sorting_analyzer folder 
            parentFolder, _ = os.path.split(sorting_analyzer_folder)
            _, nwbFileName = os.path.split(nwbPrepro_path)
            nwbFileNameSplit = os.path.splitext(nwbFileName)
            unitsWF_folder = os.path.join(parentFolder, '{}-units-{}'.format(nwbFileNameSplit[0], es_name))
            del parentFolder, nwbFileName, nwbFileNameSplit

            if os.path.isdir(unitsWF_folder):  
                if verbose:
                    print('\nTemporal folder for waveforms already exists. It will be deleted.. \n{}.............\n'.format(unitsWF_folder))
                shutil.rmtree(unitsWF_folder, ignore_errors=False)

            # Create a temporal folder to save hdf5 files with data from units    
            os.makedirs(unitsWF_folder, exist_ok=False)
            if verbose:
                print('\nTemporal folder to store waveforms was successfully created¡ .............\n')

            unitsWF_h5paths = []
            unitsWF_h5Objs = []
        
            
        """
        if add_waveforms:

            ephys_eg = nwbPrepro.processing['ecephys']['Processed'].electrical_series[es_name]
            eg_electrodes_index = ephys_eg.electrodes.to_dataframe().index.to_numpy()
            duration_samples = ephys_eg.data.shape[0]

            # Create a temporal folder to save hdf5 files with data from units
            # It will use the parent folder of the sorting_analyzer folder 
            parentFolder, _ = os.path.split(sorting_analyzer_folder)
            _, nwbFileName = os.path.split(nwbPrepro_path)
            nwbFileNameSplit = os.path.splitext(nwbFileName)
            unitsWF_folder = os.path.join(parentFolder, '{}-units-{}'.format(nwbFileNameSplit[0], es_name))
            del parentFolder, nwbFileName, nwbFileNameSplit

            if os.path.isdir(unitsWF_folder):  
                if verbose:
                    print('\nTemporal folder for waveforms already exists. It will be deleted.. \n{}.............\n'.format(unitsWF_folder))
                shutil.rmtree(unitsWF_folder, ignore_errors=False)
            
            os.makedirs(unitsWF_folder)
            if verbose:
                print('\nTemporal folder to store waveforms was successfully created¡ .............\n')
                
        unitsWF_h5paths = []
        unitsWF_h5Objs = []
        """

        for u in unit_ids:
            if verbose:
                print('adding UNIT={} to the UNITS table...........'.format(u))

            unit_copy = units_dict[u].copy()

            # Remove columns that are required as individual inputs
            for key in required_properties:
                if key in unit_copy:
                    unit_copy.pop(key)
                    
            # Remove unwanted columns
            for key in columns_to_remove:
                if key in unit_copy:
                    unit_copy.pop(key)

            # Double check waveform_mean & waveform_sd is removed from the copy
            if 'waveform_mean' in unit_copy:
                unit_copy.pop('waveform_mean')
            if 'waveform_sd' in unit_copy:
                unit_copy.pop('waveform_sd')
            
            ######################################################################################################################
            # EXTRACT WAVEFORMS
            if add_waveforms:

                unitWF_h5filePath = temp_hdf5_unitWaveforms_from_electrical_series(
                    unit_dict = units_dict[u], 
                    electrical_series = nwbPrepro.processing['ecephys']['Processed'].electrical_series[es_name], 
                    nwb_electrodes_ids = nwb_electrodes_ids, 
                    tempFolderPath = unitsWF_folder, 
                    wf_n_before = curated_params['n_before'], 
                    wf_n_after = curated_params['n_after'],
                    verbose=verbose
                    )

                unitWF_h5file = h5py.File(name=unitWF_h5filePath, mode="r")

                units_table.add_unit(
                    spike_times = units_dict[u]['spike_times'], 
                    electrodes = units_dict[u]['electrodes'], 
                    electrode_group = units_dict[u]['electrode_group'], 
                    waveform_mean = units_dict[u]['waveform_mean'],
                    waveform_sd = units_dict[u]['waveform_sd'],
                    waveforms = unitWF_h5file["dataSet"],
                    enforce_unique_id=True, 
                    **unit_copy
                    )
                
                unitsWF_h5paths.append(unitWF_h5filePath)
                unitsWF_h5Objs.append(unitWF_h5file)


                """
                if verbose:
                    print('extracting unit {} waveforms from electrical series = {}.......'.format(u, es_name))
                
                # Find the "columns" index that match each unit's electrode IDs
                unit_electrode_mask = [eg_electrodes_index==nwb_electrodes_ids[i] for i in units_dict[u]['electrodes']]

                # Convert spike_times to sample index:
                spike_samples = numpy.round((units_dict[u]['spike_times'] + ephys_eg.starting_time)*ephys_eg.rate).astype(int)
                spike_samples_start = spike_samples - curated_params['n_before']
                spike_samples_stop = spike_samples + curated_params['n_after']
                del spike_samples

                # Get each waveform from each channel
                n_spikes = units_dict[u]['spike_times'].size
                n_electrodes = units_dict[u]['electrodes'].size
                waveforms_unit_array = numpy.zeros((n_spikes, n_electrodes, waveform_samples), dtype=float)
                for t in range(n_spikes):
                    # Check for waveforms samples that might be out of the recording (i.e., the spike time is close to the start ot the end )
                    if spike_samples_start[t]<0:
                        start = 0
                        wf_start = int(abs(spike_samples_start[t]))
                    else:
                        start = spike_samples_start[t]
                        wf_start = int(0)
                    if spike_samples_stop[t]>duration_samples:
                        stop = int(duration_samples)
                    else:
                        stop = spike_samples_stop[t]
                    
                    # Loop over channels
                    for ch in range(n_electrodes):
                        waveforms_unit_array[t, ch, wf_start:wf_start + stop - start] = numpy.squeeze(ephys_eg.data[start:stop, unit_electrode_mask[ch]].astype(float))
                    
                    del start, stop, wf_start
                
                del unit_electrode_mask, spike_samples_start, spike_samples_stop, n_spikes, n_electrodes

                temp_unitsWF_hdf5 = temp_hdf5_from_numpy_with_DataChunkIterator(waveforms_unit_array, tempFolderPath=unitsWF_folder, tempName='unitsWF_u{}'.format(u), verbose=verbose)

                del waveforms_unit_array

                unitsWF_h5paths.append(temp_unitsWF_hdf5)

                unitsWF_h5 = h5py.File(name=temp_unitsWF_hdf5, mode="r")

                units_table.add_unit(
                    spike_times = units_dict[u]['spike_times'], 
                    electrodes = units_dict[u]['electrodes'], 
                    electrode_group = units_dict[u]['electrode_group'], 
                    waveform_mean = units_dict[u]['waveform_mean'],
                    waveform_sd = units_dict[u]['waveform_sd'],
                    waveforms = unitsWF_h5["dataSet"],
                    enforce_unique_id=True, 
                    **unit_copy
                    )
                
                unitsWF_h5Objs.append(unitsWF_h5)
                """

            else:

                units_table.add_unit(
                    spike_times = units_dict[u]['spike_times'], 
                    electrodes = units_dict[u]['electrodes'], 
                    electrode_group = units_dict[u]['electrode_group'], 
                    waveform_mean = units_dict[u]['waveform_mean'],
                    waveform_sd = units_dict[u]['waveform_sd'],
                    waveforms = None,
                    enforce_unique_id=True, 
                    **unit_copy
                    )

            del unit_copy
        
        ######################################################################################################################
        # WRITE PREPROCESSED FILE
        ######################################################################################################################
        if verbose:
            print('\nWriting UNITS table into NWB...........\n')
        nwbPrepro_io.write(nwbPrepro)
        nwbPrepro_io.close()

        del nwbPrepro_io, nwbPrepro

        ######################################################################################################################
        # REMOVE TEMPORAL WAVEFORMS FILES
        ######################################################################################################################
        if add_waveforms:
            if verbose:
                print('\ndeleting Temprary waveforms HDF5 file..........')
                
            for f in unitsWF_h5Objs:
                f.close()
            for f in unitsWF_h5paths:
                if verbose:
                    print('removing {}'.format(f))
                os.remove(f)
                
            if os.path.isdir(unitsWF_folder):
                shutil.rmtree(unitsWF_folder, ignore_errors=True)

        ###############################################################
        #                 RUN nwbInspector
        ###############################################################
        resultsInspector = list(inspect_nwbfile(nwbfile_path=nwbPrepro_path))
        
        if len(resultsInspector)==0:
            if verbose:
                print("congrats¡ no NWB inspector comments\n")
        else:
            print('\nNWB inspector comments:\n')
            for r in resultsInspector:
                if r.severity.value>above_nwbinspectorSeverity: 
                    print('Message : ', r.message)
                    print('Object Name : ',r.object_name)
                    print('Object Type : ',r.object_type)
                    print('Severity : ',r.severity)
                    print('Importance : ', r.importance, '\n')
        if verbose:
            print('\nNWB UNITS preprocessed successfully¡¡ ...........\n')
    

##############################################################################################################
##############################################################################################################
#                                          CREATE NWB ¡¡¡¡¡
##############################################################################################################
def createNWB_raw(filePathYAML=None, 
            Stimulus_Notes=None, KeywordExperiment=None, Experimenters=None, Experiment_Description=None, related_publications=None,
            analogEye = None, analogAccl=None, analogFix=None, analogVisualEvents=None, analogFeet=None, analogReward=None, analogTemp=None,
            expLog_parentFolder=None, TimeZone=None, process_INdisk=True, raw_by_ElectrodeGroup = True, verbose=True):
    
    if verbose:
        computing_start_time = time.time()

    if TimeZone is None:
        TimeZone = labInfo['LabInfo']['TimeZone']

    if filePathYAML is None:
        filePathYAML = askopenfilename(
        title = 'Select a YAML file to extract',
        filetypes =[('yaml Files', '*.yaml')])
                
    if not os.path.isfile(filePathYAML):

        raise Exception("YAML-file-Path : {} doesn't exist ".format(filePathYAML))

    #################################################################################################
    # Check folderSession & Create temporal folder to store intermediate processes
    # In case proccess_INdisk = True, this folder will also be used to copy original files
    folder_save, folder_read, fileName, folder_temporal = check_folderSession_process(filePathYAML, processName_prefix='createNWB', copy2disk=process_INdisk)

    filePath = os.path.join(folder_read, fileName)

    # SET LOG-process file
    orig_stdout = sys.stdout
    orig_sterr = sys.stderr

    fOutputs = open(filePath + '-logOut.txt', 'w')
    
    sys.stdout = Unbuffered(orig_stdout, fOutputs)
    sys.stderr = sys.stdout

    # Check compatible with Temprature:
    tempCompatible = check_temperature_date(fileName=fileName)
    if not tempCompatible:
        analogTemp = None

    # LOAD Log file to get Receptive Fields Information
    expDay_log = None
    if expLog_parentFolder is not None:
        expDay_logPath = os.path.join(expLog_parentFolder, fileName[0:13] + '.xlsx')
        if not os.path.isfile(expDay_logPath):
            raise Exception("Log file NOT FOUND¡¡\nThe current YAML file MUST have a Experiment Log file named {}".format(fileName[0:13] + '.xlsx'))
        else:
            # Read the LOG.xlsx
            if verbose:
                print('\n... loading expLOG file {} into pandas..... \n'.format(fileName[0:13] + '.xlsx'))
            expDay_log = pandas.read_excel(expDay_logPath, sheet_name=None, header=0, index_col=None, usecols=None, dtype=None)

    if verbose:
        print('\n... loading YAML file {} into python-dictionary..... \n'.format(fileName))

    dictYAML = yaml2dict(filePath +'.yaml', verbose=False)
    session_start_time_YAML = expYAML.getStartDateTime(dictYAML, TimeZone)

    if not os.path.isfile(filePath +'.nev'):

        nsFile = None
        session_start_time = session_start_time_YAML
        extNWB = '-noNEV'

        # If there is no NEV file, it is assume that Analog signals were not recorded. 
        # Set all analogs to None to avoid any attempt to extract signals
        analogAccl=None
        analogFix=None
        analogVisualEvents=None
        analogFeet=None
        analogReward=None
        analogTemp=None

    else:
        nsFile = get_nsFile(filePath +'.nev')
        session_start_time = getNS_StartDateTime(nsFile, TimeZone)
        extNWB = ''

    ###############################################################
    #            CREATE Session & Experimental Metadata
    ###############################################################
    sessionInfo = expYAML.getSessionInfo(dictYAML,
                                        session_start_time = session_start_time,
                                        session_id = None, # Default: it will be extracted from dictYAML
                                        session_description = None, # Default: it will be extracted from dictYAML
                                        identifier=None # Default: it will be created automatically
                                        )

    sessionInfo.update(expYAML.getExperimentInfo(dictYAML, 
                            lab=None, # it will be extracted from MonkeyInfo or dictYAML
                            institution=None, # it will be extracted from MonkeyInfo or dictYAML
                            protocol=None, # it will be extracted from MonkeyInfo or dictYAML
                            experiment_description=Experiment_Description, # Default: it will be extracted from dictYAML
                            surgery = None, # it will be extracted from MonkeyInfo
                            experimenter = Experimenters, # It has to be an input
                            stimulus_notes = Stimulus_Notes, # It has to be an input
                            notes = None, # not in use
                            keywords = KeywordExperiment, # It has to be an input
                            related_publications = related_publications  # It has to be an input
                        ))
    
    ###############################################################
    #                       START NWB-file
    ###############################################################
    nwbfile = NWBFile(**sessionInfo)
    
    ###############################################################
    #                        ADD MODULES
    ###############################################################
    # behavior module
    behavior_module = nwbfile.create_processing_module(
        name="behavior", description="processed behavioral data"
        )
    
    ###############################################################
    #                   ADD Subject Info
    ###############################################################
    nwbfile.subject = Subject(**expYAML.getSubjectInfo(dictYAML, 
                            subject_id=None, # it will be extracted from MonkeyInfo or dictYAML
                            description=None, # it will be extracted from MonkeyInfo or dictYAML
                            sex=None, # it will be extracted from MonkeyInfo or dictYAML
                            species=None, # it will be extracted from MonkeyInfo or dictYAML
                            date_of_birth=None, # it will be extracted from MonkeyInfo or dictYAML
                            age=None # not use it
                        ))
    
    ###############################################################
    #            ADD Shaker device Info
    ###############################################################
    shakerInfo = expYAML.getTactorInfo(dictYAML)
    shakerDevice = Device(
        name= 'tactor',
        description = '{} - Model: {} - Number: {}'.format(
            shakerInfo['device']['Description'], 
            shakerInfo['device']['Model'], 
            shakerInfo['device']['ModelNumber']
            ),
        manufacturer = '{} - website: {}'.format(
            shakerInfo['device']['Company'], 
            shakerInfo['device']['Website'])
            )

    nwbfile.add_device(shakerDevice)

    ###############################################################
    #                  ADD TRIALS
    ###############################################################
    # first: add colum names
    for n, d in expYAML.getTrialColNames(dictYAML, analogTemp=analogTemp).items():
        if (n != 'start_time') and (n != 'stop_time'):
            nwbfile.add_trial_column(name=n, description=d)

    # second: GET TRIALS NEV
    trialsNWB_NEV = getNWB_trials(
        dictYAML=dictYAML, 
        nsFile=nsFile, 
        analogAccl=analogAccl, 
        analogFix=analogFix, 
        analogVisualEvents=analogVisualEvents,
        analogTemp = analogTemp,
        verbose=False)
    
    # last: ADD TRIALS
    for n in range(len(trialsNWB_NEV)):
        nwbfile.add_trial(**trialsNWB_NEV[n])

    del trialsNWB_NEV

    ###############################################################
    #                    ADD Eye DATA
    ############################################################### 
    eyeData_h5objs, eyeData_h5paths = nwb_add_eyeData(
        nwb_behavior_module=behavior_module, 
        filePath=filePath, 
        dictYAML=dictYAML, 
        nsFile=nsFile,
        tempFolderPath = folder_temporal,
        analogEye = analogEye,
        eyeTrackID=analogEye['eyeTrackID'],
        eyeXYComments=analogEye['eyeXYComments'],                    
        pupilDiameterComments = analogEye['pupilDiameterComments'],
        verbose = True          
    )

    ###############################################################
    #                    ADD Foot DATA
    ############################################################### 
    if nsFile is not None:
        feet_h5objs, feet_h5paths = nwb_add_footData(
            nwbFile=nwbfile,
            nsFile=nsFile,
            tempFolderPath = folder_temporal,
            analogFeet=analogFeet,
            verbose = True
        )

    ###############################################################
    #                    ADD Reward DATA
    ############################################################### 
    if nsFile is not None:
        reward_h5objs, reward_h5paths = nwb_add_rewardData(
            nwbFile=nwbfile,
            nsFile=nsFile,
            tempFolderPath = folder_temporal,
            analogReward = analogReward,
            verbose = True
        )

    ###############################################################
    #               Add Analog Stimuli
    ###############################################################
    if nsFile is not None:
        analogStim_h5objs, analogStim_h5paths = nwb_add_nsAnalog_stimuli(
            nwbFile = nwbfile, 
            dictYAML = dictYAML,
            nsFile = nsFile,  
            tempFolderPath = folder_temporal,   
            analogAccl = analogAccl, 
            analogFix = analogFix, 
            analogVisualEvents = analogVisualEvents, 
            analogTemp = analogTemp,
            verbose = True
        )

    ###############################################################
    #       Add Raw Neural Signal
    ###############################################################
    if nsFile is not None:
        if raw_by_ElectrodeGroup:
            ###############################################################
            #       dataset per ElectrodeGroup
            ###############################################################
            electrodes_h5objs, electrodes_h5paths = nwb_add_rawElectrodeGroup(
                nwbFile = nwbfile, 
                nsFile = nsFile,
                dictYAML = dictYAML,
                tempFolderPath = folder_temporal,
                expDay_log=expDay_log,
                verbose = True
            )
        else:
            ###############################################################
            #       dataset channel by channel
            ###############################################################
            electrodes_h5objs, electrodes_h5paths = nwb_add_rawElectrode(
                nwbFile = nwbfile, 
                nsFile = nsFile,
                dictYAML = dictYAML,
                tempFolderPath = folder_temporal,
                expDay_log=expDay_log,
                verbose = True
            )

    ###############################################################
    #         Add Electrical MicroStimulation Waveforms
    ###############################################################
    if nsFile is not None:
        nwb_add_stimElectrodesWaveForms(
            nwbFile = nwbfile, 
            nsFile = nsFile,
            dictYAML = dictYAML, 
            expDay_log=expDay_log,
            verbose=True)

    ###############################################################
    #                    Write NWB file
    ###############################################################
    nwbFilePath = os.path.join(filePath + extNWB + '.nwb')
    with NWBHDF5IO(nwbFilePath, "w") as io:
        if verbose:
            print('\nwriting NWB file..........')
        io.write(nwbfile)
    
    ###############################################################
    #       Close and delete hdf5 temporal files
    ###############################################################
    if verbose:
        print('\ndeleting Temprary HDF5 files..........')
    # EYE DATA
    for f in eyeData_h5objs:
        f.close()
    for f in eyeData_h5paths:
        os.remove(f)
    
    if nsFile is not None:
        # FEET DATA
        for f in feet_h5objs:
            f.close()
        for f in feet_h5paths:
            os.remove(f)
        # REWARD DATA
        for f in reward_h5objs:
            f.close()
        for f in reward_h5paths:
            os.remove(f)
        # ANALOG STIM
        for f in analogStim_h5objs:
            f.close()
        for f in analogStim_h5paths:
            os.remove(f)            
        # RAW ELECTRODE GROUPS
        for f in electrodes_h5objs:
            f.close()
        for f in electrodes_h5paths:
            os.remove(f)

    ###############################################################
    # Close nsFiles
    if nsFile is not None:
        for f in nsFile._files:
            f.parser.fid.close()

    ###############################################################
    #                 RUN nwbInspector
    ###############################################################
    resultsInspector = list(inspect_nwbfile(nwbfile_path=nwbFilePath))
    
    if len(resultsInspector)==0:
        print("congrats¡ no NWB inspector comments\n")
    else:
        print('\nNWB inspector comments:\n')
        for r in resultsInspector:
            print('Message : ', r.message)
            print('Object Name : ',r.object_name)
            print('Object Type : ',r.object_type)
            print('Severity : ',r.severity)
            print('Importance : ', r.importance, '\n')

    print("\nNWB file was successfully created¡¡\n{}\n\n".format(fileName + extNWB + '.nwb'))

    ###############################################################
    #         MOVE NWB into the original Path
    ###############################################################
    if folder_save!=folder_read:
        if verbose:
            print('\nmoving NWB into the original directory : {}\n\n'.format(folder_save))
        shutil.move(os.path.join(folder_read, fileName + extNWB + '.nwb'), 
                    os.path.join(folder_save, fileName + extNWB + '.nwb'), 
                    copy_function = shutil.copy2)
    
    if verbose:
        print('\nNWB creation took {} min\n\n'.format((time.time()-computing_start_time)/60))
    
    ###############################################################
    # Close & copy the LOG-file
    ###############################################################
    sys.stdout = orig_stdout
    sys.stderr = orig_sterr
    fOutputs.close()

    if folder_save!=folder_read:
        if verbose:
            print('\nmoving logOut into the original directory : {}\n\n'.format(folder_save))
        shutil.move(os.path.join(folder_read, fileName + '-logOut.txt'), 
                    os.path.join(folder_save, fileName + '-logOut.txt'), 
                    copy_function = shutil.copy2)

    ###############################################################
    # Remove the temporary folder
    ###############################################################
    clear_tempdir(folder_temporal)


##############################################################################################################
##############################################################################################################
#                  CREATE NWB PREPRO FROM RAW: (LFP:low-pass Filter & Downsample some behavioral channels)
##############################################################################################################
def createNWB_prepro(nwbRaw_path, nwbPrepro_parentFolder,
            removeRipple_RawEphys = True, 
            removeRipple_FootBar = True,
            removeRipple_stimulus_list = removeRipple_stimulus_list_default, 
            lfp_params_spikeInterface = lfp_params_spikeInterface_default,
            eyeResample_params = None,
            stimResample_params = None,
            expLog_parentFolder = None,
            updateRF = False, 
            skipMissing_RF = False,
            prepro_suffix = '_prepro',
            add_channel_name = True,
            verbose=True):
    
    ###############################################################################################################################################################
    # Create a *_prepro.NWB file from Original "raw" NWB: 
    ###############################################################################################################################################################
    # Parent folder where to search for expDAY_log.XLSX files that contain Receptive Field Information
    # For S1 recordings in particular, it is assume that at this point this information MUST exists and added at this stage
    # Set to "None" to not search for this information. 
    # expLog_parentFolder = "Y:\\Data_Albus\\Albus_NeuralData\\Albus-S1_logs_xlsx"
    #
    # If True. It will check that all the electrodes/probes and field-names are completely filled, and print a message if somethin is missing
    # If False.  It will throw an error if something is missing
    # skipMissing_RF = False # Default = False
    #
    # Update Receptive field values in case the information already exists in the raw NWB's Electrode Table
    # updateRF = True # Default = True
    #
    ###############################################################################################################################################################
    # Create a *_prepro.NWB file from Original "raw" NWB: 
    # The prepro will perform the following steps:
    #
    ###############################################################################################################################################################
    # 1) Create LFPs : low-pass Filter of the signal coming from all the aquisition groups (Electrode/Probes). You need to provide the lfp filter & sampling params:
    #                  This variable can be loaded from yaulab_processing with the following default values:
    #       lfp_params_spikeInterface_default = {
    #           'lfp_description': 'SpikeInterface Preprocessing: 
    #                               1) BANDPASS (see scipy.signal.iirfilter): 
    #                                       Butterworth filter using second-order sections. 
    #                                       LowFreq: (order:5, freq-corner:0.1); 
    #                                       HighFreq: (order:5, freq-corner:300.0). 
    #                                       Margin ms: 5.0. 
    #                                       Direction: forward-backward. 
    #                               2) Resample (see scipy.signal.resample): Sampling rate 1KHz',
    #           'bandpass': {
    #               'freq_min': 0.1,
    #               'freq_max': 300.0,
    #               'margin_ms': 5.0,
    #               'dtype': None,
    #               'coeff': None,
    #               'add_reflect_padding': False,
    #               'filter_order': 5,
    #               'filter_mode': 'sos',
    #               'ftype': 'butter',
    #               'direction': 'forward-backward'
    #           },
    #           'resample': {
    #                 'resample_rate': 1000,
    #                 'margin_ms': 100.0,
    #                 'dtype': None,
    #                 'skip_checks': False,
    #           }
    #         }
    #
    #
    ###############################################################################################################################################################
    #  You can choose to remove or not the original RAW data. Default = True
    #  removeRipple_RawEphys = True
    #
    ###############################################################################################################################################################
    # 2) Remove Analog stim that are no longer needed or will be downsampled: 
    #    Possible Stimuli to remove and/or Resample: ['fixON', 'visualON', 'rewardON', 'leftAccelerometer', 'leftCommand', 'rightAccelerometer', 'rightCommand', 'thermistors']
    #    Define a list of the stimuli to remove. Accelerometers are kept at the original sampling rate to avoid aliasing
    #    removeRipple_stimulus_list_default = ['fixON', 'visualON', 'rewardON', 'leftCommand', 'rightCommand', 'thermistors']
    #
    ###############################################################################################################################################################
    # 3) Downsampling Analog Stim containers: It will save downsampled Ripple-analog stim channels:
    #    Create a list of dictionaries with the following format:
    #    stimResample_params = {
    #              stim1_NAME: {
    #                           'use_spikeInterface': False, # True or False 
    #                           'resample': {'resample_rate': 1000, 'margin_ms': 100.0, 'dtype': None, 'skip_checks': False} # LFP-Resampling parameters 
    #              },
    #              stim2_NAME: {
    #                           'use_spikeInterface': False, # True or False 
    #                           'resample': {'resample_rate': 1000, 'margin_ms': 100.0, 'dtype': None, 'skip_checks': False} # LFP-Resampling parameters 
    #              },
    #              keep adding stim to resample
    #       }
    #
    #   Default: ONLY "thermistors" will be downsampled:
    #        stimResample_params = {
    #            'thermistors': {
    #                'use_spikeInterface': False,
    #                'resample': lfp_params_spikeInterface_default['resample']
    #            }
    #        }
    #
    ###############################################################################################################################################################
    # 4) Remove Behavior-containers no longer needed or that will be downsampled: Because behavior module has container-specific names and attributes, they will 
    #               need direct/specific call to be removed
    #
    #   removeRipple_FootBar = True
    #
    #   Ripple eye tracking and pupil-diameter will be removed only if they are going to be downsampled. This will be defined by the variable "eyeResample_params":
    #   If eyeResample_params = None it will keep original Ripple eye data. Otherwise should be a dictionary with the following format:
    #   Default:
    #   eyeResample_params = { 'use_spikeInterface': False, # True or False 
    #               'resample': {'resample_rate': 1000, 'margin_ms': 100.0, 'dtype': None, 'skip_checks': False} # LFP-Resampling parameters
    #            }
    #
    ###############################################################################################################################################################
        
    ######################################################################################################################
    # RAW is opened in READ mode to ensure no modifications
    ######################################################################################################################
    nwbRaw_io = NWBHDF5IO(nwbRaw_path, mode="r")
    nwbRaw = nwbRaw_io.read()

    parent_folder_nwb, nwbFileName = os.path.split(nwbRaw_path)

    ######################################################################################################################
    # GET DEFAULT CONTAINERS TO BE REMOVED:
    ######################################################################################################################
    # Containers related to behavior that have been already preprocessed will be removed. 
    # IT WILL ALSO REMOVE ALL THE RAW-EPHYS DATA
    items_to_remove = {}

    if removeRipple_RawEphys:
        items_to_remove.update({'acquisition': [acq for acq in nwbRaw.acquisition.keys() if 'raw-' in acq]})
    
    # Check if Eye DATA will be resampled & removed
    if eyeResample_params is not None:
        items_to_remove.update({
            'eyeTracking': ['eyeRipple'],
            'pupilTracking': ['pupilRipple']
            })
    
    if removeRipple_FootBar:
        items_to_remove.update({'behavior' : ['FootPosition']})
    
    for stim in removeRipple_stimulus_list:

        printWarning = False
        if stim in resample_stimulus_list_default:
            if stimResample_params is None:
                printWarning = True
            elif stim not in stimResample_params:
                printWarning = True

        if printWarning:
            print('\n\nWarning¡¡¡¡\n\tRaw-Stimulus: {} (30K) will be removed and is not in list to be resampled\n\n'.format(stim))

        if 'stimulus' in items_to_remove.keys():
            items_to_remove['stimulus'].append(stim)
        else:
            items_to_remove.update({'stimulus': [stim]})
    
    ######################################################################################################################
    # REMOVE THE CONTAINERS
    ######################################################################################################################
    for key, values in items_to_remove.items():

        for val in values:

            if verbose:
                print('From container {} deleting : "{}" ...'.format(key.upper(), val))

            if key=='stimulus':
                if val in nwbRaw.stimulus:
                    nwbRaw.stimulus.pop(val)

            elif key=='behavior':
                if val in nwbRaw.processing['behavior'].data_interfaces:
                    nwbRaw.processing['behavior'].data_interfaces.pop(val)

            elif key=='eyeTracking':
                keysIN = [ss for ss in nwbRaw.processing['behavior']['EyeTracking'].spatial_series if val in ss]
                for ss in keysIN:
                    nwbRaw.processing['behavior']['EyeTracking'].spatial_series.pop(ss)

            elif key=='pupilTracking':
                keysIN = [ss for ss in nwbRaw.processing['behavior']['PupilTracking'].time_series if val in ss]
                for ss in keysIN:
                    nwbRaw.processing['behavior']['PupilTracking'].time_series.pop(ss)

            elif key=='acquisition':
                if val in nwbRaw.acquisition:
                    nwbRaw.acquisition.pop(val)

    ######################################################################################################################
    # Adds Electrode column "channel_name" for compatibility with NEUROCONV
    if "channel_name" not in nwbRaw.electrodes.colnames and add_channel_name:
        nwbRaw.add_electrode_column(
            name = "channel_name",
            description = 'channel ID',
            data = numpy.array(nwbRaw.electrodes.id[:]).astype("int")
            )

    ######################################################################################################################
    # generate a new set of object IDs
    nwbRaw.generate_new_id()

    ######################################################################################################################
    #                 Export first iteration of the PREPRO file
    ######################################################################################################################
    nwbName_prefix, _ = os.path.splitext(nwbFileName)
    nwbPrepro_path = os.path.join(nwbPrepro_parentFolder, nwbName_prefix + prepro_suffix + '.nwb')

    with NWBHDF5IO(nwbPrepro_path, "w") as io:
        print('\nExporting NON-preprocessed NWB file: {}\n..........\n\n'.format(nwbPrepro_path))
        io.export(src_io = nwbRaw_io, nwbfile = nwbRaw)

    nwbRaw_io.close()

    del nwbRaw_io, nwbRaw

    ######################################################################################################################
    #                  Reload RAW to do some preprocessing on the RAW signal
    #                  Load PREPRO to save the results of the preprocessing
    ######################################################################################################################
    nwbRaw_io = NWBHDF5IO(nwbRaw_path, mode="r")
    nwbRaw = nwbRaw_io.read()

    # Create Temporary directory to save intermediate processing files (hdf5)
    folder_temporal = get_tempdir(processName='{}-{}'.format('nwbPrepro', nwbName_prefix[3::]), resetDir=True)

    ######################################################################################################################
    # ADD LFPs. This function will open-write-close NWB-PREPRO for each electrode group to avoid memory issues.
    ######################################################################################################################
    if lfp_params_spikeInterface is not None:
        nwb_add_lfp_from_nwbRaw_neuroconv(nwbRaw, nwbPrepro_path, 
            tempFolderPath = folder_temporal, 
            lfp_params_spikeInterface=lfp_params_spikeInterface, 
            verbose=verbose
        )

    ######################################################################################################################
    # OPEN FILE IN WRITING MODE TO ADD REMAINING PROCESSED SIGNALS & UPDATE ELECTRODE TABLE
    ######################################################################################################################
    nwbPrepro_io = NWBHDF5IO(nwbPrepro_path, mode="r+")
    nwbPrepro = nwbPrepro_io.read()

    ######################################################################################################################
    # ADD resampled ripple EYE-DATA
    ######################################################################################################################
    # use_spikeInterface:
    #       True: it will use SI resample functio wichi try scipy.signal.decimate first
    #             if there is NaN, then it will do scipy.signal.resample
    #       False: it will use scipy.signal.resample directly
    if eyeResample_params is not None:
        eye_h5objs, eye_h5paths = nwb_resample_nsEyeData(
                nwbRaw_behavior_module = nwbRaw.processing['behavior'], 
                nwbPrepro_behavior_module = nwbPrepro.processing['behavior'],
                resample_params = eyeResample_params['resample'], 
                tempFolderPath = folder_temporal, 
                use_spikeInterface = eyeResample_params['use_spikeInterface'],
                verbose=verbose
            )
    else:
        eye_h5objs = []
        eye_h5paths = []
    
    ######################################################################################################################
    # ADD resampled ripple STIMULUS-DATA
    ######################################################################################################################
    # use_spikeInterface:
    #       True: it will use SI resample functio wichi try scipy.signal.decimate first
    #             if there is NaN, then it will do scipy.signal.resample
    #       False: it will use scipy.signal.resample directly
    stim_h5objs_list = []
    stim_h5paths_list = []
    if stimResample_params is not None:
        for stimulusName, resample_params in stimResample_params.items():
            stim_h5objs, stim_h5paths = nwb_resample_nsAnalog_stimuli(
                                                nwbRaw, 
                                                nwbPrepro, 
                                                stimulusName = stimulusName, 
                                                resample_params = resample_params['resample'],
                                                tempFolderPath = folder_temporal,
                                                use_spikeInterface = resample_params['use_spikeInterface'],
                                                verbose=verbose
                                            )
            if stim_h5objs is not None:
                stim_h5objs_list.append(stim_h5objs)
                stim_h5paths_list.append(stim_h5paths)

    ######################################################################################################################
    # ADD/UPDATE ELECTRODE'S RECEPTIVE FIELDS COLUMNS:
    ######################################################################################################################
    # Making this step at the end, it will help that "neuroconv" is not going to check 
    # for receptive field columns while adding LFPs' electrode table
    # To update Electrodes, the file must be opened in writing mode
    if expLog_parentFolder is not None:
        if verbose:
            print('\nChecking Receptive Field information')
            
        # GET YAML: It assumes the YAML is in the same folder as the nwbRaw
        yaml_filePath = os.path.join(parent_folder_nwb, nwbFileName[0:20] + '.yaml')
        # Read YAML:
        dictYAML = yaml2dict(yaml_filePath, verbose=False)

        # Get expDAY_log.xlsx:
        # Get LOG file path correspondign to the YAML-day
        expDay_logPath = os.path.join(expLog_parentFolder, nwbFileName[0:13] + '.xlsx')
        # Read the LOG.xlsx
        expDay_log = pandas.read_excel(expDay_logPath, sheet_name=None, header=0, index_col=None, usecols=None, dtype=None)

        nwb_update_electrodeTable_ReceptiveFields(nwbFile=nwbPrepro, dictYAML=dictYAML, expDay_log=expDay_log, updateRF=updateRF, skipMissing_RF=skipMissing_RF, verbose=verbose)

    ######################################################################################################################
    # WRITE PREPROCESSED FILE
    ######################################################################################################################
    print('\nWriting preprocessed "eyeData" and "StimData" into NWB...........\n')
    nwbPrepro_io.write(nwbPrepro)

    nwbPrepro_io.close()
    nwbRaw_io.close()

    ######################################################################################################################
    #       Close and delete temporal files
    ######################################################################################################################
    if verbose:
        print('\ndeleting Temprary HDF5 files..........')
    # EYE DATA
    for f in eye_h5objs:
        f.close()
    for f in eye_h5paths:
        if verbose:
            print('removing {}'.format(f))
        os.remove(f)
    # STIM DATA
    for f in stim_h5objs_list:
        f.close()
    for f in stim_h5paths_list:
        if verbose:
            print('removing {}'.format(f))
        os.remove(f)

    if verbose:
        print('\nPreprocessing DONE¡..........\n\n')

    ###############################################################
    #                 RUN nwbInspector
    ###############################################################
    resultsInspector = list(inspect_nwbfile(nwbfile_path=nwbPrepro_path))
    
    if len(resultsInspector)==0:
        print("congrats¡ no NWB inspector comments\n")
    else:
        print('\nNWB inspector comments:\n')
        for r in resultsInspector:
            print('Message : ', r.message)
            print('Object Name : ',r.object_name)
            print('Object Type : ',r.object_type)
            print('Severity : ',r.severity)
            print('Importance : ', r.importance, '\n')

    print('\nNWB preprocessed was created successfully¡¡ ...........\n')

    ######################################################################################################################
    # Remove ALL remaining temporary files & folders 
    ######################################################################################################################
    clear_tempdir(folder_temporal)


####################################################################################################################################################################################
#                                                      MAIN FUNCTION TO EXPORT A CURATED FOLDER INTO EACH OF ITS SESSIONS
#       IT WILL CREATE THE SORTING ANALYZER (RECORDING & SORTING) PER SESSION AND IT WILL EXPORT THEM INTO ITS CORRESPONDING preproNWB-FILE:
####################################################################################################################################################################################
def write_curatedFolder_into_nwb_sessions(curatedFolder_path, parentFolder_preproNWB, 
    start_unit_id=None, 
    ms_before=ms_before_default, 
    ms_after=ms_after_default, 
    add_waveforms = True,
    rewriteAnalyzer = False,
    rewriteNWB_recording = False,
    rewriteNWB_units = False,
    removeAnalyzer = True,
    removeRecording = True,
    removeCuration = True,
    verbose = True
    ):
    
    #
    ###############################################################################################################################################################
    # "curatedFolder_path" # Default = None
    # THE CURATION FOLDER PATH from where to extract RECORDING & SORTING. If set to "None", it will open a message window asking for the curated folder
    ###############################################################################################################################################################
    #
    ###########################################################################################################################################
    # "ms_before" # Default = 0.6
    # "ms_after" # Default = 1.5
    # GENERAL PARAMETER FOR "WAVEFORM TEMPLATES & LOCATIONS"
    # The best practice is to match these values to the ones used during sorting.
    ###########################################################################################################################################
    #
    ###########################################################################################################################################
    # "start_unit_id" # Default = None
    # If = integer # it will reset the unitID with consecutive numbers, starting at variable value
    # If = None # it will use the unitID from PHY/sorting_analyzer as "unit_name"
    ###########################################################################################################################################
    #
    ###########################################################################################################################################
    # "add_waveforms" # Default = True
    # Whether or not to ADD waveforms from each spike time. 
    # This information will be extracted from the recording already imported to the NWB at:
    # "nwb.processing.ecephys.ephys-ElectrodeGroup" 
    ###########################################################################################################################################
    #
    ###########################################################################################################################################
    # "rewriteAnalyzer" # default = False
    # If the curated folder has been already splitted into sessions and exported as sorting_analyzer(s)
    # it will read the existing temporal folders instead of computing the sorting_analyzer again. 
    # Useful in case something goes wrong while exporting to NWB and there is no need to re-compute it.
    ###########################################################################################################################################
    #
    ###########################################################################################################################################
    # "rewriteNWB_recording" # Default = False
    # If = False, and the recording has been already written into the NWB, it will be skipped. 
    # If = True, and the recording has been already written into the NWB, it will be removed and rewrite it. 
    ###########################################################################################################################################
    #
    ###########################################################################################################################################
    # "rewriteNWB_units" # Default = False
    # If = False, and the units table has been already written into the NWB, it will be skipped. 
    # If = True, and If the units table has been already written into the NWB, it will be removed and rewrite it. 
    ###########################################################################################################################################
    #
    ###########################################################################################################################################
    # "removeAnalyzer" # Default = False
    # Whether or not to delete the recording(s) & sorting_analyzer(s) created per session, after exporting the recordings & units from all sessions
    ###########################################################################################################################################
    #
    ###########################################################################################################################################
    # "removeCuratedFolder" # Default = False
    # Whether or not to delete the binary recording associated with the curated folder (the concatenated recording used to sort & curate)
    # The curated PHY/Sorting_analyzer folder will not be deleted for now
    ###########################################################################################################################################
    #
    ###########################################################################################################################################
    # "verbose" # Default = True
    # Whether or not to print messages related to the exporting process
    ###########################################################################################################################################

    sortingAnalyzerSessions_info2export = export_curatedFolder_to_sortingAnalyzer_sessions(
        curatedFolder_path, 
        start_unit_id=start_unit_id, 
        ms_before=ms_before, 
        ms_after=ms_after, 
        rewriteAnalyzer=rewriteAnalyzer,
        verbose = verbose
        )

    curated_params = sortingAnalyzerSessions_info2export['curated_params']
    sorting_analyzer_params = sortingAnalyzerSessions_info2export['sorting_analyzer_params']

    for si_recording_folder in  sortingAnalyzerSessions_info2export['recordingFolder_list']:

        nwb_add_processed_si_recording_SIdataChunkIterator(
            si_recording_folder=si_recording_folder, 
            parentFolder_preproNWB=parentFolder_preproNWB, 
            rewrite = rewriteNWB_recording,
            verbose=verbose,
            above_nwbinspectorSeverity=1 # It will run nwb-inspector, print message according to "above_nwbinspectorSeverity" (HIGH = 2, LOW = 1). Set to "0" it will print all
            )

    for sorting_analyzer_folder in sortingAnalyzerSessions_info2export['sortingAnalyzerFolder_list']:

        nwb_add_units_from_sorting_analyzer_session(
            sorting_analyzer_folder = sorting_analyzer_folder, 
            parentFolder_preproNWB = parentFolder_preproNWB, 
            curated_params = curated_params, 
            sorting_analyzer_params = sorting_analyzer_params, 
            add_waveforms=add_waveforms, 
            rewrite=rewriteNWB_units, 
            verbose=verbose, 
            above_nwbinspectorSeverity=1 # It will run nwb-inspector, print message according to "above_nwbinspectorSeverity" (HIGH = 2, LOW = 1). Set to "0" it will print all
            )
        
    if removeAnalyzer:
        for sorting_analyzer_folder in sortingAnalyzerSessions_info2export['sortingAnalyzerFolder_list']:
            clear_tempdir(sorting_analyzer_folder)
        for si_recording_folder in  sortingAnalyzerSessions_info2export['recordingFolder_list']:
            clear_tempdir(si_recording_folder)

    if removeRecording:
        print('WARNING ¡¡¡ you are going to delete the binary recording associated with the curated sorting')
        clear_tempdir(curated_params['si_recording_folder'])

        if curated_params['curated_with_phy'] and not removeCuration:
            
            with open(os.path.join(curated_params['phy_or_sorter_analyzer_path'] / "params.py"), 'r') as file: 
                paramsPHY_data = file.readlines() 
        
            paramsPHY_data[0] = f"dat_path = {None}\n"
        
            # Update recording path in params.py
            with open(os.path.join(curated_params['phy_or_sorter_analyzer_path'] / "params.py"), "w") as f:
                f.writelines(paramsPHY_data)
    
    if removeCuration:
        if curated_params['curated_with_phy']:
            print('WARNING ¡¡¡ you are going to delete the PHY folder associated with the curated sorting')
        else:
            print('WARNING ¡¡¡ you are going to delete the SortingAnalyzer folder associated with the curated sorting')

        clear_tempdir(curated_params['phy_or_sorter_analyzer_path'])

