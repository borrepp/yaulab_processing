from ..yaulab_extras import labInfo
from .yaml_tools import expYAML
from .ripple_tools import AnalogIOchannel, getTrialMarkers, get_stimTimeStamps, get_rawElectrodeInfo
import numpy
import matplotlib.pyplot as plt
import datetime
import os
import time
import h5py
from scipy import signal
from hdmf.data_utils import DataChunkIterator


# TOOLS to extract & combine Behavioral Data from *.nev (markers) & *.YAML files into a NWB file 
# file format follow YAULAB convention of the task : visual cues + 2 shakers + footbar + eye tracking

version = '0.0.1'

#######################################################################
# SOME DEFAULT PARAMS:
#######################################################################

defaultNoTime = float(labInfo['StimDefaults']['NoTime'])
add2stimDuration = float(labInfo['SecsToAddAccelerometer'][0])
add2stimStop = float(labInfo['AddFirstStimAccelerometer']['Secs'][0])
numTrials2addStopTime = labInfo['AddFirstStimAccelerometer']['trialNum'][0]

tolerancePhotodiode = float(labInfo['MarkerOffsetTolerance']['PhotodiodeON'][0])
add2fixOnset = float(labInfo['SecsToAddPhotodiode'][0])
timeZone = labInfo['LabInfo']['TimeZone']

thresholdReward_mV = float(labInfo['ThresholdReward_mV'])
thresholdFeet_mV = float(labInfo['ThresholdFeet_mV'])

# AnalogIOchannels nomenclature : 
# chanName          ChanNum
# 'leftCommand':        1,
# 'leftAccelerometer':  2,
# 'rightCommand':       3,
# 'rightAccelerometer': 4,
# 'eyeHorizontal':      5,
# 'pupilDiameter':      6,
# 'leftFoot':           7,
# 'rightFoot':          8,
# 'rewardON':           9,
# 'fixON':              10,
# 'visualON':           11,
# 'leftProbeTEMP':      12,
#

########################################################################################################################
# SOME FUNCTIONS TO PROCESS RIPPLE & YAML DATA
########################################################################################################################

##################################################################################################
# Plot a segment of time from an Analog class
def plot_analogEvent(analogIOchannel_cls, stimStartSecs, stimStopSecs):
    
    stimStart_index = analogIOchannel_cls.get_timeIndex([stimStartSecs])[0]
    stimStop_index = analogIOchannel_cls.get_timeIndex([stimStopSecs])[0]

    analogIOchannel_cls.get_indexTime(range(stimStart_index-100, stimStop_index+100))

    tSnippet1 = analogIOchannel_cls.get_indexTime(range(stimStart_index, stimStop_index))
    snippet1 = analogIOchannel_cls.get_data(start_index=stimStart_index, index_count=stimStop_index-stimStart_index)

    fig, axs1 = plt.subplots()

    axs1.set_title(analogIOchannel_cls.get_info()['chanName'])
    axs1.plot(tSnippet1, snippet1, color='C0')
    axs1.set_xlabel("time (s)")
    axs1.set_ylabel("Voltage")

    plt.show()

##################################################################################################
# Extract a TimeSeries from Ripple (nsFile) as a dataset ("hdf5" file). 
# First dimension is Time, all entities to be extracted must match on time-dimension (e.g., sampleRate)
# It will concatenate all the entityIndexes as columns
def temp_TimeSeries_hdf5(nsFile, entityIndexes, itemCount, tempFolderPath, tempName,
            dtype='float', 
            compression="gzip", 
            compression_opts=4,
            chunks=True, 
            verbose=True):
    
    # Get fileNamePath
    tempFilePath = os.path.join(tempFolderPath, "{}.hdf5".format(tempName))

    # Check if there is already a file with this name
    if os.path.isfile(tempFilePath):
        try:
            os.remove(tempFilePath)
        except:
            print('Warning:\nUnable to delete the file:\n{}\nprobably is still open or in use\n\n'.format(tempFilePath))
    
    # Ceate the h5 file
    h5Temporal = h5py.File(name=tempFilePath, mode="w")
    
    nChans = len(entityIndexes)

    # 1D dataset
    if nChans==1:
        dset = h5Temporal.create_dataset(name="dataSet", 
                            shape=(itemCount,), 
                            dtype=dtype,
                            compression=compression, 
                            compression_opts=compression_opts,
                            chunks=chunks)
        
        nChunks_1d = int(numpy.ceil(dset.shape[0]/dset.chunks[0]))
        chunksInfo = []
        for t in range(nChunks_1d):
            start_i = int(t*dset.chunks[0])
            stop_i = min(dset.shape[0], (start_i+dset.chunks[0]))
            chunksInfo.append({
                'start_i': start_i,
                'stop_i': stop_i,
                'nCount': int(stop_i - start_i)
                })
        
        startTime = time.time()
        nChunks = len(chunksInfo)
        if verbose:
            print('extracting: {}  ....'.format(tempName))
        nsEntity_i = nsFile.get_entity(entityIndexes[0])
        for i in range(nChunks):
            if verbose:
                if (time.time()-startTime)>5:
                    print("extracting: {}  .... chunk {} out of {}...........".format(tempName, i, nChunks))
                    startTime = time.time()
            
            dset[chunksInfo[i]['start_i']:(chunksInfo[i]['stop_i'])] = nsEntity_i.get_analog_data(
                start_index=chunksInfo[i]['start_i'], index_count=chunksInfo[i]['nCount'])
    
    # 2D dataset        
    else:
        dset = h5Temporal.create_dataset(name="dataSet", 
                            shape=(itemCount, nChans), 
                            dtype=dtype,
                            compression=compression, 
                            compression_opts=compression_opts,
                            chunks=chunks)
    
        # USE HDF5 chunk Size to extract data
        # first dimension is time, 2nd dimension is Channels:
        nChunks_1d = int(numpy.ceil(dset.shape[0]/dset.chunks[0]))
        nChunks_2d = int(numpy.ceil(dset.shape[1]/dset.chunks[1]))

        chunksInfo = []
        # Chunks will be extracted by time first, and channels second
        for c in range(nChunks_2d):
            chan_i = int(c*dset.chunks[1])
            chan_e = int(min(dset.shape[1], (chan_i+dset.chunks[1])))
            for t in range(nChunks_1d):
                start_i = int(t*dset.chunks[0])
                stop_i = min(dset.shape[0], (start_i+dset.chunks[0]))
                chunksInfo.append({
                    'start_i': start_i,
                    'stop_i': stop_i,
                    'nCount': int(stop_i - start_i), 
                    'col_i': numpy.arange(chan_i, chan_e),
                    'entity_i': [int(entityIndexes[i]) for i in range(chan_i, chan_e)]
                })
        
        startTime = time.time()
        nChunks = len(chunksInfo)
        if verbose:
            print('extracting: {}  ....'.format(tempName))
        for i in range(nChunks):
            if verbose:
                if (time.time()-startTime)>5:
                    print("extracting: {}  .... chunk {} out of {}...........".format(tempName, i, nChunks))
                    startTime = time.time()
            
            for e in range(len(chunksInfo[i]['entity_i'])):
                dset[chunksInfo[i]['start_i']:(chunksInfo[i]['stop_i']), chunksInfo[i]['col_i'][e]] = nsFile.get_entity(
                    chunksInfo[i]['entity_i'][e]).get_analog_data(start_index=chunksInfo[i]['start_i'], index_count=chunksInfo[i]['nCount'])
            
    h5Temporal.close()

    return tempFilePath

def temp_TimeSeries_hdf5_analog_cls(nsFile, analog_chanNames, itemCount, tempFolderPath, tempName,
            dtype='float', 
            compression="gzip", 
            compression_opts=4,
            chunks=True, 
            verbose=True):
    
    analogEntities = []
    for chanName in analog_chanNames:
        analogEntities.append(AnalogIOchannel(nsFile=nsFile, chanName=chanName))

    # Get fileNamePath
    tempFilePath = os.path.join(tempFolderPath, "{}.hdf5".format(tempName))

    # Check if there is already a file with this name
    if os.path.isfile(tempFilePath):
        try:
            os.remove(tempFilePath)
        except:
            print('Warning:\nUnable to delete the file:\n{}\nprobably is still open or in use\n\n'.format(tempFilePath))
    
    # Ceate the h5 file
    h5Temporal = h5py.File(name=tempFilePath, mode="w")
    
    nChans = len(analog_chanNames)

    # 1D dataset
    if nChans==1:

        dset = h5Temporal.create_dataset(name="dataSet", 
                            shape=(itemCount,), 
                            dtype=dtype,
                            compression=compression, 
                            compression_opts=compression_opts,
                            chunks=chunks)
        
        nChunks_1d = int(numpy.ceil(dset.shape[0]/dset.chunks[0]))
        chunksInfo = []
        for t in range(nChunks_1d):
            start_i = int(t*dset.chunks[0])
            stop_i = min(dset.shape[0], (start_i+dset.chunks[0]))
            chunksInfo.append({
                'start_i': start_i,
                'stop_i': stop_i,
                'nCount': int(stop_i - start_i)
                })
        
        startTime = time.time()
        nChunks = len(chunksInfo)
        if verbose:
            print('extracting: {}  ....'.format(tempName))

        nsEntity_i = analogEntities[0]
        for i in range(nChunks):
            if verbose:
                if (time.time()-startTime)>5:
                    print("extracting: {}  .... chunk {} out of {}...........".format(tempName, i, nChunks))
                    startTime = time.time()
            
            dset[chunksInfo[i]['start_i']:(chunksInfo[i]['stop_i'])] = nsEntity_i.get_data(
                start_index=chunksInfo[i]['start_i'], index_count=chunksInfo[i]['nCount'])
            
    # 2D dataset        
    else:
        dset = h5Temporal.create_dataset(name="dataSet", 
                            shape=(itemCount, nChans), 
                            dtype=dtype,
                            compression=compression, 
                            compression_opts=compression_opts,
                            chunks=chunks)
    
        # USE HDF5 chunk Size to extract data
        # first dimension is time, 2nd dimension is Channels:
        nChunks_1d = int(numpy.ceil(dset.shape[0]/dset.chunks[0]))
        nChunks_2d = int(numpy.ceil(dset.shape[1]/dset.chunks[1]))

        chunksInfo = []
        # Chunks will be extracted by time first, and channels second
        for c in range(nChunks_2d):
            chan_i = int(c*dset.chunks[1])
            chan_e = int(min(dset.shape[1], (chan_i+dset.chunks[1])))
            for t in range(nChunks_1d):
                start_i = int(t*dset.chunks[0])
                stop_i = min(dset.shape[0], (start_i+dset.chunks[0]))
                chunksInfo.append({
                    'start_i': start_i,
                    'stop_i': stop_i,
                    'nCount': int(stop_i - start_i), 
                    'col_i': numpy.arange(chan_i, chan_e),
                    'entity_i': [i for i in range(chan_i, chan_e)]
                })
        
        startTime = time.time()
        nChunks = len(chunksInfo)
        if verbose:
            print('extracting: {}  ....'.format(tempName))
        for i in range(nChunks):
            if verbose:
                if (time.time()-startTime)>5:
                    print("extracting: {}  .... chunk {} out of {}...........".format(tempName, i, nChunks))
                    startTime = time.time()
            
            for e in range(len(chunksInfo[i]['entity_i'])):
                dset[chunksInfo[i]['start_i']:(chunksInfo[i]['stop_i']), chunksInfo[i]['col_i'][e]] = analogEntities[chunksInfo[i]['entity_i'][e]].get_data(
                    start_index=chunksInfo[i]['start_i'], index_count=chunksInfo[i]['nCount']
                    )
            
    h5Temporal.close()

    return tempFilePath

def temp_hdf5_unitWaveforms_from_electrical_series(unit_dict, electrical_series, nwb_electrodes_ids, tempFolderPath, wf_n_before, wf_n_after, 
            compression="gzip", 
            compression_opts=4,
            chunks=True,
            verbose=True):

    if verbose:
        print('extracting unit {} waveforms.......'.format(unit_dict['unit_name']))

    # Get general information
    electrical_series_electrodes_index = electrical_series.electrodes.to_dataframe().index.to_numpy()
    duration_samples = electrical_series.data.shape[0]
    es_dtype = electrical_series.data.dtype

    # Find the "columns" index that match each unit's electrode IDs
    unit_electrode_mask = [electrical_series_electrodes_index==nwb_electrodes_ids[i] for i in unit_dict['electrodes']]

    # Convert spike_times to sample index:
    spike_samples = numpy.round((unit_dict['spike_times'] + electrical_series.starting_time)*electrical_series.rate).astype(int)
    spike_samples_start = spike_samples - wf_n_before
    spike_samples_stop = spike_samples + wf_n_after
    del spike_samples

    # Get each waveform from each channel
    n_spikes = unit_dict['spike_times'].size
    n_electrodes = unit_dict['electrodes'].size

    # Ceate the h5 file
    tempFilePath = os.path.join(tempFolderPath, "unit_u{}.hdf5".format(unit_dict['unit_name']))
    h5Temporal = h5py.File(name=tempFilePath, mode="w")

    # Ceate the h5 dataset
    dset = h5Temporal.create_dataset(
        name="dataSet", 
        shape=(n_spikes, n_electrodes, wf_n_before + wf_n_after), 
        dtype=es_dtype,
        compression=compression, 
        compression_opts=compression_opts,
        chunks=chunks
        )
    
    # Loop over spike events
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
        
        # Loop over channels to ensure that channel order matches with units' electrode IDS
        for ch in range(n_electrodes):
            dset[t, ch, wf_start:wf_start + stop - start] = numpy.squeeze(electrical_series.data[start:stop, unit_electrode_mask[ch]].astype(es_dtype))
        
        del start, stop, wf_start

    del unit_electrode_mask, spike_samples_start, spike_samples_stop, n_spikes, n_electrodes

    h5Temporal.close()
        
    return tempFilePath


def temp_hdf5_from_numpy(array, tempFolderPath, tempName,
            compression="gzip", 
            compression_opts=4,
            chunks=True,
            verbose=True):
    
    # Get fileNamePath
    tempFilePath = os.path.join(tempFolderPath, "{}.hdf5".format(tempName))

    # Check if there is already a file with this name
    if os.path.isfile(tempFilePath):
        try:
            os.remove(tempFilePath)
        except:
            print('Warning:\nUnable to delete the file:\n{}\nprobably is still open or in use\n\n'.format(tempFilePath))
    
    # Ceate the h5 file
    h5Temporal = h5py.File(name=tempFilePath, mode="w")

    if verbose:
        print('creating temporal hdf5 dataset: {}  ....'.format(tempName))

    h5Temporal.create_dataset(name="dataSet",
                data=array, 
                compression=compression, 
                compression_opts=compression_opts,
                chunks=chunks)
    
    h5Temporal.close()

    return tempFilePath

def temp_hdf5_from_numpy_with_DataChunkIterator(array, tempFolderPath, tempName,
            compression="gzip", 
            compression_opts=4,
            chunks=True,
            verbose=True):
    
    # Get fileNamePath
    tempFilePath = os.path.join(tempFolderPath, "{}.hdf5".format(tempName))

    # Check if there is already a file with this name
    if os.path.isfile(tempFilePath):
        try:
            os.remove(tempFilePath)
        except:
            print('Warning:\nUnable to delete the file:\n{}\nprobably is still open or in use\n\n'.format(tempFilePath))
    
    # Create DataChunkIterator
    chunk_mb = 10.0
    if array.ndim>1:
        chunk_channels = min(array.shape[1], 64)  # from https://github.com/flatironinstitute/neurosift/issues/52#issuecomment-1671405249

        buffer_size = min(array.shape[0],
            int(chunk_mb * 1e6 / (array.dtype.itemsize * chunk_channels)),
        )
    else:
        buffer_size = min(array.shape[0],
            int(chunk_mb * 1e6 / array.dtype.itemsize),
        )
    numpy_DataChunkIterator = DataChunkIterator(data=array, buffer_size=buffer_size)

    # Ceate the h5 file
    h5Temporal = h5py.File(name=tempFilePath, mode="w")

    if verbose:
        print('creating temporal hdf5 dataset: {}  ....'.format(tempName))

    dset = h5Temporal.create_dataset(name="dataSet",
                compression = compression, 
                compression_opts = compression_opts,
                shape = numpy_DataChunkIterator.maxshape,
                dtype = numpy_DataChunkIterator.dtype,
                chunks = chunks
                )
    
    for buffer in numpy_DataChunkIterator:
        dset[buffer.selection] = buffer.data
    
    h5Temporal.close()

    return tempFilePath


def temp_hdf5_from_HDMFGenericDataChunkIterator(HDMFGenericDataChunkIterator, tempFolderPath, tempName,
            compression="gzip", 
            compression_opts=4,
            verbose=True):
    
    # Get fileNamePath
    tempFilePath = os.path.join(tempFolderPath, "{}.hdf5".format(tempName))

    # Check if there is already a file with this name
    if os.path.isfile(tempFilePath):
        try:
            os.remove(tempFilePath)
        except:
            print('Warning:\nUnable to delete the file:\n{}\nprobably is still open or in use\n\n'.format(tempFilePath))
    
    # Ceate the h5 file
    h5Temporal = h5py.File(name=tempFilePath, mode="w")

    if verbose:
        print('creating temporal hdf5 dataset: {}  ....'.format(tempName))

    dset = h5Temporal.create_dataset(name="dataSet",
                compression = compression, 
                compression_opts = compression_opts,
                shape = HDMFGenericDataChunkIterator.maxshape,
                dtype = HDMFGenericDataChunkIterator.dtype,
                chunks = HDMFGenericDataChunkIterator.chunk_shape
                )
    
    for buffer in HDMFGenericDataChunkIterator:
        dset[buffer.selection] = buffer.data
    
    h5Temporal.close()

    return tempFilePath
    

def temp_resample_hdf5_dataset(hdf5dataset, dataset_rate, resample_rate, tempFolderPath, tempName, margin_ms=100, add_reflect_padding = True, add_zeros = False, conversion=1.0, offset=0.0, verbose=True):

    # Pipeline copied from SpileInterface
    if hdf5dataset.ndim>1:
        print('\nWARNING¡¡¡¡ \nThe dataset has more than one channel (nChans = {})\nIt can create memory issues'.format(hdf5dataset.shape[1]))
    
    # Get a margin for the original sampling
    margin = int(margin_ms * dataset_rate / 1000)
    
    # get margin for the resampled case
    margin_rs = int((margin / dataset_rate) * resample_rate)

    paddding_list = [(margin, margin)]
    for d in range(hdf5dataset.ndim-1):
        paddding_list.append((0,0))

    if add_reflect_padding:
        dataset_pad = numpy.pad(hdf5dataset[:], paddding_list, mode="reflect")
    elif add_zeros:
        dataset_pad = numpy.pad(hdf5dataset[:], paddding_list, mode="constant", constant_values=0.0)

    num = int(dataset_pad.shape[0] / dataset_rate * resample_rate)

    dataset_resampled = (signal.resample(dataset_pad, num, axis=0) * conversion) + offset

    del dataset_pad
    
    resampled_h5path = temp_hdf5_from_numpy(array=dataset_resampled[margin_rs : num - margin_rs], tempFolderPath=tempFolderPath, tempName=tempName, verbose=verbose)

    return resampled_h5path

##################################################################################################
# Get the offset between eyePC timeStamps and NS-zeroTime
def get_eyePC_offset(dictYAML, eyePCstartTime, nsFile=None):

    eyeDelta = datetime.timedelta(hours = eyePCstartTime.hour, 
                              minutes= eyePCstartTime.minute, 
                              seconds = eyePCstartTime.second, 
                              microseconds = eyePCstartTime.microsecond)

    eyeOFFset = eyeDelta.total_seconds()-expYAML.getStartTimeSecs(dictYAML)

    if nsFile is not None:
        nsTrialInfo = getTrialMarkers(nsFile)
        trialsNEV = nsTrialInfo['trials']
        startTrialNev = trialsNEV[0]['markerTime'][trialsNEV[0]['markerID'].index(1)]
    else:
        startTrialNev = 0

    return startTrialNev + eyeOFFset

def get_averageTemp(analogTemp_cls, ti = None, tf = None):
        
    therm_info = analogTemp_cls.get_info()

    nSamples = int(therm_info['item_count'])
    endTime = analogTemp_cls.get_indexTime([nSamples])[0]

    if ti is None:
        ti = 0
        
    if tf is None:
        tf = endTime

    if tf>endTime:
        print('Warning... TimeEnd to calculate average temprature is out of range, it will limited to the end of the recording')
        tf = endTime
        
    start_index = analogTemp_cls.get_timeIndex([ti])[0]
    stop_index = analogTemp_cls.get_timeIndex([tf])[0]

    return numpy.mean(analogTemp_cls.get_data(start_index, stop_index - start_index))
    


##################################################################################################
# Extract Foot Hold, Left, Right, Both as behavioral Events
# return Dictionary easy to convert to pd.dataframe & NWB-trial table
# colNames : 'start_time', 'stop_time', 'feetResponseID', 'feetResponse'
def get_feetEvents(nsFile, chunkSizeSecs = 60, showPlot = False):

    eventID_description = [
        'holdBoth', # 0
        'releaseLeft', # 1
        'releaseRight', # 2
        'releaseBoth', # 3
    ]

    nsAnalogLeftFeet = AnalogIOchannel(nsFile=nsFile, chanName='leftFoot')
    nsAnalogRightFeet = AnalogIOchannel(nsFile=nsFile, chanName='rightFoot')

    diffTime = 0.001 # time before change of status
    leftInfo = nsAnalogLeftFeet.get_info()

    nSamples = int(leftInfo['item_count'])
    endTime = nsAnalogLeftFeet.get_indexTime([nSamples])[0]

    chunkSize = int(numpy.ceil(chunkSizeSecs*leftInfo['samplingRate']))

    chunkStart = [int(i) for i in range(0, nSamples, chunkSize)]

    if len(chunkStart)==1:
        chunkStop = [nSamples]
    else:
        if chunkStart[-1]==(nSamples-1):
            chunkStart.pop(-1)

        chunkStop = chunkStart[1:]
        chunkStop.append(nSamples)
    
    nChunks = len(chunkStart)

    eventTime = []
    eventID = []

    for c in range(nChunks):

        print('processing footSignals..... chunk {} out of {}.....'.format(c+1, nChunks))

        start_index = chunkStart[c]
        count_samples = int(chunkStop[c]-chunkStart[c])

        leftData = nsAnalogLeftFeet.get_data(start_index, count_samples)
        rightData = nsAnalogRightFeet.get_data(start_index, count_samples)

        leftRight = (leftData>thresholdFeet_mV) + ((rightData>thresholdFeet_mV)*2)
        events = numpy.nonzero(abs(numpy.diff(leftRight))>0)[0]

        appendZERO = False
        if c==0:
            appendZERO = True
        else:
            # check if previous sample was the same
            leftData_prev = nsAnalogLeftFeet.get_data(start_index-1, 1)>thresholdFeet_mV
            rightData_prev = nsAnalogRightFeet.get_data(start_index-1, 1)>thresholdFeet_mV

            leftRight_prev = leftData_prev + (rightData_prev*2)

            if leftRight_prev!=leftRight[0]:
                # print(leftRight_prev, leftRight[0])
                appendZERO = True
        
        if len(events)==0 and c==0:
            eventTime.append(start_index/leftInfo['samplingRate'])
            eventID.append(leftRight[0])
        elif len(events)>0:
            if appendZERO:
                eventTime.append(start_index/leftInfo['samplingRate'])
                eventID.append(leftRight[0])
            for e in events:
                eventTime.append((e+start_index)/leftInfo['samplingRate'])
                eventID.append(leftRight[e+1])


        if showPlot:
            fig, axs1 = plt.subplots()
            tData = nsAnalogLeftFeet.get_indexTime(range(start_index, start_index+count_samples))

            axs1.set_title(nsAnalogLeftFeet.get_info()['chanName'] + ' and ' + nsAnalogRightFeet.get_info()['chanName'])
            axs1.plot(tData, leftData-10, color='b')
            axs1.plot(tData, rightData+10, color='g')
            for i in range(len(eventTime)):
                if eventTime[i]>=tData[0] and eventTime[i]<=tData[-1]:
                    axs1.vlines(x=eventTime[i], ymin= -0.3, ymax = 4000, colors='r')
                    axs1.text(x=eventTime[i], y=4005, s=str(eventID[i]),horizontalalignment='center',
                        verticalalignment='center', fontsize=8)
            axs1.set_xlim(tData[0]-1, tData[-1]+1)

            plt.show()

    # Convert feetEvents into a dictionary of intervals
    # list of dictionaries {start_time:,  stop_time:, feetID, feetDescription}
    nEv = len(eventID)

    feetIntervals = []
    for e in range(nEv):

        if e==(nEv-1):
            stopTime = endTime
        else:
            stopTime = eventTime[e+1] - diffTime

        if stopTime<=eventTime[e]:
            stopTime += diffTime

        feetIntervals.append({
            'start_time': eventTime[e],
            'stop_time': stopTime,
            'feetResponseID': eventID[e],
            'feetResponse': eventID_description[eventID[e]]
        })
    
    return feetIntervals

##################################################################################################
# Extract Reward ON as a Dictionary colNames : 'start_time', 'stop_time'
def get_rewardEvents(analogReward_cls, chunkSizeSecs = 60, showPlot = False):

    diffTime = 0.001 # time before change of status
    rewardInfo = analogReward_cls.get_info()

    nSamples = int(rewardInfo['item_count'])
    endTime = analogReward_cls.get_indexTime([nSamples])[0]

    chunkSize = int(numpy.ceil(chunkSizeSecs*rewardInfo['samplingRate']))

    chunkStart = [int(i) for i in range(0, nSamples, chunkSize)]
    
    if len(chunkStart)==1:
        chunkStop = [nSamples]
    else:
        if chunkStart[-1]==(nSamples-1):
            chunkStart.pop(-1)

        chunkStop = chunkStart[1:]
        chunkStop.append(nSamples)
    
    nChunks = len(chunkStart)

    eventTime = []
    eventID = []

    for c in range(nChunks):

        print('processing Reward Signal..... chunk {} out of {}.....'.format(c+1, nChunks))

        start_index = chunkStart[c]
        count_samples = int(chunkStop[c]-chunkStart[c])

        rewardData = analogReward_cls.get_data(start_index, count_samples)

        rewardON = rewardData>thresholdReward_mV
        events = numpy.nonzero(abs(numpy.diff(rewardON))>0)[0]

        appendZERO = False
        if c==0:
            appendZERO = True
        else:
            # check if previous sample was the same
            rewardData_prev = analogReward_cls.get_data(start_index-1, 1)>thresholdReward_mV

            if rewardData_prev!=rewardON[0]:
                appendZERO = True
        
        if len(events)>0:
            if appendZERO:
                eventTime.append(start_index/rewardInfo['samplingRate'])
                eventID.append(rewardON[0])
            
            for e in events:
                eventTime.append((e+start_index)/rewardInfo['samplingRate'])
                eventID.append(rewardON[e+1])

        if showPlot:
            fig, axs1 = plt.subplots()
            tData = analogReward_cls.get_indexTime(range(start_index, start_index+count_samples))

            axs1.set_title(analogReward_cls.get_info()['chanName'])
            axs1.plot(tData, rewardData, color='b')
            for i in range(len(eventTime)):
                if eventTime[i]>=tData[0] and eventTime[i]<=tData[-1]:
                    if eventID[i]:
                        label = 'ON'
                        y = 5005
                    else:
                        label = 'OFF'
                        y = -0.5
                    axs1.vlines(x=eventTime[i], ymin= -0.3, ymax = 5000, colors='r')
                    axs1.text(x=eventTime[i], y=y, s=label,horizontalalignment='center',
                        verticalalignment='center', fontsize=8)
            axs1.set_xlim(tData[0]-1, tData[-1]+1)

            plt.show()

    # Convert feetEvents into a dictionary of intervals
    # list of dictionaries {start_time:,  stop_time:, feetID, feetDescription}
    nEv = len(eventID)

    rewardIntervals = []
    for e in range(nEv):
        # Only save RewardON eventID ==1
        if eventID[e]==1:

            if e==(nEv-1):
                stopTime = endTime
            else:
                stopTime = eventTime[e+1] - diffTime

            if stopTime<=eventTime[e]:
                stopTime += diffTime

            rewardIntervals.append({
                'start_time': eventTime[e],
                'stop_time': stopTime,
                'label': 'rewardON',
                'labelID': eventID[e]
            })

    return rewardIntervals

#######################################################################################################
#                                 !! NO LONGER IN USE !!! 
# This function does not take into account that the reward can be delivered manually at random times
# Reward times are saved as timeIntervals (like foot events)
#######################################################################################################
#
# Get Reward ON & OFF times
def get_trialRewardONOFF_fromAnalog(analogReward_cls, trialNEV, interTrialTime, showPlot=True):

    threshold_mV = 2500
    diffTime = 0.000 # time before change of status

    rewardInfo = analogReward_cls.get_info()

    ####################################################
    # Marker 1: fixationTarget is ON the screen
    signal_start = trialNEV['markerTime'][trialNEV['markerID'].index(1)]
    
    ####################################################
    # Stop of the trial 
    signal_stop = max(trialNEV['markerTime']) + (interTrialTime/1.2)

    start_index = analogReward_cls.get_timeIndex([signal_start])[0]
    stop_index = analogReward_cls.get_timeIndex([signal_stop])[0]

    signal = analogReward_cls.get_data(start_index=start_index, index_count=stop_index-start_index)

    peaks = numpy.nonzero(signal>threshold_mV)[0]

    if len(peaks)>0:
        timeUP = analogReward_cls.get_indexTime([start_index+peaks[0]-1])[0]-diffTime
        timeDOWN = analogReward_cls.get_indexTime([start_index+peaks[-1]-1])[0]-diffTime
    else:
        timeUP = []
        timeDOWN = []

    if showPlot:

        fig, axs1 = plt.subplots()

        tData = analogReward_cls.get_indexTime(range(start_index, stop_index))

        axs1.set_title(rewardInfo['chanName'] + ' - Trial: {}'.format(trialNEV['trialNum']))
        axs1.plot(tData, signal, color='b')
        if len(peaks)>0:
            axs1.vlines(x=timeUP, ymin= -0.3, ymax = 5000, colors='r')
            axs1.text(x=timeUP, y=5005, s='on', horizontalalignment='center',
                            verticalalignment='center', fontsize=8)
            axs1.vlines(x=timeDOWN, ymin= -0.3, ymax = 5000, colors='r')
            axs1.text(x=timeDOWN, y=5005, s='off', horizontalalignment='center',
                            verticalalignment='center', fontsize=8)
        axs1.set_xlim(tData[0]-1, tData[-1]+1)

        plt.show()
    
    return {'rewardON': timeUP, 'rewardOFF': timeDOWN}

##################################################################################################
# Get FIX_ON and FIX_OFF based on photodiode signal. It will use marker 1 to cut the signal. 
# To get the ON & OFF event it will normalize between the min & the max and use 
# the variable  normThreshold as a cut point
def get_trialFixONOFF_fromAnalog(analogFix_cls, trialNEV, interTrialTime, normThreshold, showPlot=False):

    ####################################################
    # Marker 1: fixationTarget is ON the screen
    fixTarget_Start = trialNEV['markerTime'][trialNEV['markerID'].index(1)]-tolerancePhotodiode
    if trialNEV['trialNum']==1:
        signal_start = 0
    else:
        signal_start  = fixTarget_Start - (interTrialTime/1.2)
    
    ####################################################
    # Stop of the trial 
    signal_stop = max(trialNEV['markerTime']) + (interTrialTime/1.2)

    signal_start_index = analogFix_cls.get_timeIndex([signal_start])[0]
    signal_stop_index = analogFix_cls.get_timeIndex([signal_stop])[0]
    
    if signal_stop_index>analogFix_cls.get_info()['item_count']:
        signal_stop_index = analogFix_cls.get_info()['item_count']

    signal = analogFix_cls.get_data(start_index=signal_start_index, index_count=signal_stop_index-signal_start_index)

    minSignal = min(signal)
    maxSignal = max(signal)

    stepSignal =  ( (signal - minSignal) / (maxSignal - minSignal) )>= normThreshold

    if stepSignal[0]:
        stepSignal = stepSignal==False

    peaks = numpy.nonzero(stepSignal)[0]

    indexUP = peaks[0]
    timeUP = analogFix_cls.get_indexTime([signal_start_index+indexUP])[0]

    indexDOWN = peaks[-1]
    timeDOWN = analogFix_cls.get_indexTime([signal_start_index+indexDOWN])[0]

    if showPlot:

        tSnippet = analogFix_cls.get_indexTime(range(signal_start_index, signal_stop_index))

        fig, axs1 = plt.subplots()

        axs1.set_title(analogFix_cls.get_info()['chanName'] + ' - Trial: {}'.format(trialNEV['trialNum']))
        axs1.plot(tSnippet, signal, color='k')
        axs1.vlines(x=timeUP, ymin= minSignal, ymax = maxSignal, colors='r')
        axs1.vlines(x=timeDOWN, ymin= minSignal, ymax = maxSignal, colors='b')
        axs1.set_xlabel("Time (s)")
        axs1.set_ylabel("Amplitude (mV)")

        plt.show()
    
    return {'fixON': timeUP, 'fixOFF': timeDOWN}

####################################################################################################################
# Get VISUAL-EVENT_ON based on photodiode signal. It search for marker 15 (Visual Cue ON) or 5 (Choice Targets ON). 
# To get the ON of any visual event it will normalize between the min & the max and use 
# the variable  normThreshold as a cut point. To detect the next event it will use the photodiodeDuration
def get_trialVisualEventsON_fromAnalog(analogVisualEvents_cls, trialNEV, normThreshold,  
            photodiodeDuration, choiceTargetON=True, showPlot=False, showWarningPlot=True):
    
    samplesPulse = (photodiodeDuration*analogVisualEvents_cls.get_info()['samplingRate'])

    visualEvents = {'cueON': [], 'choiceON': []}

    # Check if marker 15 (Visual Cue ON) or 5 (Choice Targets ON) occurs
    if any([trialNEV['markerID'].count(15), trialNEV['markerID'].count(5)]):

        # Sort by time
        nevIndex = numpy.argsort(numpy.array(trialNEV['markerTime']))
        nNEV = len(trialNEV['markerTime'])
        
        if choiceTargetON:
            marker_idx = [nevIndex[i] for i in range(nNEV) if trialNEV['markerID'][nevIndex[i]]==15 or trialNEV['markerID'][nevIndex[i]]==5]
        else:
            marker_idx = [nevIndex[i] for i in range(nNEV) if trialNEV['markerID'][nevIndex[i]]==15]
            
        visEv_ID = [trialNEV['markerID'][i] for i in marker_idx]
        visEv_Time = [trialNEV['markerTime'][i] for i in marker_idx]

        signal_index = analogVisualEvents_cls.get_timeIndex([min(trialNEV['markerTime']), max(trialNEV['markerTime'])])

        signal = analogVisualEvents_cls.get_data(start_index=signal_index[0], index_count=signal_index[1] - signal_index[0])

        minSignal = min(signal)
        maxSignal = max(signal)

        stepSignal =  ( (signal - minSignal) / (maxSignal - minSignal) )>= normThreshold

        samplesStep = numpy.floor(samplesPulse*0.5)

        signalUP = numpy.nonzero(stepSignal)[0]
        lenUP = numpy.size(signalUP)
        diffUP = numpy.diff(signalUP)
        upON = numpy.concatenate((numpy.array([0]), numpy.nonzero(diffUP>samplesStep)[0]+1))
        nUPs = numpy.size(upON)
        if nUPs>1:
            upOFF = numpy.concatenate((upON[1:]-1, [lenUP-1]))
        else:
            upOFF = numpy.array([lenUP-1])

        validPeaks = (upOFF - upON)>samplesStep

        tVisualEvents = analogVisualEvents_cls.get_indexTime(signal_index[0] + signalUP[upON[validPeaks]])

        warningPlotexist = False
        raise_Exception = False
        # Create New Markers 
        # 1st check if number of visual events Matches
        if len(tVisualEvents)==len(visEv_Time):
            for m in range(len(visEv_ID)):
                if visEv_ID[m]==15:
                    visualEvents['cueON'].append(tVisualEvents[m])
                elif visEv_ID[m]==5:
                    visualEvents['choiceON'].append(tVisualEvents[m])

        else:
            # Possible artifact on the photodiode can give more than ON events than expected
            if len(tVisualEvents)>len(visEv_Time):
                if showWarningPlot:
                    warningPlotexist = True
                print('\nWARNING¡¡\nNumber of VisualEvents did NOT match:\nmarkerIDs: {}, n={} vs Detected={}\
                    \nmarkerTimes: {}\ntimesDetected: {}\nIt will take the closest timeStamp and ignore the other\n'.format(
                        visEv_ID, len(visEv_Time), len(tVisualEvents), 
                        visEv_Time, tVisualEvents)
                    )
                for m in range(len(visEv_ID)):
                    mID = visEv_ID[m]
                    if mID==15:
                        mName = 'cueON'
                    elif mID==5:
                        mName = 'choiceON'
                    closestAnalog = tVisualEvents[numpy.abs(numpy.asarray(tVisualEvents)-visEv_Time[m]).argmin()]
                    absDifference = abs(visEv_Time[m]-closestAnalog)
                    if absDifference<=tolerancePhotodiode:
                        newMarkerTime = closestAnalog
                    else:
                        warningPlotexist = True
                        newMarkerTime = min([closestAnalog, visEv_Time[m]])
                        print('Photodiode Time for marker {}({}) was not found closer to the expected time by the configFile.\n\
                            it will take the earlier timeStamp between YAML and photodiode.\n\
                            yaml timeStamp: {},\n\
                            closest Photodiode (NEV) timeStamp: {},\n\
                            absDifference (secs): {},\n\
                            TrialNev: {}\n\
                            Tolerance (secs): {}\n'.format(
                                mID, mName, visEv_Time[m], closestAnalog,
                                absDifference, trialNEV['trialNum'], tolerancePhotodiode))
                    
                    visualEvents[mName].append(newMarkerTime)

            elif len(tVisualEvents)<len(visEv_Time):
                raise_Exception = True
                warningPlotexist = True
                print('Number of VisualEvents did NOT match:\nmarkerIDs: {}, n={} vs Detected={}\
                        \nmarkerTimes: {}\ntimesDetected: {}\n'.format(visEv_ID, len(visEv_Time), len(tVisualEvents), 
                                                                    visEv_Time, tVisualEvents)
                        )

        # PLOT
        if showPlot or warningPlotexist:
            tSignal = analogVisualEvents_cls.get_indexTime(range(signal_index[0], signal_index[1]))

            fig, axs1 = plt.subplots()

            axs1.set_title(analogVisualEvents_cls.get_info()['chanName'] + ' Trial: {}'.format(trialNEV['trialNum']))
            axs1.plot(tSignal, signal, color='k')
            axs1.vlines(x=trialNEV['markerTime'], ymin= min(signal), ymax = max(signal), colors='r')
            axs1.vlines(x=visualEvents['cueON'], ymin= min(signal), ymax = max(signal), colors='g')
            axs1.vlines(x=visualEvents['choiceON'], ymin= min(signal), ymax = max(signal), colors='b')
            for m in range(len(trialNEV['markerTime'])):
                axs1.text(x=trialNEV['markerTime'][m], y=max(signal), s=str(trialNEV['markerID'][m]), 
                        horizontalalignment='center', verticalalignment='center', fontsize=8)

            plt.show()
        
        if raise_Exception:
            raise Exception()

    return visualEvents

##################################################################################################
#                                 !! NO LONGER IN USE !!!  
# software was updated. Now there are two photodiodes to signal fixation-only and to signal ONSET 
# of any other visual event (visual cues & choice targets)
##################################################################################################
#
# If FIX-photodiode is placed on top of the real fixation target, it can be possible to extract 
# ONSET & OFFSET of visual cues draw on top of the fixation:
#   It will use FIX_ON and FIX_OFF period to serch for changes in the photodiode signal. 
#   It will use marker 15 as a starting point to search for the first change.
#   It will return visualCueON, OFF, and FIX_ON & FIX_OFF as a dictionary

def get_trialFixVisualEvents_fromAnalog(analogFix_cls, trialNEV, threshold_mV, minGap_cueOFF_fixOFF, 
                interTrialTime, normThreshold, showPlot=False):
    
    # It will get the main ON-OFF event, using the min & max values 
    time_ONOFF = get_trialFixONOFF_fromAnalog(analogFix_cls, trialNEV, interTrialTime, normThreshold, showPlot=showPlot)

    cueON = []
    cueOFF = []

    nVisualCue = trialNEV['markerID'].count(15)

    # Assume that signal always start after marker 15
    if nVisualCue>0:

        if nVisualCue>1:
            print('\n....... WARNING:\nfunction "get_visualCueATfix_fromAnalog" has not been fully tested\
                  to handle more than one visualCue at fixationLocation\n(trial {} has {} visualCues)'.format(
                      trialNEV['trialNum'], nVisualCue
                  ))
        visualCuesON = [trialNEV['markerTime'][i] for i in range(len(trialNEV['markerID'])) if trialNEV['markerID'][i]==15]

        signal_start_index = analogFix_cls.get_timeIndex([time_ONOFF['fixON']])[0]
        signal_stop_index = analogFix_cls.get_timeIndex([time_ONOFF['fixOFF']])[0]

        signal = analogFix_cls.get_data(start_index=signal_start_index, index_count=signal_stop_index-signal_start_index)

        stepMin = numpy.ceil(minGap_cueOFF_fixOFF*analogFix_cls.get_info()['samplingRate']).astype(int)

        for v in range(len(visualCuesON)):

            cueON_index = analogFix_cls.get_timeIndex([visualCuesON[v]])[0]

            cueSignal = signal[cueON_index-signal_start_index:]
            stepSignal = numpy.nonzero(abs(cueSignal-cueSignal[0])>=threshold_mV)[0]

            if len(stepSignal)>0:

                cueON_time = analogFix_cls.get_indexTime([cueON_index + stepSignal[0]])[0]
                cueON.append(cueON_time)

                jump_index = numpy.nonzero(numpy.diff(stepSignal)>=stepMin)[0]
                if len(jump_index)>0:
                    cueOFF.append(analogFix_cls.get_indexTime([cueON_index + stepSignal[jump_index[0]-1]])[0])
                else:
                    cueOFF.append(time_ONOFF['fixOFF'])

    if showPlot:

        if nVisualCue==0:

            visualCuesON = [trialNEV['markerTime'][i] for i in range(len(trialNEV['markerID'])) if trialNEV['markerID'][i]==15]

            signal_start_index = analogFix_cls.get_timeIndex([time_ONOFF['fixON']])[0]
            signal_stop_index = analogFix_cls.get_timeIndex([time_ONOFF['fixOFF']])[0]

            signal = analogFix_cls.get_data(start_index=signal_start_index, index_count=signal_stop_index-signal_start_index)

        tSignal = analogFix_cls.get_indexTime(range(signal_start_index, signal_stop_index))

        fig, axs1 = plt.subplots()

        axs1.set_title('Fix Epoch - Trial: {}'.format(trialNEV['trialNum']))
        axs1.plot(tSignal, signal, color='k')
        axs1.vlines(x=cueON, ymin= min(signal), ymax = max(signal), colors='r')
        axs1.vlines(x=cueOFF, ymin= min(signal), ymax = max(signal), colors='r')
        axs1.vlines(x=visualCuesON, ymin= min(signal), ymax = max(signal), colors='b')

        plt.show()

    # Get Unique Times from cueON & cueOFF
    if len(cueON)>0:
        cueON = list(numpy.unique(numpy.array(cueON)))
    if len(cueOFF)>0:
        cueOFF = list(numpy.unique(numpy.array(cueOFF)))

    time_ONOFF.update({'cueON': cueON, 'cueOFF': cueOFF})

    return time_ONOFF
        
###########################################################################################
# Get Onset and Offset of an signal that crosses a threshold. It find the first and the last peak and use 
# the midlle point of those peaks to center the signal according to the expected duration of the event.
def get_acclONOFF_fromAnalog(analogAccl_cls, stimStartSecs, stimStopSecs, stimDurationSecs=None, 
                            thresholdHigh_std=15, thresholdLow_std=5, baselineStartSecs=-1, baselineStopSecs=-1, 
                            showPlot=False, showPlot_lowThreshold=True, showPlot_noDetected=True,
                            trialID = None, stimID = None):
    if trialID is None:
        trialID = 'unknown'
    if stimID is None:
        stimID = 'unknown'
        
    samplingRate = analogAccl_cls.get_info()['samplingRate']

    stimStart_index = analogAccl_cls.get_timeIndex([stimStartSecs])[0]
    stimStop_index = analogAccl_cls.get_timeIndex([stimStopSecs])[0]

    if baselineStartSecs <0:
        baselineStart_index = stimStart_index
    else:
        baselineStart_index = analogAccl_cls.get_timeIndex([baselineStartSecs])[0]

    if baselineStopSecs <0:
        baselineStop_index = stimStop_index
    else:
        baselineStop_index = analogAccl_cls.get_timeIndex([baselineStopSecs])[0]


    stdBaseline = numpy.std(analogAccl_cls.get_data(start_index=baselineStart_index, 
                                             index_count=baselineStop_index-baselineStart_index))
    
    snippet = analogAccl_cls.get_data(start_index=stimStart_index, 
                                           index_count=stimStop_index-stimStart_index)
    snippet = numpy.absolute((snippet - numpy.mean(snippet))/stdBaseline)
    peaks_index_high = numpy.nonzero(snippet>=thresholdHigh_std)

    if peaks_index_high[0].size>0:
        peaks_index = peaks_index_high
        threshold_std = thresholdHigh_std
        showPlot_lowThreshold=False
    else:
        peaks_index = numpy.nonzero(snippet>=thresholdLow_std)
        threshold_std = thresholdLow_std
        print('{} at Trial={}, stim={}, timeInterval: [{}-{}] used the low Threshold'.format(
            analogAccl_cls.get_info()['chanName'], trialID, stimID, stimStartSecs, stimStopSecs
        ))
    
    if peaks_index[0].size>0:

        threshold2 = numpy.mean(snippet[peaks_index]) - (numpy.std(snippet[peaks_index])*1.5)

        if threshold2>threshold_std:
            peaks = numpy.nonzero(snippet>=threshold2)
        else:
            peaks = numpy.nonzero(snippet>=threshold_std)

        centerPeak = numpy.min(peaks) + numpy.round((numpy.max(peaks)-numpy.min(peaks))/2).astype(int)

        if stimDurationSecs is None:
            stimDurationSecs = (numpy.max(peaks)-numpy.min(peaks))/samplingRate
        
        durationSamples = numpy.ceil(stimDurationSecs*samplingRate).astype(int)
        half = numpy.ceil(durationSamples/2).astype(int)

        tStart_index = stimStart_index + centerPeak - half
        tStop_index = stimStart_index + centerPeak + half

        if showPlot or showPlot_lowThreshold:

            tSnippet = analogAccl_cls.get_indexTime(range(stimStart_index, stimStop_index))
            tSnippet1 = analogAccl_cls.get_indexTime(range(stimStart_index-20, stimStop_index+20))
            tSnippet2 = analogAccl_cls.get_indexTime(range(tStart_index, tStop_index+1))

            snippet1 = analogAccl_cls.get_data(start_index=stimStart_index-20, index_count=stimStop_index-stimStart_index+40)
            snippet2 = analogAccl_cls.get_data(start_index=tStart_index, index_count=tStop_index-tStart_index+1)

            fig, (axs1, axs2) = plt.subplots(1, 2)

            axs1.set_title('{} at Trial={}, stim={}'.format(analogAccl_cls.get_info()['chanName'], trialID, stimID))
            axs1.plot(tSnippet, snippet, color='C0')
            axs1.plot([tSnippet[p] for p in peaks[0]], snippet[peaks], "x", color='olive')
            axs1.set_xlabel("# sample")
            axs1.set_ylabel("Amp (Z-score)")

            axs2.set_title('snippet')
            axs2.plot(tSnippet1, snippet1, 'r-')
            axs2.plot(tSnippet2, snippet2, color='C0')
            axs2.set_xlabel("# sample")
            axs2.set_ylabel("Amplitude")

            plt.show()

        acclONOF = [analogAccl_cls.get_indexTime([tStart_index])[0], analogAccl_cls.get_indexTime([tStop_index])[0]]
    
    else:
        acclONOF = [defaultNoTime, defaultNoTime]

        print('\nWARNING¡¡¡\nNo Accelerometer events were detected in {} at Trial={}, stim={}\
              \nThreshold std={}. Time interval = [{}, {}]'.format(
            analogAccl_cls.get_info()['chanName'], threshold_std,  trialID, stimID, stimStartSecs, stimStopSecs
        ))

        if showPlot or showPlot_noDetected:

            tSnippet = numpy.arange(0, stimStop_index-stimStart_index)
            tSnippet1 = numpy.arange(stimStart_index-20, stimStop_index+20)
            snippet1 = analogAccl_cls.get_data(start_index=stimStart_index-20, index_count=stimStop_index-stimStart_index+40)

            fig, (axs1, axs2) = plt.subplots(1, 2)

            axs1.set_title('{}\nTrial={}, stim={}'.format(analogAccl_cls.get_info()['chanName'], trialID, stimID))
            axs1.plot(tSnippet, snippet, color='C0')
            axs1.hlines(y=threshold_std, xmin= min(tSnippet), xmax = max(tSnippet), colors='r')
            axs1.set_xlabel("# sample")
            axs1.set_ylabel("Amp (Z-score)")

            axs2.set_title('snippet')
            axs2.plot(tSnippet1, snippet1, 'r-')
            axs2.set_xlabel("# sample")
            axs2.set_ylabel("Amplitude")

            plt.show()
    
    return acclONOF

#def get_microStimONOOFF_fromNEV()
############################################################################################################
# Return a list of YAML-markerID with the times from NEV-markerTime. If there is a YAML-marker that do not 
# exist in the NEV-markers, it will extrapolate the time-difference and include it. NEV-markers that do not
# exist in the YAML-markers will be omitted
def mergeMarkerTimes(markerID_nev, markerTime_nev, markerID_yaml, markerTime_yaml, trialNum, verbose=False):

    # Sort by time
    nevIndex = numpy.argsort(numpy.array(markerTime_nev))
    nNEV = len(markerID_nev)
    copyNEV = [markerID_nev[nevIndex[i]] for i in range(nNEV)]
    copyNEVTimes = [markerTime_nev[nevIndex[i]] for i in range(nNEV)]

    yamlIndex = numpy.argsort(numpy.array(markerTime_yaml))
    nYAML = len(markerID_yaml)
    copyYAML = [markerID_yaml[yamlIndex[i]] for i in range(nYAML)]
    copyYAMLTimes = [markerTime_yaml[yamlIndex[i]] for i in range(nYAML)]

    # Get NEV markers that exist in YAML markers
    markerID = []
    markerTime = []
    markerTime_YAML = []

    yamlID_remain = []
    yamlTimes_remain = []

    for i in range(nYAML):
        # Marker 1 is always the first one
        if i==0:
            if copyYAML[0]==1 and copyNEV[0]==1:

                markerID.append(copyNEV[0])
                markerTime.append(copyNEVTimes[0])

                markerTime_YAML.append(copyYAMLTimes[0])

                copyNEV.pop(0)
                copyNEVTimes.pop(0)

            else:
                raise Exception(' MARKERS should start with marker 1')
        else:
            # Start extracting first marker from NEV that matches YAML
            if copyNEV.count(copyYAML[i])>0:
                index_in_nev = copyNEV.index(copyYAML[i])

                markerID.append(copyNEV[index_in_nev])
                markerTime.append(copyNEVTimes[index_in_nev])

                markerTime_YAML.append(copyYAMLTimes[i])

                copyNEV.pop(index_in_nev)
                copyNEVTimes.pop(index_in_nev)
            else:
                yamlID_remain.append(copyYAML[i])
                yamlTimes_remain.append(copyYAMLTimes[i])

    # Report YAML markers not found in NEV markers
    nRemain = len(yamlID_remain)
    if nRemain>0 and verbose:
        print('YAML markers: {} not found on NEV (trial: {})'.format(yamlID_remain, trialNum)) 
        print('YAML markers: {} \nNEV markers: {}\n'.format(markerID_yaml, markerID_nev)) 
    
    # Find closest YAML markers and add the timespan to the final list
    for i in range(nRemain):
        yamlID_i = yamlID_remain[i]
        yamlTime_i = yamlTimes_remain[i]

        index_closest = numpy.abs(numpy.asarray(markerTime_YAML)-yamlTime_i).argmin()
        diffTime = yamlTime_i - markerTime_YAML[index_closest]

        markerID.append(yamlID_i)
        markerTime.append(markerTime[index_closest] + diffTime)

    # Some INFO ABOUT THE NEV markers not Found in YAML
    if len(copyNEV)>0:
        print('NEV markers: {} not found on YAML (trial: {})\n'.format(copyNEV, trialNum)) 
        if verbose:
            print('NEV markers: {} \nYAML markers: {}\n'.format(markerID_nev, markerID_yaml))  

    # Return data sorted by time
    nMarkers = len(markerID)
    markerIndex = numpy.argsort(numpy.array(markerTime))
    return [markerID[markerIndex[i]] for i in range(nMarkers)], [markerTime[markerIndex[i]] for i in range(nMarkers)]

######################################################################################################
# USE MARKER IDs & TimeStamps from NEV(ripple) to merge with YAML-trial into a Dictionary Format
def updateMarkerTime(trialYAML, trialNEV,
                    analogAccl=None,
                    analogFix=None, 
                    analogVisualEvents=None, 
                    analogTemp = None,
                    microStimulation = None,
                    choiceTargetON = True,
                    verbose=True):

    # Merge markerTimes
    print('...merging TimeStamps from NEV into YAML format (trial: {})'.format(trialNEV['trialNum']))

    markerID, markerTime = mergeMarkerTimes(trialNEV['markerID'], trialNEV['markerTime'], 
                                    trialYAML['markerID'], trialYAML['markerTime'], 
                                    trialNum=trialNEV['trialNum'], verbose=verbose)
    
    # update OutcomeTime
    outcomeTime = markerTime[markerID.index(trialYAML['outcomeID'])]

    #########################################################################################################################
    # GET stimTimeON & stimTimeOFF and StimsDict
    #########################################################################################################################
    nMarkers = len(markerID)
    stimTimeON = [markerTime[indx] for indx in range(nMarkers) if markerID[indx]==3]
    if stimTimeON is None:
        nStimON = 0
    else:
        nStimON = len(stimTimeON)

    stimTimeOFF_marker4 = [markerTime[indx] for indx in range(nMarkers) if markerID[indx]==4]
    if stimTimeOFF_marker4 is None:
        nStimOFF_marker = 0
    else:
        nStimOFF_marker = len(stimTimeOFF_marker4)

    if (nStimON-nStimOFF_marker)>1:
        print('WARNING¡ Trial {} has more than one stimTimeON (marker3) without stimTimeOFF (marker4).'.format(trialYAML['ID'])) 
        print('To get stimTimeOFF it will use the next stimTimeON (minus{}) or the outcomeTime'.format(labInfo['SecsToStopPreviousStim'][0]))
        print('markerID: {}\nstimTimeON: {}\nstimTimeOFF: {}\noutcomeTime: {}'.format(markerID, stimTimeON, stimTimeOFF_marker4))  
    elif (nStimON-nStimOFF_marker)<0:
        raise Exception('It was detected more stimTimeOFF than stimTimeON\nmarkerID: {}\nstimTimeON: {}\nstimTimeOFF: {}\noutcomeTime: {}'.format(
            markerID, stimTimeON, stimTimeOFF_marker4))

    stimTimeOFF = []
    for s in range(nStimON):
            if s<nStimOFF_marker:
                stimTimeOFF.append(stimTimeOFF_marker4[s])
            elif s<(nStimON-1):
                stimTimeOFF.append(stimTimeON[s+1]-labInfo['SecsToStopPreviousStim'][0])
            else:
                stimTimeOFF.append(outcomeTime)

    nStimOFF = len(stimTimeOFF)

    dictStims = [value for key, value in trialYAML.items() if key.startswith('Stim ')]
    if nStimON>len(dictStims):
        print('On trial {} there were more stimON (marker3) than YAML Stim-parameters\nnStimON = {}, n-stimYAML: {}'.format(trialNEV['trialNum'], nStimON, len(dictStims)))
        print('markerIDs : ', markerID, '\nstimON (marker3) : ', stimTimeON, '\n\nStim Dictionaries:\n')
        for dictStim in dictStims:
            print(dictStim, '\n')


    #####################################################################################################################
    # Create AnalogTimes for Tactile Stimulation
    #####################################################################################################################
    if analogAccl is not None:

        if analogAccl['exists']:

            analogLeftAccl = analogAccl['nsAnalog'][0]
            analogRightAccl = analogAccl['nsAnalog'][1]

            tactStimParams = [expYAML.getTactileStimParams(value) for key, value in trialYAML.items() if key.startswith('Stim ')]
            
            if nStimON>0:
                bslnStart = markerTime[markerID.index(1)]
                bslnStop = stimTimeON[0]

            # APPEND new MARKERS
            # 3001 = LeftON, 
            # 4001 = LeftOFF
            # 3002 = RightON, 
            # 4002 = rightOFF
            for s in range(nStimON):
                ############################
                # If marker 3 occurs, Tactile will be delivered even if the subject abort in the middle of the stim.
                # Get Tactile STIM

                if s==0 and trialNEV['trialNum']<numTrials2addStopTime:
                    secs2addStop = add2stimDuration
                    if nStimON>1:
                        if stimTimeON[s+1]-stimTimeOFF[s]<add2stimDuration:
                            secs2addStop = stimTimeON[s+1]-stimTimeOFF[s]
                else:
                    secs2addStop = 0.0

                markerID.append(3001)
                markerID.append(4001)
                markerID.append(3002)            
                markerID.append(4002)

                if analogAccl['showPlot']:
                    print('\ngetting Accelerometer signal: stim {} of trial {}'.format(s+1, trialNEV['trialNum']))
                    print(tactStimParams[s])
                    if trialNEV['trialNum']<numTrials2addStopTime:
                        print('secs2add: {}'.format(secs2addStop)) 

                if tactStimParams[s]['leftValid']:
                    leftDurationSecs = (tactStimParams[s]['left']['Duration']/1000)+add2stimDuration
                    leftOFF = stimTimeON[s] + (tactStimParams[s]['left']['Start Time']/1000) + leftDurationSecs # leftShaker endTime
                    if leftOFF>=stimTimeOFF[s]:
                        stopStim = leftOFF+add2stimDuration+secs2addStop
                    else:
                        stopStim = stimTimeOFF[s]+add2stimDuration+secs2addStop
                    leftONOFF = get_acclONOFF_fromAnalog(analogLeftAccl, stimTimeON[s], stopStim, stimDurationSecs=leftDurationSecs, 
                                    thresholdHigh_std=analogAccl['thresholdHigh_std'], 
                                    thresholdLow_std = analogAccl['thresholdLow_std'], 
                                    baselineStartSecs=bslnStart, baselineStopSecs=bslnStop, 
                                    showPlot=analogAccl['showPlot'],
                                    showPlot_lowThreshold=analogAccl['showPlot_lowThreshold'], 
                                    showPlot_noDetected=analogAccl['showPlot_noDetected'],
                                    trialID = trialNEV['trialNum'],
                                    stimID = s+1
                                    )
                    
                    markerTime.append(leftONOFF[0])
                    markerTime.append(leftONOFF[1])

                else:
                    markerTime.append(defaultNoTime)
                    markerTime.append(defaultNoTime)

                if tactStimParams[s]['rightValid']:
                    rightDurationSecs = (tactStimParams[s]['right']['Duration']/1000) + add2stimDuration
                    rightOFF = stimTimeON[s] + (tactStimParams[s]['right']['Start Time']/1000) + rightDurationSecs # rightShaker endTime
                    if rightOFF>=stimTimeOFF[s]:
                        stopStim = rightOFF+add2stimDuration+secs2addStop
                    else:
                        stopStim = stimTimeOFF[s]+add2stimDuration+secs2addStop
                    rightONOFF = get_acclONOFF_fromAnalog(analogRightAccl, stimTimeON[s], stopStim, stimDurationSecs=rightDurationSecs, 
                                    thresholdHigh_std=analogAccl['thresholdHigh_std'], 
                                    thresholdLow_std = analogAccl['thresholdLow_std'], 
                                    baselineStartSecs=bslnStart, baselineStopSecs=bslnStop, 
                                    showPlot=analogAccl['showPlot'],
                                    showPlot_lowThreshold=analogAccl['showPlot_lowThreshold'], 
                                    showPlot_noDetected=analogAccl['showPlot_noDetected'],
                                    trialID = trialNEV['trialNum'],
                                    stimID = s+1
                                    )
                    
                    markerTime.append(rightONOFF[0])
                    markerTime.append(rightONOFF[1])
                else:
                    markerTime.append(defaultNoTime)
                    markerTime.append(defaultNoTime)

    #####################################################################################################################
    # Create AnalogTimes for MicroStimulation
    #####################################################################################################################
    if microStimulation is not None:

        # It will always get a dictionary per Stim. If YAML file does not have XIPP info, it will create one with default params. 
        stimIDs = [key for key in trialYAML.keys() if key.startswith('Stim ')]
        
        for s in range(nStimON):

            # Check if origYAML have MicroStim dictionary
            d = [i for i in range(len(trialYAML[stimIDs[s]])) if 'XIPP Stimulus' in trialYAML[stimIDs[s]][i]]

            if len(d)>0:
                
                microStim_ = expYAML.getMicroStimParams(
                    dictStim = trialYAML[stimIDs[s]], 
                    microStimChannel = microStimulation['global_channelID'], 
                    expStartTime = microStimulation['expStartTime']
                    )

                startTime_chans = microStim_['microStim']['ChannelStart_time']
                stopTime_chans = microStim_['microStim']['ChannelStop_time']

                if microStim_['valid']:

                    startICMS_epoch = stimTimeON[s]

                    if s==(nStimON-1):
                        stopICMS_epoch = outcomeTime
                    else:
                        stopICMS_epoch = stimTimeON[s+1]

                    # Get the pulse duration of the last Valid Train 
                    # DEFAULT 3 trains per Stim
                    pulse_Duration = []
                    for t in range(3):
                        if microStim_['microStim']['Duration'][t]>0:
                            pulse_Duration.append(microStim_['microStim']['InterphaseInterval'][t] + microStim_['microStim']['Phase1_Width'][t] + microStim_['microStim']['Phase2_Width'][t])

                    lastWF_duration = (pulse_Duration[-1]%(52/0.03))/1000000
                    
                    for ch in range(len(microStim_['microStim']['Channel'])):

                        channel_ = microStim_['microStim']['Channel'][ch]

                        if channel_>0:

                            ch_ts = [mS['timeStamps'] for mS in microStimulation['microStim_ns'] if mS['frontEnd_electrode_id']==channel_][0]
                            ch_ts_i = numpy.nonzero((ch_ts>=startICMS_epoch) & (ch_ts<=stopICMS_epoch))[0]

                            if ch_ts_i.size>0:

                                ch_stim = ch_ts[ch_ts_i]
                                next_pulse = max(ch_ts_i)+1

                                # Check if the next pulse-WF timeStamp is 
                                if next_pulse<ch_ts.size:
                                    if max(ch_stim)+lastWF_duration >= ch_ts[next_pulse]:
                                        print('Trial Num: {}\nTrial ID: {},\n{},\nChannel-{} ID: {}'.format(trialNEV['trialNum'], trialYAML['ID'], stimIDs[s], ch+1, channel_))
                                        print('Last microStim-WaveForm timeStamp: {}\nEndPulse timeStamp: {}\nNext microStim-WaveForm pulse startTime: {}'.format(
                                            max(ch_stim), max(ch_stim)+lastWF_duration, ch_ts[next_pulse]
                                        ))
                                        print('Last index for this microStim-WaveForm timeStamp: {}\nIndex of the next microStim-WaveForm timeStamp: {}\nTotal microStim-WaveForms: {}'.format(
                                            next_pulse-1, next_pulse, ch_ts.size
                                        ))
                                        raise Exception('Ending of microStim overlaps with the next microStim timeStamp')
                                
                                startTime_chans[ch] = min(ch_stim)
                                stopTime_chans[ch] = max(ch_stim)+lastWF_duration
                
                trialYAML[stimIDs[s]][d[0]].update({
                    'XIPP Stimulus Channel Times': {'startTime': startTime_chans, 'stopTime': stopTime_chans}
                    })
                
    #####################################################################################################################
    # Create tiemStamps for Fix ON-OFF & VisualCue at FixPosition
    #####################################################################################################################
    if analogFix is not None:
        
        if analogFix['exists']:
            fixTimes = get_trialFixONOFF_fromAnalog(analogFix['nsAnalog'], trialNEV, interTrialTime = analogFix['interTrialTime'], 
                        normThreshold = analogFix['normThreshold'], showPlot=analogFix['showPlot'])
        
            # APPEND new MARKERS
            #  1000 = fixON, 
            # 18000 = fixOFF,
            markerID.append(1000)
            markerID.append(18000)
            markerTime.append(fixTimes['fixON'])
            markerTime.append(fixTimes['fixOFF'])

    #####################################################################################################################
    # Create tiemStamps for VisualCues and ChoiceTargets
    #####################################################################################################################
    if analogVisualEvents is not None:

        if analogVisualEvents['exists']:

            visualEvents = get_trialVisualEventsON_fromAnalog(analogVisualEvents['nsAnalog'], trialNEV, 
                normThreshold=analogVisualEvents['normThreshold'],  
                photodiodeDuration=analogVisualEvents['photodiodeDuration'], 
                choiceTargetON = choiceTargetON,
                showPlot=analogVisualEvents['showPlot'],
                showWarningPlot=analogVisualEvents['showWarningPlot'])

            # APPEND new MARKERS
            # 15000 = visualCueON, 
            nCueON = len(visualEvents['cueON'])
            for n in range(nCueON):
                markerID.append(15000)
                markerTime.append(visualEvents['cueON'][n])

            # 5000 = choiceTargetsON
            nChoiceON = len(visualEvents['choiceON'])
            for n in range(nChoiceON):
                markerID.append(5000)
                markerTime.append(visualEvents['choiceON'][n])

    ##############################################################################################
    # Get the average Temperature of the trial
    ##############################################################################################
    trialTemps_dict = {}
    if analogTemp is not None:
        if analogTemp['exists']:
            temps = []
            for therm_cls in analogTemp['nsAnalog']:
                temps.append(get_averageTemp(therm_cls, ti = markerTime[markerID.index(1)], tf = max(markerTime)))
            trialTemps_dict.update({'trialTemp': temps})

    ##############################################################################################
    # sort Marker by Time
    nMarkers = len(markerID)
    markerIndex = numpy.argsort(numpy.array(markerTime))

    # Return the same directory but replace markerID, markerTime, outcomeTime and trialNum
    removeKey = ['markerID', 'markerTime', 'outcomeTime', 'trialNum']

    return {**{key:value for key, value in trialYAML.items() if key not in removeKey},
            **{
                'trialNum': trialNEV['trialNum'],
                'outcomeTime': outcomeTime , 
                'markerID':[markerID[markerIndex[i]] for i in range(nMarkers)], 
                'markerTime':[markerTime[markerIndex[i]] for i in range(nMarkers)],
               },
            **trialTemps_dict
            }

##################################################################################################################
# Convert a single YAMLformat trial into a dictionary with keys as columns that will match NWB table
def parseTrial2Row(dictYAML, trial, expStartTime, starTime_nextTrial):

    maxStimTypes = expYAML.getMaxStimTypes(dictYAML)
    fixMode = expYAML.getFixMode(dictYAML)
    fixTargetInfo = expYAML.getFixTargetInfo(dictYAML)
    visualCuesInfo = expYAML.getVisualCueInfo(dictYAML)
    responseMode = expYAML.getReponseMode(dictYAML)
    choiceTargetInfo = expYAML.getChoiceTargetInfo(dictYAML)
    shakerInfo = expYAML.getTactorInfo(dictYAML)
    microStimChannel = expYAML.getGlobal_microStim_channelID(dictYAML)

    return expYAML.parserTrial(trial, expStartTime, starTime_nextTrial, maxStimTypes, fixMode, fixTargetInfo, 
                      visualCuesInfo, responseMode, choiceTargetInfo, shakerInfo, microStimChannel)

##################################################################################################################
# Convert a list of YAMLformat trials into a dictionary with keys as columns that will match NWB table of trials
def parseTrials2Rows(dictYAML, trialList=None, expStartTime=0):

    if trialList is None:
        trialList = expYAML.getALLtrials(dictYAML)
        expStartTime = trialList[1]['markerTime'][trialList[1]['markerID'].index(1)]

    nTrials = len(trialList)

    # GET END TIME 
    durationExpSecs = expYAML.getStopTimeSecs(dictYAML) - expYAML.getStartTimeSecs(dictYAML) + 1
    stopTime = trialList[0]['markerTime'][trialList[0]['markerID'].index(1)] + durationExpSecs

    maxStimTypes = expYAML.getMaxStimTypes(dictYAML)
    fixMode = expYAML.getFixMode(dictYAML)
    fixTargetInfo = expYAML.getFixTargetInfo(dictYAML)
    visualCuesInfo = expYAML.getVisualCueInfo(dictYAML)
    responseMode = expYAML.getReponseMode(dictYAML)
    choiceTargetInfo = expYAML.getChoiceTargetInfo(dictYAML)
    shakerInfo = expYAML.getTactorInfo(dictYAML)
    microStimChannel = expYAML.getGlobal_microStim_channelID(dictYAML)

    trialRows = []
    for t in range(nTrials):
        print('....parsing trial {} out of {}.......'.format(t+1, nTrials))
        if t==(nTrials-1):
            starTime_nextTrial = stopTime
        else:
            starTime_nextTrial = trialList[t+1]['markerTime'][trialList[t+1]['markerID'].index(1)]

        trialRows.append(
            expYAML.parserTrial(trialList[t], expStartTime, starTime_nextTrial, maxStimTypes, fixMode, fixTargetInfo, 
                      visualCuesInfo, responseMode, choiceTargetInfo, shakerInfo, microStimChannel)
        )
    return trialRows

###################################################################################################################
# Merge NEV-markers & ANALOG-signals with YAML-trials into Trial-Dictionary 
# Dict- keys match the columns of the NWB table of trials
def getNWB_trials(dictYAML, nsFile=None, 
                analogAccl=None, 
                analogFix=None, 
                analogVisualEvents=None,
                analogTemp = None,
                verbose=False):

    # If nsFile is not provided, it will assume only YAML info will be extracted
    if nsFile is None:

        trials = expYAML.getALLtrials(dictYAML)
        expStartTime = trials[0]['markerTime'][trials[0]['markerID'].index(1)]

    else:

        expStartTime = 0

        trialsYAML = expYAML.getALLtrials(dictYAML)

        # Temporal? solution to passive stimulation
        responseMode = expYAML.getReponseMode(dictYAML)
        choiceTargetON = True
        # If noResponse, it is assume that choiceTargets were not shown on the screen
        # Marker5 (choiceTargetON) is still created, but in practice, no photodiode signal is sent
        # setting choiceTargetON = False, it will prevent to search for marker5 in the photodiode signal
        if responseMode=='noResponse':
            choiceTargetON = False


        nsTrialInfo = getTrialMarkers(nsFile)
        trialsNEV = nsTrialInfo['trials']

        # First Check same amount of trials:
        nYAML = len(trialsYAML)
        nNEV = len(trialsNEV)

        if nNEV != nYAML:
            ##########################################################################################################
            # TO DO:
            # nNEV < nYAML
            #   NEV TRIALS is lower than YAML trials, assume recording was stopped before the behavior. 
            #   In this case, get only the YAML trials that were recorded in NEV. But, ask user to confirm 
            #   that it is ok to do it.
            # 
            # nNEV > nYAML
            #   If NEV TRIALS is higher that YAML trials, There will be missing information about the trial type
            #   in the NEV trials that not have YAML counterpart: No visual IDs, no Tactile FR or AMP will be known.
            #   In this case, it will set the remaining NEV trials as abort trials. 
            #   There are two possible options for this case:
            #       1) The NEV and YAML files do not correspond to the same session.
            #       2) The behavior stop and start again in the same recording.
            #   It will check the difference between the first N common trials (markers = 1), it should
            #   be less that a second. If there is only one trial in common, use the dateTime 
            #   to check differences (assume the dateTime on both machines will be the same).
            #   Display the difference in time between files and ask the user if it wants to continue.
            ##########################################################################################################
            raise Exception('trials from NEV (n={}) must be the same as trials from YAML (n={})'.format(nNEV, nYAML))

        # Get Analog Accelerometer
        if analogAccl is not None:
            if analogAccl['exists']:
                analogAccl['shakerInfo'] = expYAML.getTactorInfo(dictYAML)
                analogAccl['nsAnalog'] = [
                    AnalogIOchannel(nsFile=nsFile, chanName='leftAccelerometer', 
                                                acclSensitivity=analogAccl['shakerInfo']['leftAcclSensitivity']),
                    AnalogIOchannel(nsFile=nsFile, chanName='rightAccelerometer', 
                                                acclSensitivity=analogAccl['shakerInfo']['rightAcclSensitivity'])
                ]
        
        # Get Analog MicrosStimulation
        microStimulation = {
            'microStim_ns': get_stimTimeStamps(nsFile), 
            'global_channelID': expYAML.getGlobal_microStim_channelID(dictYAML), 
            'expStartTime': expStartTime
        }

        # Used to detect FixON-OFF epoch (get some "baseline") and
        # Used to add enough time to detect RewardOFF FixON-OFF
        interTrialTime = dictYAML['Experimental Visual Settings']['Timing Settings']['Intertrial time']/1000

        # Duration of the Right photodiode pulse
        photodiodeDuration = dictYAML['Experimental Visual Settings']['Photodiode Target']['Right']['Duration (ms)']/1000

        # Get Analog Fixation Photodiode
        if analogFix is not None:
            if analogFix['exists']:
                analogFix['interTrialTime'] = interTrialTime
                analogFix['nsAnalog'] = AnalogIOchannel(nsFile=nsFile, chanName='fixON')

        # Get Analog Visual Events Photodiode
        if analogVisualEvents is not None:
            if analogVisualEvents['exists']:
                analogVisualEvents['photodiodeDuration'] = photodiodeDuration
                analogVisualEvents['nsAnalog'] = AnalogIOchannel(nsFile=nsFile, chanName='visualON')

        if analogTemp is not None:
            if analogTemp['exists']:
                analogTemp['nsAnalog'] = []
                for themID in analogTemp['thermistorIDs']:
                    analogTemp['nsAnalog'].append(AnalogIOchannel(nsFile=nsFile, chanName=themID))



        # Update & Merge NEV-trialInfo with YAML-trialInfo
        trials = [
            updateMarkerTime(trialYAML = trialsYAML[i], trialNEV = trialsNEV[i], 
                    analogAccl=analogAccl,
                    analogFix=analogFix,
                    analogVisualEvents=analogVisualEvents,
                    analogTemp = analogTemp,
                    microStimulation = microStimulation,
                    choiceTargetON = choiceTargetON,
                    verbose=verbose)
            for i in range(nNEV)
            ]    

    return parseTrials2Rows(dictYAML, trialList=trials, expStartTime=expStartTime)


def getNWB_rawElectrodes(dictYAML, nsFile, expDay_log=None, verbose=True):

    if expDay_log is None:
        electrodesDevicesYAML = expYAML.getElectrodeList(dictYAML)
        receptiveFieldsInfo = None
    else:
        electrodesDevicesYAML = expYAML.getElectrodeList_with_ReceptiveField(dictYAML, expDay_log, skipMissing_RF=False)
        receptiveFieldsInfo = electrodesDevicesYAML['receptiveFieldsInfo']

    electrodesGroupsYAML = electrodesDevicesYAML['electrodesGroups']
    electrodesYAML = electrodesDevicesYAML['electrodes']
    nChansYAML = len(electrodesYAML)

    electrodesNS = get_rawElectrodeInfo(nsFile)
    nChansNS = len(electrodesNS)

    if nChansYAML != nChansNS:
        if verbose:
            print('\nWARNING: YAML (nChans: {}) and raw NS electrodes (nChans: {}) must match.\
                \nElectrodes without YAML information will be set to default values'.format(
                nChansYAML, nChansNS
                ))

    # Merge Electrode Information from YAML and NS
    nsKeys2keep = ['entity_type', 'entity_index', 'id', 'port_id', 'frontEnd_id', 'frontEnd_electrode_id',
        'units', 'item_count', 'sample_rate', 'resolution', 'probe_info',
        'high_freq_corner', 'high_freq_order', 'high_filter_type',
        'low_freq_corner', 'low_freq_order', 'low_filter_type',
        ]
    
    ymlKeys2add = ['deviceName', 'group_id', 'location', 'rel_id',
        'ap', 'ml', 'dv', 'rel_ap', 'rel_ml', 'rel_dv',
        ]
    
    defaultYAML = ['Unknown', int(0), 'unknown', int(0),
        float(0), float(0), float(0), float(0), float(0), float(0)
        ]
    
    if receptiveFieldsInfo is not None:
        for k, v in receptiveFieldsInfo.items():
            ymlKeys2add.append(k)
            defaultYAML.append(v['defaultvalue'])
    
    electrodeYAML_id = [e['id'] for e in electrodesYAML]
    electrodesDict = []
    for c in range(nChansNS):
        # Find electrode in  YAML
        if electrodeYAML_id.count(electrodesNS[c]['id'])==1:
            yaml_info = electrodesYAML[electrodeYAML_id.index(electrodesNS[c]['id'])]
            yaml_data = {keyYAML: yaml_info[keyYAML] for keyYAML in ymlKeys2add}
        else:
            if verbose:
                print('\nWARNING:\
                    \nFor electrode {} [portID: {} & frontEnd: {}], Device & coordinates will be set to "Unknown" and "0"\
                    \nIt was not found on YAML electrodeList: {}.'.format(
                        electrodesNS[c]['id'], electrodesNS[c]['port_id'], electrodesNS[c]['frontEnd_id'],
                        electrodeYAML_id
                    ))
            yaml_data = {ymlKeys2add[i]:defaultYAML[i] for i in range(len(ymlKeys2add))}
        
        ns_data = {keyNS:electrodesNS[c][keyNS] for keyNS in nsKeys2keep}
        electrodesDict.append(
            {**ns_data, **yaml_data}
        )

    # CREATE ELECTRODE GROUPS DICTIONARY
    electrodeGroups = []
    
    for g in electrodesGroupsYAML:

        # get PORT & FRONTEND-SLOT IDs
        elecs = [e for e in range(len(electrodesDict)) if electrodesDict[e]['group_id']==g['group_id']]

        if len(elecs)==0:
            if g['group_id']!=0:
                if verbose:
                    print('WARINING: CHEK YAML electrodeINFO¡¡¡\nIt was not found any NS electrodes for electrodeGroup: {}\n\
                        from device: {}'.format(g['group_id'], g['deviceName']))
                electrodeGroups.append({
                    'group_id': g['group_id'],
                    'name': g['deviceName'] + '-NoRecorded',
                    'description': 'Raw signals from electrodes belonging to this Electrode Group were not found in the recording system',
                    'deviceName': g['deviceName'],
                    'manufacturer': 'Technical specs by codeName: ' + g['deviceName'],
                    'location': g['location'],
                    'position': g['position'],
                    'electrodes_id': [-1]
                    })
        else:

            elecsGroup_id = [electrodesDict[e]['id'] for e in elecs]

            if len(elecs)==1:
                electrodeType = 'Single Electrode. Electrode ID: {}. '.format(elecsGroup_id)
                electrodesName = '-{}'.format(elecsGroup_id[0])
                deviceDescription = 'Single Electrode'
            else:
                electrodeType = 'MultiElectrode Array. nChans: {}; electrodes ID: {}. '.format(len(elecs), elecsGroup_id)
                electrodesName ='-{}-{}'.format(min(elecsGroup_id), max(elecsGroup_id))
                deviceDescription = 'MultiElectrode Array. nChans: {}'.format(len(elecs))

            connectionInfo = 'It was connected to'
            portSlotsID = []
            nPortSlots = 0
            for i in range(len(elecs)):
                portSlot = [electrodesDict[elecs[i]]['port_id'], electrodesDict[elecs[i]]['frontEnd_id']]
                appendInfo = False
                if i==0:
                    appendInfo = True
                elif portSlot not in portSlotsID:
                    appendInfo = True
                    
                if appendInfo:
                    if nPortSlots>0:
                        connectionInfo += '; portID: {}; frontEnd-slot: {}'.format(portSlot[0], portSlot[1])
                    else:
                        connectionInfo += ' portID: {}; frontEnd-slot: {}'.format(portSlot[0], portSlot[1])
                    portSlotsID.append(portSlot)
                    nPortSlots += 1

            portsName = ''
            for p in range(len(portSlotsID)):
                if p>0:
                    portsName += '-'
                portsName += portSlotsID[p][0]
                portsName += str(portSlotsID[p][1])

            electrodeGroups.append({
                'group_id': g['group_id'],
                'name': portsName + electrodesName,
                'description': electrodeType + connectionInfo,
                'deviceName': g['deviceName'],
                'deviceDescription': deviceDescription,
                'manufacturer': 'Technical specs by codeName: ' + g['deviceName'],
                'location': g['location'],
                'position': g['position'],
                'electrodes_id': [electrodesDict[e]['id'] for e in elecs]
                })
    
    # CREATE DEVICE DICTIONARY
    deviceNames = []
    devices = []
    for g in range(len(electrodeGroups)):
        appendDevice = False
        if g==0:
            appendDevice = True
        else:
            if electrodeGroups[g]['deviceName'] not in deviceNames:
                appendDevice = True
        
        if appendDevice:
            devices.append({
                'name': electrodeGroups[g]['deviceName'],
                'description': electrodeGroups[g]['deviceDescription'],
                'manufacturer': electrodeGroups[g]['manufacturer']
            })
            deviceNames.append(electrodeGroups[g]['deviceName'])


    return electrodesDict, electrodeGroups, devices, receptiveFieldsInfo

###################################################################################################################

def create_rawElectrodeGroup_hdf5(electrodesDict, nsFile, groupID, tempFolderPath, verbose=True):
    
    group_index = [i for i in range(len(electrodesDict)) if electrodesDict[i]['group_id']==groupID]
    entities_index = [electrodesDict[i]['entity_index'] for i in group_index]
    
    # Confirm all entities have the same Filtering, Resolution, Convertion, Rate
    groupInfo = {
        'high_filter_type': [],
        'high_freq_order': [],
        'high_freq_corner': [],
        'low_filter_type': [],
        'low_freq_order': [],
        'low_freq_corner': [],
        'resolution': [],
        'units': [],
        'sample_rate': [],
        'item_count': []
        }
    
    for key in groupInfo.keys():
        list2check = []
        for i in group_index:
            e = electrodesDict[i]
            list2check.append(e[key])
        unique_key = numpy.unique(numpy.array(list2check))
        if unique_key.size>1:
            raise Exception('Electrode in Group: {} were colelcted with different settings: {} : {}\n{}'.format(
                groupID, key, unique_key, list2check
            ))
        else:
            groupInfo[key] = unique_key[0]
    
    groupInfo.update({'electrode_index': group_index})

    groupName = "electrodeGroup-{}".format(groupID)

    tempFilePath = temp_TimeSeries_hdf5(nsFile, entityIndexes=entities_index,  itemCount = int(groupInfo['item_count']),
                tempFolderPath=tempFolderPath, tempName=groupName, verbose=verbose)

    return tempFilePath, groupInfo