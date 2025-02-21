import os
import numpy
from ..yaulab_extras import pyns
import pytz
import datetime as dt
from tkinter.filedialog import askopenfilename

# TOOLS to extract Data from *.nev (markers) & *.ns5 (Raw 30K Analog & Neural Signals)
# Analog I/O follow YAULAB setup convention of the task : 
# visual cues + 2 shakers + footbar + eye tracking

version = '0.0.1'

########################################################################################
#  RIPPLE GENERAL INFO:
#
#  EntityTypes:
#
#  Unknown entity                   0
#  Event entity                     1
#  Analog entity                    2
#  Segment entity                   3
#  Neural event entity              4
#
#
#  Neural recording Front End channels are numbered from 1 to 5120. 
#
#  Front end ID is based on connection to the neural processor (numbered 1 - 16). 
#  Front ends connected to 
#  port A are numbered 1-4, channelNumbers = 1:128
#  port B are numbered 5-8, channelNumbers = 129:256
#  port C are numbered 9-12, channelNumbers = 257:384
#  port D are numbered 13-16. channelNumber = 385:512
#
#  Stimulation data channel numbers start at 5121 (512*10 + 1).
#  Analog I/O input channel numbers start at 10241 (512*20 + 1).
#
#  nev = 
#    Digital Events sampled at 10 Ks/sec
#    Waveform snippets: 1.7 ms (52 sample) data segment sampled at 30 kS/s
#    MicroStim waveforms: acquired in continuous 1.7 ms segments for each
#            stimulation pulse sampled at 30 kS/s (52 sample)
#
#  ns2 = 1 Ks/sec (LFP,  analog-I/O)
#  ns3 = 2 Ks/sec (EMG, ECoG)
#  ns5 = 30 Ks/sec (Raw data, analog-I/O)
#  ns6 = 7.5 Ks/sec (EMG)
#
# Formats to read bynary data
# #Hexadecimal representation 
# "{0:x}".format(value)
# #Octal representation
# "{0:o}".format(value)
# #Decimal representation
# "{0:d}".format(value)
# #Binary representation
# "{0:b}".format(value)

# Analog Channel IDs:
# AnalogIOchannels = {
#     'leftCommand': 1,
#     'leftAccelerometer': 2,
#     'rightCommand': 3,
#     'rightAccelerometer': 4,
#     'eyeHorizontal': 5,
#     'pupilDiameter': 6,
#     'leftFoot': 7,
#     'rightFoot': 8,
#     'rewardON': 9,
#     'fixON': 10,
#     'visualON': 11,
#     'leftProbeTemp': 12
#     }

ripple_port_Labels = ['A', 'B', 'C', 'D']
analog_samplingRate = 30000.00 # Get Analog inputs saved at 30K (.ns5)
raw_samplingRate = 30000.00 # Get Raw Neural Signals saved at 30K (.ns5)

thermistorLeftProbe_info = {
    'R0': 10000,  # ohm
    'coeffA': 0.001284850279,
    'coeffB': 0.0002076544735,
    'coeffC': 0.0000002004280704
}

def get_nsFile(filePathNEV=None, verbose=True):
    if filePathNEV is None:
        filePathNEV = askopenfilename(
        title = 'Select a NEV file to load',
        filetypes =[('nev Files', '*.nev')])
                
    if not os.path.isfile(filePathNEV):

        raise Exception("NEV-file-Path : {} doesn't exist ".format(filePathNEV))
    
    _, fileName = os.path.split(os.path.abspath(filePathNEV))

    if verbose:
        print('... loading NEV file {} and getting markers from DIGITAL inputs into python-dictionary..... \n'.format(fileName))
    
    return pyns.NSFile(filePathNEV)

#################################################################################
def getNS_StartDateTime(nsFile, TimeZone = None):

    if TimeZone is None:
        dateTimeNS = nsFile.get_time()
    else:
        dateTimeNS_orig = nsFile.get_time()
        dateTimeNS = pytz.timezone(TimeZone).localize(dt.datetime.combine(dateTimeNS_orig.date(), dateTimeNS_orig.time()))
        
    return dateTimeNS

#################################################################################
# Get DigitalInputs (MakerID & Time) and split them into Trials 
# Use Marker = 1 to start a each Trial
#################################################################################
def getTrialMarkers(nsFile):

    entityTypes = [entity.entity_type for entity in nsFile.entities]
    if entityTypes.count(1)==1:
        event_index = entityTypes.index(1)
    else:
        raise Exception('NEV file must content one event entity (current eventEntity = {})'.format(entityTypes.count(1)))

    eventEntity = nsFile.get_entity(event_index)
    if not 'digin' in eventEntity.label:
        raise Exception('Event entity must contain digitalInput values (current Event input = {})'.format(eventEntity.label))

    timeStamps = []
    markerID = []
    for i in range(eventEntity.item_count):
        evTuple = eventEntity.get_event_data(i)
        timeStamps.append(evTuple[0])
        markerID.append(evTuple[1][0])

    trialStart_i = [i for i in range(eventEntity.item_count) if markerID[i]==1]
    trialEnd_i = trialStart_i[1::]
    trialEnd_i.append(eventEntity.item_count)

    nTrials = len(trialStart_i)
    trialsDict = {'startTime': nsFile.get_time(), 'nTrials': nTrials, 'trials': []}
    for t in range(nTrials):
        # create Trial Dictionary
        # print(trialStart_i[t], trialEnd_i[t])
        trialsDict['trials'].append({
            'trialNum': t+1,
            'markerID': markerID[trialStart_i[t]:trialEnd_i[t]], 
            'markerTime': timeStamps[trialStart_i[t]:trialEnd_i[t]]
        }
        )

    return trialsDict

def appendTrialMarkers(trialsDict, markerID, markerTime):

    trialNEV = trialsDict['trials']
    nMarkers = len(markerID)

    trialsDict_new = {'startTime':trialsDict['startTime'], 'nTrials': trialsDict['nTrials'], 'trials': []}

    for t in range(trialsDict['nTrials']):

        markerID_Trial = []
        markerTime_Trial = []

        markerID_Trial = trialNEV[t]['markerID']
        markerTime_Trial = trialNEV[t]['markerTime']

        # find Markers within this trial
        startTrial = markerTime_Trial[markerID_Trial.index(1)] 
        if t==(trialsDict['nTrials']-1):
            stopTrial = max(markerTime_Trial)
            
        else:
            stopTrial = trialNEV[t+1]['markerTime'][trialNEV[t+1]['markerID'].index(1)]
        
        for m in range(nMarkers):
            if markerTime[m]>=startTrial and markerTime[m]<= stopTrial:
                markerID_Trial.append(markerID[m])
                markerTime_Trial.append(markerTime[m])

        # Sort by time
        newIndex = numpy.argsort(numpy.array(markerTime_Trial))
        nNew= len(markerTime_Trial)
        trialsDict_new['trials'].append({
            'trialNum': trialNEV[t]['trialNum'],
            'markerID': [markerID_Trial[newIndex[i]] for i in range(nNew)],
            'markerTime': [markerTime_Trial[newIndex[i]] for i in range(nNew)]
        })

    return trialsDict_new


def get_rawElectrodeInfo(nsFile):

    electrodeList = []

    for i in range(nsFile.get_entity_count()):

        entity = nsFile.get_entity(i)
        
        if entity.entity_type==2 and entity.electrode_id<5121 and entity.item_count>0:

            elecInfo = entity.get_analog_info()
            
            # Confirm sampling rate (30K, .ns5):  
            if float(elecInfo.sample_rate)==raw_samplingRate:
  
                port_index = int(numpy.ceil(entity.electrode_id/128))-1
                rel_index = int(numpy.ceil((entity.electrode_id-(128*port_index))/32))-1

                electrodeList.append({
                    "entity_index": i,
                    "entity_type": entity.entity_type,
                    "electrode_id": entity.electrode_id,
                    "id": int(entity.electrode_id),
                    "port_id": ripple_port_Labels[port_index],
                    "frontEnd_id": int(rel_index+1),
                    "frontEnd_electrode_id": int(entity.electrode_id-(128*port_index)-(32*rel_index)),
                    "label_id": entity.label.decode('utf-8'),
                    "item_count": entity.item_count,
                    "sample_rate": float(elecInfo.sample_rate),
                    "min_val": elecInfo.min_val,
                    "max_val": elecInfo.max_val,
                    "units": elecInfo.units.decode('utf-8'),
                    "resolution": elecInfo.resolution,
                    "location_x": elecInfo.location_x,
                    "location_y": elecInfo.location_y,
                    "location_z": elecInfo.location_z,
                    "location_user": elecInfo.location_user,
                    "high_freq_corner": elecInfo.high_freq_corner,
                    "high_freq_order": elecInfo.high_freq_order,
                    "high_filter_type": elecInfo.high_filter_type,
                    "low_freq_corner": elecInfo.low_freq_corner,
                    "low_freq_order": elecInfo.low_freq_order,
                    "low_filter_type": elecInfo.low_filter_type,
                    "probe_info": elecInfo.probe_info,
                })

    return electrodeList

def get_stimElectrodeInfo(nsFile):

    electrodeList = []

    for i in range(nsFile.get_entity_count()):

        entity = nsFile.get_entity(i)
        
        if entity.entity_type==3 and entity.electrode_id>5120 and entity.electrode_id<10241 and entity.item_count>0:

            electrode_id = entity.electrode_id-5120
            segmentInfo = entity.get_segment_info()
            segmentSourceInfo = entity.get_seg_source_info()
            port_index = int(numpy.ceil(electrode_id/128))-1
            rel_index = int(numpy.ceil((electrode_id-(128*port_index))/32))-1

            electrodeList.append({
                "entity_index": i,
                "entity_type": entity.entity_type,
                "electrode_id": entity.electrode_id,
                "id": int(electrode_id),
                "port_id": ripple_port_Labels[port_index],
                "frontEnd_id": int(rel_index+1),
                "frontEnd_electrode_id": int(electrode_id-(128*port_index)-(32*rel_index)),
                "label_id": entity.label.decode('utf-8'),
                "item_count": entity.item_count,
                "source_count": segmentInfo.source_count,
                "sample_rate": float(segmentInfo.sample_rate),
                "min_sample_count": segmentInfo.min_sample_count,
                "max_sample_count ": segmentInfo.max_sample_count,
                "units": segmentInfo.units,
                "min_val": segmentSourceInfo.min_val,
                "max_val": segmentSourceInfo.max_val,
                "resolution": segmentSourceInfo.resolution,
                "subsample_shift": segmentSourceInfo.subsample_shift,
                "location_x": segmentSourceInfo.location_x,
                "location_y": segmentSourceInfo.location_y,
                "location_z": segmentSourceInfo.location_z,
                "location_user": segmentSourceInfo.location_user,
                "high_freq_corner": segmentSourceInfo.high_freq_corner,
                "high_freq_order": segmentSourceInfo.high_freq_order,
                "high_filter_type": segmentSourceInfo.high_filter_type,
                "low_freq_corner": segmentSourceInfo.low_freq_corner,
                "low_freq_order": segmentSourceInfo.low_freq_order,
                "low_filter_type": segmentSourceInfo.low_filter_type,
                "probe_info": segmentSourceInfo.probe_info,
            })
    return electrodeList


def get_stimTimeStamps(nsFile):

    timeStampsList = [] 

    for i in range(nsFile.get_entity_count()):

        entity = nsFile.get_entity(i)
        
        if entity.entity_type==3 and entity.electrode_id>5120 and entity.electrode_id<10241 and entity.item_count>0:

            electrode_id = entity.electrode_id-5120
            segmentInfo = entity.get_segment_info()
            segmentSourceInfo = entity.get_seg_source_info()
            port_index = int(numpy.ceil(electrode_id/128))-1
            rel_index = int(numpy.ceil((electrode_id-(128*port_index))/32))-1
            
            
            timeStamps = []
            for idx in range(entity.item_count):
                wvData = entity.get_segment_data(index=idx)
                timeStamps.append(wvData[0])

            timeStampsList.append({
                "entity_index": i,
                "entity_type": entity.entity_type,
                "electrode_id": entity.electrode_id,
                "id": int(electrode_id),
                "port_id": ripple_port_Labels[port_index],
                "frontEnd_id": int(rel_index+1),
                "frontEnd_electrode_id": int(electrode_id-(128*port_index)-(32*rel_index)),
                "label_id": entity.label.decode('utf-8'),
                "item_count": entity.item_count,
                "source_count": segmentInfo.source_count,
                "sample_rate": float(segmentInfo.sample_rate),
                "probe_info": segmentSourceInfo.probe_info,
                "timeStamps": numpy.array(timeStamps),
            })

    return timeStampsList

def mV_to_degC(signal_mV, r0_ohms, coeffA, coeffB, coeffC):

    RT = r0_ohms *(5000 / (signal_mV) - 1)
    T = numpy.log(RT)
    T = 1 / (coeffA + (coeffB + (coeffC * T * T )) * T ) # Temp Kelvin   Steinhart–Hart equation

    C = T - 273.15 # Convert Kelvin to Celcius

    return C
#################################################################
# General AnalogInput Info from YAULAB setup
#################################################################
class AnalogIOchannelID:
    
    # Analog I/O input channel numbers start at 10241 (512*20 + 1).
    analogChanStart = 512*20
    unknown =            [0, 'Analog channel is not recognized']
    leftCommand =        [1, 'Command signal (mV) to drive LEFT shaker']
    leftAccelerometer =  [2, 'Accelerometer signal (microns/secs^2) from the LEFT shaker']
    rightCommand =       [3, 'Command signal (mV) to drive RIGHT shaker']
    rightAccelerometer = [4, 'Accelerometer signal (microns/secs^2) from the RIGHT shaker']
    eyeHorizontal =      [5, "Horizontal eye position in degrees (relative to the eye's center)"]
    pupilDiameter =      [6, 'Pupil diameter in pixels']
    leftFoot =           [7, "5V signal wich indicates that subject's LEFT-foot is not holding the footbar"]
    rightFoot =          [8, "5V signal wich indicates that subject's RIGHT-foot is not holding the footbar"]
    rewardON =           [9, "5V signal wich indicates that reward's valve is open"]
    fixON =              [10, "LEFT Photodiode: 5V signal wich indicates light intensity on the fixation center"]
    visualON =           [11, "RIGHT Photodiode: 5V signal indicating every time something is drawn on the screen (aka, visual Event)"]
    leftProbeTEMP =      [12, "Temperature (centigrades) of the Thermistor from the left Probe: 5V signal indicating the temprature was converted into degrees"]

    # samplingRate (30K, .ns5)
    samplingRate = analog_samplingRate

    # 1 Volt = 64 pixels
    mVolt2pixel = 0.064

    # 1 volt = 10 degree
    mVolt2deg = 0.01 

    @classmethod
    def get_chan_name(cls, chanNum):
        if chanNum==cls.leftCommand[0]:
            return 'leftCommand'
        elif chanNum==cls.leftAccelerometer[0]:
            return 'leftAccelerometer'
        elif chanNum==cls.rightCommand[0]:
            return 'rightCommand'
        elif chanNum==cls.rightAccelerometer[0]:
            return 'rightAccelerometer'
        elif chanNum==cls.eyeHorizontal[0]:
            return 'eyeHorizontal'
        elif chanNum==cls.pupilDiameter[0]:
            return 'pupilDiameter'
        elif chanNum==cls.leftFoot[0]:
            return 'leftFoot'
        elif chanNum==cls.rightFoot[0]:
            return 'rightFoot'
        elif chanNum==cls.rewardON[0]:
            return 'rewardON'
        elif chanNum==cls.fixON[0]:
            return 'fixON'
        elif chanNum==cls.visualON[0]:
            return 'visualON'
        elif chanNum==cls.leftProbeTEMP[0]:
            return 'leftProbeTEMP'
        else:
            return 'unknown'
    
    @classmethod
    def get_chan_num(cls, chanName):
        if chanName=='leftCommand':
            return cls.leftCommand[0]
        elif chanName=='leftAccelerometer':
            return cls.leftAccelerometer[0]
        elif chanName=='rightCommand':
            return cls.rightCommand[0]
        elif chanName=='rightAccelerometer':
            return cls.rightAccelerometer[0]
        elif chanName=='eyeHorizontal':
            return cls.eyeHorizontal[0]
        elif chanName=='pupilDiameter':
            return cls.pupilDiameter[0]
        elif chanName=='leftFoot':
            return cls.leftFoot[0]
        elif chanName=='rightFoot':
            return cls.rightFoot[0]
        elif chanName=='rewardON':
            return cls.rewardON[0]
        elif chanName=='fixON':
            return cls.fixON[0]
        elif chanName=='visualON':
            return cls.visualON[0]
        elif chanName=='leftProbeTEMP':
            return cls.leftProbeTEMP[0]
        else:
            return cls.unknown[0]
    
    @classmethod
    def get_chanNumDescription(cls, chanNum):
        if chanNum==cls.leftCommand[0]:
            return cls.leftCommand[1]
        elif chanNum==cls.leftAccelerometer[0]:
            return cls.leftAccelerometer[1]
        elif chanNum==cls.rightCommand[0]:
            return cls.rightCommand[1]
        elif chanNum==cls.rightAccelerometer[0]:
            return cls.rightAccelerometer[1]
        elif chanNum==cls.eyeHorizontal[0]:
            return cls.eyeHorizontal[1]
        elif chanNum==cls.pupilDiameter[0]:
            return cls.pupilDiameter[1]
        elif chanNum==cls.leftFoot[0]:
            return cls.leftFoot[1]
        elif chanNum==cls.rightFoot[0]:
            return cls.rightFoot[1]
        elif chanNum==cls.rewardON[0]:
            return cls.rewardON[1]
        elif chanNum==cls.fixON[0]:
            return cls.fixON[1]
        elif chanNum==cls.visualON[0]:
            return cls.visualON[1]
        elif chanNum==cls.leftProbeTEMP[0]:
            return cls.leftProbeTEMP[1]
        else:
            return cls.unknown[1]
        
    @classmethod
    def get_chanNameDescription(cls, chanName):
        if chanName=='leftCommand':
            return cls.leftCommand[1]
        elif chanName=='leftAccelerometer':
            return cls.leftAccelerometer[1]
        elif chanName=='rightCommand':
            return cls.rightCommand[1]
        elif chanName=='rightAccelerometer':
            return cls.rightAccelerometer[1]
        elif chanName=='eyeHorizontal':
            return cls.eyeHorizontal[1]
        elif chanName=='pupilDiameter':
            return cls.pupilDiameter[1]
        elif chanName=='leftFoot':
            return cls.leftFoot[1]
        elif chanName=='rightFoot':
            return cls.rightFoot[1]
        elif chanName=='rewardON':
            return cls.rewardON[1]
        elif chanName=='fixON':
            return cls.fixON[1]
        elif chanName=='visualON':
            return cls.visualON[1]
        elif chanName=='leftProbeTEMP':
            return cls.leftProbeTEMP[1]
        else:
            return cls.unknown[1]

#######################################################################
# MAIN CLASS to get Info & data from a given Analog I/O Channel
#######################################################################
class AnalogIOchannel:

    def __init__(self, nsFile=None, chanNum=None, chanName=None, acclSensitivity=None):

        if nsFile is None:
            raise Exception('NEV file object must be specified')
        
        if chanName is None and chanNum is None:
            raise Exception('ChanName or ChanNum must be specified')
        
        if chanNum is None:
            chanNum = AnalogIOchannelID.get_chan_num(chanName)

        if chanName is None:
            chanName = AnalogIOchannelID.get_chan_name(chanNum)
        
        self.chanNum = chanNum
        self.chanName = chanName
        self.rippleChanNum = self.chanNum + AnalogIOchannelID.analogChanStart
        self.acclSensitivity = acclSensitivity

        ############################################################
        # SET description of the channel
        self.description = AnalogIOchannelID.get_chanNumDescription(self.chanNum)
        
        ############################################################
        # SET convertion Factor and units 
        # Eye degree
        if self.chanNum==5: 
            self.convertion_factor = AnalogIOchannelID.mVolt2deg
            self.units = 'degrees'
        # Pupil diameter
        elif self.chanNum==6:
            self.convertion_factor = AnalogIOchannelID.mVolt2pixel
            self.units = 'pixels'
        # Accelerometers
        elif self.chanNum==2 or self.chanNum==4:
            self.convertion_factor = 9.80665*(10**6)/self.acclSensitivity
            self.units = 'microns/sec^2'
        elif self.chanName=='leftProbeTEMP':
            self.convertion_factor = 1.0
            self.units = 'centigrades'
        else:
            self.convertion_factor = 1.0
            self.units = 'mV'
        
        ############################################################
        # Set Entity
        self.entity_index = None
        self.entity = None
        for i in range(nsFile.get_entity_count()):
            entity = nsFile.get_entity(i)
            if entity.entity_type==2 and entity.electrode_id==self.rippleChanNum:  
                # Confirm sampling rate (30K, .ns5):  
                fs = float(entity.get_analog_info().sample_rate)
                if fs==AnalogIOchannelID.samplingRate:
                    self.entity_index = int(i)
                    self.entity = entity

        if self.entity_index is None or self.entity is None:
            print('AnalogEntity sampled at 30 KHz (.ns5) for Electrode {} ({}, Ripple ID: {}) did not exist'.format(
                self.chanNum, self.chanName, self.rippleChanNum
                ))
            self.entity_exist = False
        else:
            self.entity_exist = True

        self.info = None
        self.info_set = False

    def get_convertion_factor(self):
        return self.convertion_factor
    
    def get_units(self):
        return self.units
    
    def get_description(self):        
        return self.description
    
    def set_info(self):

        if not self.info_set and self.entity_exist:
                
            analogInfo = self.entity.get_analog_info()
            samplingRate = float(analogInfo.sample_rate)
            analogUnits = analogInfo.units.decode('utf-8')
            if analogUnits=='mV':
                convertion2mV = 1.0000
            elif analogUnits=='V':
                convertion2mV = 1000.0000
            elif analogUnits=='uV':
                convertion2mV = 1.0000/1000.0000

            self.info = {
                'index' : self.entity_index,
                'chanName' : self.chanName,
                'description' : self.description + ". Signal was recorded on analogInput: {} ({})".format(self.chanNum, analogInfo.probe_info),
                'units': self.units,
                'convertionFactor' : self.convertion_factor,
                'item_count' : self.entity.item_count,
                'samplingRate' : samplingRate,
                'analogConvertion2mV' : convertion2mV,
            }

            self.info_set = True
        
    def get_info(self):
        if not self.info_set:
            self.set_info()
        return self.info
    
    def get_data(self, start_index=0, index_count=None):

        if self.entity_exist:

            if start_index<0: 
                raise Exception('start_index {} must be positive and lower than {}'.format(
                    start_index, self.info['item_count']))
            
            if not self.info_set:
                self.set_info()
            
            if index_count is None:
                index_count = self.info['item_count']

            if index_count<0: 
                raise Exception('index_count {} must be positive'.format(index_count))
            elif (start_index+index_count) > self.info['item_count']:
                    raise Exception('index_count {} is out of range\
                        \nFor a "start_index"={} a valid count must be between lower than {})'.format(
                        index_count, start_index, self.info['item_count']-index_count))
            
            if self.chanName=='leftProbeTEMP':
                return mV_to_degC(
                    signal_mV = self.entity.get_analog_data(start_index=int(start_index), index_count=int(index_count))*self.info['analogConvertion2mV'],
                    r0_ohms = thermistorLeftProbe_info['R0'],
                    coeffA = thermistorLeftProbe_info['coeffA'],
                    coeffB = thermistorLeftProbe_info['coeffB'],
                    coeffC = thermistorLeftProbe_info['coeffC']
                    )
            else:
                return self.entity.get_analog_data(start_index=int(start_index), index_count=int(index_count))*self.info['analogConvertion2mV']
        
        else:
            return []
        
    def get_timeIndex(self, timeSecs):

        if self.entity_exist:
            if not self.info_set:
                self.set_info()

            return [numpy.floor(t*self.info['samplingRate']).astype(int) for t in timeSecs]
        else:
            return []
    
    def get_indexTime(self, index):
        
        if self.entity_exist:
            if not self.info_set:
                self.set_info()
        
            return [float(i/self.info['samplingRate']) for i in index]
        else:
            return []


#######################################################################
# MAIN CLASS to get Info & data from a given MicroStim Channel
#######################################################################
class SegmentStimChannel:

    def __init__(self, nsFile, electrode_id=None):

        if nsFile is None:
            raise Exception('NEV file object must be specified')
        
        if electrode_id is None:
            raise Exception('Electrode_id must be specified (numbered according to Ripple ID: 1 to 5120)')
        
        self.entity_index = None
        self.entity = None
        for i in range(nsFile.get_entity_count()):
            entity = nsFile.get_entity(i)
            if entity.entity_type==3 and entity.electrode_id==(electrode_id + 5120):        
                self.entity_index = int(i)
                self.entity = entity
        
        if self.entity_index is None or self.entity is None:
            print('SegmentEntity for Electrode {} did not exist')
            self.entity_exist = False
        else:
            self.entity_exist = True
        
        self.info = None
        self.info_set = False
    
    def set_info(self):

        if not self.info_set and self.entity_exist:

            electrode_id = self.entity.electrode_id-5120
            segmentInfo = self.entity.get_segment_info()
            segmentSourceInfo = self.entity.get_seg_source_info()
            port_index = int(numpy.ceil(electrode_id/128))-1
            rel_index = int(numpy.ceil((electrode_id-(128*port_index))/32))-1

            self.info = {
                    "entity_index": self.entity_index,
                    "entity_type": self.entity.entity_type,
                    "electrode_id": self.entity.electrode_id,
                    "id": int(electrode_id),
                    "port_id": ripple_port_Labels[port_index],
                    "frontEnd_id": int(rel_index+1),
                    "frontEnd_electrode_id": int(electrode_id-(128*port_index)-(32*rel_index)),
                    "label_id": self.entity.label.decode('utf-8'),
                    "item_count": self.entity.item_count,
                    "source_count": segmentInfo.source_count,
                    "sample_rate": float(segmentInfo.sample_rate),
                    "min_sample_count": segmentInfo.min_sample_count,
                    "max_sample_count": segmentInfo.max_sample_count,
                    "units": segmentInfo.units,
                    "min_val": segmentSourceInfo.min_val,
                    "max_val": segmentSourceInfo.max_val,
                    "resolution": segmentSourceInfo.resolution,
                    "subsample_shift": segmentSourceInfo.subsample_shift,
                    "location_x": segmentSourceInfo.location_x,
                    "location_y": segmentSourceInfo.location_y,
                    "location_z": segmentSourceInfo.location_z,
                    "location_user": segmentSourceInfo.location_user,
                    "high_freq_corner": segmentSourceInfo.high_freq_corner,
                    "high_freq_order": segmentSourceInfo.high_freq_order,
                    "high_filter_type": segmentSourceInfo.high_filter_type,
                    "low_freq_corner": segmentSourceInfo.low_freq_corner,
                    "low_freq_order": segmentSourceInfo.low_freq_order,
                    "low_filter_type": segmentSourceInfo.low_filter_type,
                    "probe_info": segmentSourceInfo.probe_info
                }
            self.info_set = True

    def get_info(self):
        if not self.info_set:
            self.set_info()
        return self.info
    
    def get_data(self, index=None, verbose=True):
        
        if self.entity_exist:

            if not self.info_set:
                self.set_info()

            if index is None:
                index = [i for i in range(self.info['item_count'])]

            if verbose:
                print('Extracting {} out of {} MicroStim waveforms from electrode: {} ({})'.format(
                    len(index), self.info['item_count'], self.info['id'], self.info['label_id']
                    ))
            
            
            timeStamps = []
            waveForms= []
            unitID= []
            
            for i in index:
                # Check if index has valid value(s)
                if i<0: 
                    raise Exception('index {} values must be positive'.format(i))
                elif i >=self.info['item_count']:
                    raise Exception('index {} is out of range (valid index must be between [{} - {}])'.format(i, 0, self.info['item_count']-1))
                wvData = self.entity.get_segment_data(index=i)
                timeStamps.append(wvData[0])
                waveForms.append(wvData[1])
                unitID.append(wvData[2])
            
            nSamples = len(index)

            data = {
                'timeStamps': numpy.array(timeStamps),
                'waveForms': numpy.stack(waveForms, axis=0).reshape(nSamples, 1, self.info['max_sample_count']),
                'unitID': numpy.array(unitID)
                }
        else:
            data = {
                'timeStamps': None,
                'waveForms': None,
                'unitID': None
                }
        
        return data