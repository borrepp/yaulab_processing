import os, struct
import uuid
import datetime as dt
import dateutil
import numpy
import pandas
import pytz
import yaml
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog

from ..yaulab_extras import monkeysDict, labInfo, electrodeDevices


# TOOLS to extract & Session, Subject, ElectrodeInfo, ShakerInfo & Behavioral Data from *.YAML files 
# file format follow YAULAB convention of the task : 
# visual cues + 2 shakers + footbar + eye tracking

version = '0.0.1'

#################################################################################
# Read YAML into Dictionary
#################################################################################}
def yaml2dict(filePathYAML = None, verbose=True):

    if filePathYAML is None:
        filePathYAML = askopenfilename(
        title = 'Select a YAML file to load',
        filetypes =[('yaml Files', '*.yaml')])
                
    if not os.path.isfile(filePathYAML):

        raise Exception("YAML-file-Path : {} doesn't exist ".format(filePathYAML))
    
    _, fileName = os.path.split(os.path.abspath(filePathYAML))

    if verbose:
        print('\n... loading YAML file {} into python-dictionary..... \n'.format(fileName))

    # Extract YAML as nested Dictionaries
    with open(filePathYAML, 'r') as stream:
        dictYAML = yaml.safe_load(stream)

    return dictYAML


#################################################################################
# Read Binary EYE-startTime
#################################################################################
def getEyeStartTime(filePathEYE = None, verbose=True):

    """
    The following is the format of the eye Binary-output.

    float —> 32bit -->  4 bytes
    char —>  8bit  -->  1 byte

    struct = {
          char [12]   // time stamp     // 96 bit -> 12 bytes
          float [1]   // eye_pos_x      // 32 bit -> 4 bytes
          float [1]   // eye_pos_y      // 32 bit -> 4 bytes
          float [1]   // pupil_diameter // 32 bit -> 4 bytes
     }

    """

    if filePathEYE is None:
        filePathEYE = askopenfilename(
        title = 'Select an EYE file to load',
        filetypes =[('eye Files', '*.eye')])
                
    if not os.path.isfile(filePathEYE):

        raise Exception("EYE-file-Path : {} doesn't exist ".format(filePathEYE))
    
    _, fileName = os.path.split(os.path.abspath(filePathEYE))

    if verbose:
        print('\n... reading startTime EYE file {}..... \n'.format(fileName))

    fBin = open(filePathEYE, 'rb')
    datBin = fBin.read()
    fBin.close()

    nStream = len(datBin)

    if nStream==0:
        startTime = None
    else:
        if nStream % 24 != 0:
            raise Exception("Binary data doesn't match proper dimensions : last sample has {} bytes out of 24".format(nStream % 24))

        startTime = struct.unpack('@12s', datBin[0:12])

        startTime = dt.datetime.strptime(startTime[0].decode(), "%H:%M:%S.%f").time()

    return startTime
               
#################################################################################
# Read Binary EYE-data 
#################################################################################
def getEyeData(filePathEYE = None, offsetSecs = 0, verbose=True):

    """
    The following is the format of the eye Binary-output.

    float —> 32bit -->  4 bytes
    char —>  8bit  -->  1 byte

    struct = {
          char [12]   // time stamp     // 96 bit -> 12 bytes
          float [1]   // eye_pos_x      // 32 bit -> 4 bytes
          float [1]   // eye_pos_y      // 32 bit -> 4 bytes
          float [1]   // pupil_diameter // 32 bit -> 4 bytes
     }

    """

    if filePathEYE is None:
        filePathEYE = askopenfilename(
        title = 'Select an EYE file to load',
        filetypes =[('eye Files', '*.eye')])
                
    if not os.path.isfile(filePathEYE):

        raise Exception("EYE-file-Path : {} doesn't exist ".format(filePathEYE))
    
    _, fileName = os.path.split(os.path.abspath(filePathEYE))

    if verbose:
        print('\n... loading EYE file {} into python-dictionary..... \n'.format(fileName))

    fBin = open(filePathEYE, 'rb')
    datBin = fBin.read()
    fBin.close()

    nStream = len(datBin)

    eyeData = {'time': [], 'x': [], 'y': [], 'pupil': []}

    if nStream>0:

        if nStream % 24 != 0:
            raise Exception("Binary data doesn't match proper dimensions : last sample has {} bytes out of 24".format(nStream % 24))

        startTime = struct.unpack('@12s', datBin[0:12])

        startTime = dt.datetime.strptime(startTime[0].decode(), "%H:%M:%S.%f")

        for t in range(0, nStream, 24):

            tDat = struct.unpack('@12s3f', datBin[t:t+24])

            currentTime = dt.datetime.strptime(tDat[0].decode(), "%H:%M:%S.%f")
            lapsedTime = (currentTime-startTime).total_seconds()

            eyeData['time'].append(lapsedTime + offsetSecs)
            eyeData['x'].append(tDat[1])
            eyeData['y'].append(tDat[2])
            eyeData['pupil'].append(tDat[3])

            del tDat, currentTime, lapsedTime
    
    return eyeData

#################################################################
# Some Variables used to get default parameters
#################################################################

dictOutcomes = {
    10 : 'correct', 
    11 : 'fail to fix CORRECT target',
    12 : 'fail to fix ANY target',
    13 : 'no saccade', 
    14 : 'eye-abort',
    140 : 'foot-abort',
    141 : 'eyeFoot-abort',
    16 : 'experiment stopped'
}

dictOutcomes.update({i : 'incorrect choiceTarget {}'.format(i-30) for i in range(31, 61)})
dictOutcomes.update({i : 'fail to fix incorrect choiceTarget {}'.format(i-60) for i in range(61, 91)})

trialKeys2remove = ['OutCome', 'Marker Signal Sequence']

ripple_port_Labels = ['A', 'B', 'C', 'D']

default_RF = {'BodySide': 'notInspected', 'BodyPart': 'notInspected', 'Modality': 'notInspected'}

description_RF = {
    'BodySide': 'description of the body side (left/right) that responded to touch (or possible vision) mapped during the experimental session. Null values can be "notDrivable", "notInspected" or "none"', 
    'BodyPart': 'description of the body part(s) responsive to touch. When receptive fields cover larger areas, this field can include mulitple regions of the body. Example names: head_backhead, head_neck, face_eyebrow, face_cheek,face_ upperlip, face_lowerlip, wrist, palm_distalPads, whorls, [FINGERS: f1D(M)(P): examples: f2M, f3MDP]', 
    'Modality': 'description of the Sensory (Sub)modality: proprioception, cutaneous, visual.'
    }

null_BodySide_RF = ['notdrivable', 'notinspected', 'notconnected', 'none']


################################################
# General Params & Functions for all YAMLs
###############################################
class expYAML:

    ###########################################################
    # If exists, GET DEFAULT MONKEY INFO
    ###########################################################
    @classmethod
    def getMonkeyInfo(cls, dictYAML):

        # Check if Monkey ID exist
        monkeyNames = list(monkeysDict.keys())
        monkeyInfo = None

        for name in monkeyNames:
            if name.lower() in dictYAML['Subject Info']['Subject ID'].lower():
                monkeyInfo = monkeysDict[name]

        return monkeyInfo

    ###########################################################
    # Subject Info (nwb-like format)
    ###########################################################
    @classmethod
    def getSubjectInfo(cls, dictYAML, subject_id=None, description=None, sex=None,
                    species=None, date_of_birth=None, age=None):

        # Check if monkey exist:
        monkeyInfo = cls.getMonkeyInfo(dictYAML)

        subject_id = dictYAML['Subject Info']['Subject ID']
        description = dictYAML['Subject Info']['Subject ID']

        # Extract REQUIRED info from YAML, If doesn't exist it will prompt a dialog window to enter the info
        if monkeyInfo is None:

            sbj = dictYAML['Subject Info']

            # Sex is Required
            if sex is None:
                sex = sbj['Sex']
                if sex is None:
                    root = tk.Tk()
                    root.withdraw()
                    sex = simpledialog.askstring(
                        title = 'Sex not found',
                        prompt = 'What is the sex of the subject \n'\
                        ' “F” (female), “M” (male), “U” (unknown), or “O” (other) is recommended'
                        )
                    sex = sex[0].upper()

            # Species is Required
            if species is None:
                species = sbj['Species']
                if species is None:
                    root = tk.Tk()
                    root.withdraw()
                    species = simpledialog.askstring(
                        title = 'Species not found',
                        prompt = 'Species must be defined. \n'\
                            'The formal latin binomal name is recommended, e.g., “Mus musculus" '
                        )

            if date_of_birth is None:
                date_of_birth = sbj['Date of Birth']
                if date_of_birth == 'n/a':
                    date_of_birth = None
                if isinstance(date_of_birth, str):
                    # Try default dateFormat (Month/Day/Year)
                    date_of_birth = dateutil.parser(date_of_birth)
            
            if age is None:      
                age = sbj['Age']
                if age is not None:
                    if age.lower() == 'n/a':
                        age = None

            # DOB or AGE is Required
            if age is None and date_of_birth is None:
                root = tk.Tk()
                root.withdraw()                
                ageDOB = simpledialog.askstring(
                    title = 'Age/DOB not found',
                    prompt = 'The age of the subject or Date Of Birth is needed. \n'\
                        'Which input will be provided (type: AGE or DOB)',
                    )
                    
                if 'age' in ageDOB.lower():
                    root = tk.Tk()
                    root.withdraw()
                    age = simpledialog.askstring(
                        title = 'AGE',
                        prompt = 'Age : The ISO 8601 Duration format is recommended, e.g., “P90D” for 90 days old. \n'\
                            'A timedelta will automatically be converted to The ISO 8601 Duration format.',
                        )
                else:
                    root = tk.Tk()
                    root.withdraw()
                    date_of_birth = simpledialog.askstring(
                        title = 'Date of Birth',
                        prompt = "Enter numeric datetime of the date of birth ('MM-DD-YYYY').",
                        )
                    date_of_birth = dt.datetime.strptime(date_of_birth, "%m-%d-%Y")
        else:

            print("Monkey ID was recognized .. getting default Params")

            subject_id = monkeyInfo['Subject ID']
            description = description + ' - Tatoo : ' + monkeyInfo['Tatoo']
            species = monkeyInfo['Species']
            sex = monkeyInfo['Sex']
            date_of_birth = monkeyInfo['Date of Birth']
            age = None
        
        # Return SubjectInfo as a Dictionary matching the fields required by NWB-format
        return {
            'age': age,
            'description': description,
            'sex' : sex[0].upper(),
            'species' : species,
            'subject_id' : subject_id,
            'date_of_birth': date_of_birth,
            }
    
    ###########################################################
    # Session Info (nwb-like format)
    ###########################################################
    @classmethod
    def getSessionInfo(cls, dictYAML, session_start_time, session_id = None, session_description = None, identifier=None):

        if session_id is None:
            # use YAML namefile 
            session_id = 'session_'+ dictYAML['Experiment Started'].replace(':', '')

        if session_description is None:
            if bool(dictYAML['Subject Info']['Comments']):
                session_description = dictYAML['Subject Info']['Comments'] # required
        
        #Training with Headfixed to keep eye at fix-target and categorize both hands
        if identifier is None:
            identifier=str(uuid.uuid4())  # required

        return {
            'session_id' : session_id, 
            'session_description' : session_description, 
            'identifier' : identifier, 
            'session_start_time' : session_start_time
            }
    
    ###########################################################
    # Experiment YAML-Metadata (nwb-like format)
    ###########################################################
    @classmethod
    def getExperimentInfo(cls, dictYAML, 
                        lab=None, institution=None, protocol=None, experiment_description=None,
                        surgery = None, experimenter = None, stimulus_notes = None, notes = None,
                        keywords = None, related_publications = None
                        ):

        # Check if monkey exist:
        monkeyInfo = cls.getMonkeyInfo(dictYAML)

        if monkeyInfo is None:
            if lab is None:
                lab = dictYAML['Subject Info']['Lab'] 

            if institution is None:
                institution = dictYAML['Subject Info']['Institution'] 

            if protocol is None:
                protocol = dictYAML['Subject Info']['Protocol']

            if surgery is None:
                surgery = dictYAML['Subject Info']['Surgery']

        else:
            lab = monkeyInfo['Lab']
            institution = monkeyInfo['Institution']
            protocol = monkeyInfo['Protocol']
            surgery = monkeyInfo['Surgery']

        if experiment_description is None:
            experiment_description = dictYAML['Subject Info']['Comments']
        
        return {
            'lab': lab,
            'institution': institution,
            'protocol': protocol,
            'experiment_description': experiment_description,
            'surgery': surgery,
            'experimenter': experimenter, 
            'stimulus_notes': stimulus_notes, 
            'notes': notes,
            'keywords': keywords, 
            'related_publications': related_publications
            }
    
    #################################################################################
    # Get the number of Electrode devices: 
    # Single electrodes (i.e. FHC)
    # Single Probes (i.e. Plexon probes)
    # Get default Ripple information per channel based on the YAML Electrode fields:
    # PortID, FrontID, frontEnd_electrode_id
    #################################################################################
    @classmethod
    def getElectrodeList(cls, dictYAML):
        
        electrodeDicts = [value for key, value in dictYAML['Subject Info'].items() if key.startswith('Electrode')]
        electrodesKeys = [key for key, value in dictYAML['Subject Info'].items() if key.startswith('Electrode')]
        nElectrodesGroups = len(electrodeDicts)
        electrodeResults = {
            'electrodesGroups': [{
                'deviceName': 'Unknown-ElectrodeProbe',
                'position': [float(0), float(0), float(0)],
                'location': 'Unknown',
                'nChans': int(0),
                'group_id': int(0),
                'port_id': 'Unknown'
            }], 
            'electrodes': []
            }
        
        for eD in range(nElectrodesGroups):

            coordinates = [float(c) for c in str(electrodeDicts[eD]['Coordinates(AP, ML, DV)']).split(' ')]
            electrodeDF = electrodeDevices[electrodeDicts[eD]['Name']]
            nChans = len(electrodeDF)

            # Each electrode group indicates a device. Each devices has a field called "Front end".
            # It consists of 4 number: [frontEnd_id, MicroStim, startCountChannel, stopCountChannel]
            # Each device can be connected up to 4 frontEnds (1 to 4)
            # Each frontEnd has a True/False value whether is a microStimulation frontEnd or not
            # Electrodes within each frontEnd are numbered starting from 1 up the number of electrodes connected to that fronEnd
            if ripple_port_Labels.count(electrodeDicts[eD]['Port ID'])!=1:
                raise Exception('{} has an incorrect Port ID: {}.\nDeviceName: {}, valid PortIDs: {})'.format(
                    electrodesKeys[eD], electrodeDicts[eD]['Port ID'], electrodeDicts[eD]['Name'], ripple_port_Labels
                ))
            port_index = ripple_port_Labels.index(electrodeDicts[eD]['Port ID'])
            
            id = []
            port_id = []
            frontEnd_id = []
            frontEnd_electrode_id = []
            microStimChan = []

            nFrontEnds = len(electrodeDicts[eD]['Front End'])
            for f in range(nFrontEnds):
                frontEnd_info = [int(c) for c in str(electrodeDicts[eD]['Front End'][f]).split(' ')]
                for e in range(frontEnd_info[2], frontEnd_info[3]+1):
                    id.append(int(e + (128*port_index) + (32*(frontEnd_info[0]-1))))
                    port_id.append(electrodeDicts[eD]['Port ID']) 
                    frontEnd_id.append(int(frontEnd_info[0]))
                    frontEnd_electrode_id.append(int(e))
                    microStimChan.append(frontEnd_info[1]==1)

            nChansYAML = len(id)
            

            if nChansYAML!=nChans:
                raise Exception(
                    '\nYAML chanNum (n={}) must match Device-ElectrodeGroup map (device: {}, expected # of Chans={})\n'.format(
                    nChansYAML, electrodeDicts[eD]['Name'], nChans) +
                    'YAML Group: {}\nYAML-electrode channels: {}'.format(
                    electrodesKeys[eD], frontEnd_electrode_id
                ))
            
            electrodeResults['electrodesGroups'].append({
                'deviceName': electrodeDicts[eD]['Name'],
                'position': coordinates,
                'location': electrodeDicts[eD]['Brain Area'],
                'nChans': nChans,
                'group_id': int(eD+1), 
                'port_id': electrodeDicts[eD]['Port ID']
                })
            
            for c in range(nChans):
                electrodeResults['electrodes'].append({
                    'deviceName': electrodeDicts[eD]['Name'],
                    'group_id': int(eD+1),
                    'location': electrodeDicts[eD]['Brain Area'],
                    'id': id[c], 
                    'rel_id': int(electrodeDF.iloc[c]['chanNumProbe']),
                    'ap': coordinates[0]-float(electrodeDF.iloc[c]['ap']),
                    'ml': coordinates[1]-float(electrodeDF.iloc[c]['ml']),
                    'dv': coordinates[2]-float(electrodeDF.iloc[c]['dv']),
                    'rel_ap': float(electrodeDF.iloc[c]['ap']),
                    'rel_ml': float(electrodeDF.iloc[c]['ml']), 
                    'rel_dv': float(electrodeDF.iloc[c]['dv']),            
                    'port_id': port_id[c],
                    'frontEnd_id': frontEnd_id[c],
                    'frontEnd_electrode_id': frontEnd_electrode_id[c],
                    'microStimChan': microStimChan[c]
                })

        return electrodeResults
    
    ################################################################################################
    # Update the electrode Information with the Receptive Field mapping recorded in the expDAY.xlsx 
    # Note: expDAY.xlsx must be loaded previously as pandas.dataframe (input = expDay_log)
    ###########################################################
    @classmethod
    def getElectrodeList_with_ReceptiveField(cls, dictYAML, expDay_log, skipMissing_RF=False):

        ##########################################################
        # Extract Electrode Information from YAML
        electrodeInfo_YAML = cls.getElectrodeList(dictYAML)

        ##################################################################################################
        # Extract Electrode Information from EXCEL
        # Each Electrode is saved as independent sheets. Multiple Sheets are exported as dictionaries:
        log_electrodeIDs = [{'group_id':int(key[9:]), 'dataFrame': val} for key, val in expDay_log.items() if key.startswith('Electrode')]

        # Check electrode Groups match between YAML and expDAY.xlsx
        set_electrodeID_yaml = {group['group_id'] for group in electrodeInfo_YAML['electrodesGroups'] if group['group_id']!=0}
        set_electrodeID_log = {group['group_id'] for group in log_electrodeIDs}

        if set_electrodeID_yaml.difference(set_electrodeID_log):

            raise Exception('Electrodes: {} in YAML are not in log_expDAY (excel has Electrodes: {})'.format(set_electrodeID_yaml.difference(set_electrodeID_log), set_electrodeID_log))
        
        elif set_electrodeID_log.difference(set_electrodeID_yaml):
            # It is possible that a Ripple headstage was connected and recording but no electrode was attached. 
            # If that occurs, NWB tools will name it as "Unknown-ElectrodeProbe"
            # To handle RF for this electrode, the expDAY.xlsx must contain that electrode sheet with:
            #       "BodySide_1" = 'notConnected' 
            # for ALL the channels of the headstage that was recorded.

            electrodes_to_check = set_electrodeID_log.difference(set_electrodeID_yaml)
            exception_electrodes = []

            for e in electrodes_to_check:

                msgs_prefix = 'To validate EXCEL "Electrode{}" as not connected'.format(e)

                error_msgs = []
                error_flag = False

                electrode_index = [i for i in range(len(log_electrodeIDs)) if log_electrodeIDs[i]['group_id']==e]
                if len(electrode_index)!=1:
                    raise Exception('Electrode Num : {} mismatch in the excel file'.format(e))
                
                pd_electrode = log_electrodeIDs[electrode_index[0]]['dataFrame']

                # Search for default  "BodySide_1" = 'notConnected' in all electrodes:
                if pd_electrode['BodySide_1'].isnull().sum().any():
                    error_flag = True
                    error_msgs.append('{} must NOT contain NaN for "BodySide_1" values\nNumber of NaN values present: {}'.format(msgs_prefix, pd_electrode['BodySide_1'].isnull().sum()))
                    
                body_side_ = pd_electrode['BodySide_1'].unique()

                if body_side_.size !=1:
                    error_flag = True
                    error_msgs.append('{} "BodySide_1" must be have a single unique value acroos all electrodes\nUnique "BodySide_1" = {} (size={})'.format(msgs_prefix, body_side_, body_side_.size))

                if type(body_side_[0]) is not str:
                    error_flag = True
                    error_msgs.append('{} "BodySide_1" must be string\nUnique "BodySide_1" = {} (type = {})'.format(msgs_prefix, body_side_[0], type(body_side_[0])))

                if 'no' not in body_side_[0].lower() and 'connected' not in body_side_[0].lower():
                    error_flag = True
                    error_msgs.append('{} "BodySide_1" must be explicit about it\nUnique "BodySide_1" = {}'.format(msgs_prefix, body_side_[0]))

                exception_electrodes.append({'electrodeID': e, 'error_flag':error_flag, 'error_msgs': error_msgs})
            
            for exception_electrode in exception_electrodes:
                if exception_electrode['error_flag']:
                    for error_msg in exception_electrode['error_msgs']:
                        print(error_msg)
                    if not skipMissing_RF:
                        raise Exception('Electrode: {} in log_expDAY is not in YAML (yaml Electrodes: {})'.format(exception_electrode['electrodeID'], set_electrodeID_yaml))

        ##################################################################################################
        # Get the max number of receptive fields detected per channel and across all the Electrode
        valid_RF = []
        valid_RF_per_electrode = []
        # It will extract only the electrodes in common¡¡¡
        valid_electrodeIDs = list(set_electrodeID_log.intersection(set_electrodeID_yaml))

        for e in valid_electrodeIDs:

            electrode_index = [i for i in range(len(log_electrodeIDs)) if log_electrodeIDs[i]['group_id']==e]
            if len(electrode_index)!=1:
                raise Exception('Electrode Num : {} mismatch in the excel file'.format(e))
                
            eg = log_electrodeIDs[electrode_index[0]]
            
            eg.update({'nChans': [group['nChans'] for group in electrodeInfo_YAML['electrodesGroups'] if group['group_id']==eg['group_id']][0]})

            pd_electrode = eg['dataFrame']

            # Check there is no column Names missing:
            if any([True for col_name in pd_electrode.columns if col_name.startswith('Unnamed')]):
                raise Exception('Electrode-{} has data with incomplete Receptive Fields column names¡¡'.format(eg['group_id']))

            nPossible_bodyParts = len([col_name for col_name in pd_electrode.columns if col_name.startswith('BodyPart_')])

            valid_RF_eg = []

            for n in range(nPossible_bodyParts):
                # Check if all the aspect of the RF have at least one channel valid (not NaN's)
                parts_valids = []
                for c in default_RF.keys():
                    # Check all RF-parts names exist:
                    if c +'_{}'.format(int(n+1)) not in pd_electrode.columns:
                        raise Exception('Electrode-{} detetected "BodyPart_{}", but the corresponding {}_{} was not found'.format(eg['group_id'], n+1, c, n+1))
                    else:
                        parts_valids.append(not pd_electrode[c +'_{}'.format(int(n+1))].isnull().all())

                if all(parts_valids):
                    valid_RF.append(n+1)
                    valid_RF_eg.append([c +'_{}'.format(int(n+1)) for c in default_RF.keys()])
            
            valid_RF_per_electrode.append(valid_RF_eg)
        
        # Create a dictionary with the total number of ReceptiveRield names
        RF_index = list(numpy.unique(numpy.array(valid_RF)))
        RF_names_description = {'receptiveField_n': {'description': 'Number of Receptive Fields found while mapping during the experimental session', 'defaultvalue': float(0)}}
        for n in RF_index:
            for c in default_RF.keys():
                RF_names_description.update({
                    c +'_{}'.format(int(n)): {'description': 'For the Receptive Field number = {}, this is the {}'.format(n, description_RF[c]), 'defaultvalue': default_RF[c]}
                    })

        electrodeInfo_YAML.update({'receptiveFieldsInfo': RF_names_description})

        #############################################################################################################################################
        # Update 'electrodes' dictionary on a channel by channel basis.
        #############################################################################################################################################
        for e in electrodeInfo_YAML['electrodes']:

            # Update with DEFAULT Electrode Info to match "Max" body parts
            for RF_name, dictVal in RF_names_description.items():
                e.update({RF_name: dictVal['defaultvalue']})
            
            # find the excel-Sheet (dataFrame) corresponding to this electrode's "electrodeGroup"
            log_eGroup = [eg for eg in log_electrodeIDs if eg['group_id']==e['group_id']][0]

            # find the ROW matching the electrode's "ChanNum"
            # In case is a single channel, search for the closest "Depth"
            if log_eGroup['nChans']==1:
                # If "Depth" exist, search for the closest "dv" coordinate
                if 'Depth' in log_eGroup['dataFrame']:
                    e_row = log_eGroup['dataFrame'].iloc[abs(log_eGroup['dataFrame']['Depth']-e['dv']).argmin()]
                elif 'ChanNum' in log_eGroup['dataFrame']:
                    e_index = log_eGroup['dataFrame']['ChanNum']==e['rel_id']
                    if e_index.any():
                        e_row = log_eGroup['dataFrame'].iloc[e_index.argmax()]
                    else:
                        raise Exception('Electrode-{} should contain "Depth" or "ChanNum" column to identify the receptive field'.format(e['group_id']))
                else:
                    raise Exception('Electrode-{} should contain "Depth" or "ChanNum" column to identify the receptive field'.format(e['group_id']))
            else:
                if 'ChanNum' not in log_eGroup['dataFrame']:
                    raise Exception('Electrode-{} should contain "ChanNum" column to identify the receptive field'.format(e['group_id']))
                else:
                    e_index = log_eGroup['dataFrame']['ChanNum']==e['rel_id']
                    if e_index.any():
                        e_row = log_eGroup['dataFrame'].iloc[e_index.argmax()]
                    else:
                        raise Exception('from Electrode-{}, chanNum={} was not found in the Receptive Field map (check excel file)'.format(e['group_id'], e['rel_id']))

            # Check that the valid RF for this electrode are completed properly:
            e_RF_attributeIDs = valid_RF_per_electrode[log_eGroup['group_id']-1]
            nRF = 0
            for e_RF_attribute in e_RF_attributeIDs:

                bodySide = e_row.loc[[a for a in e_RF_attribute if 'BodySide_' in a][0]]

                # Update RF only if BodySide was NOT "notDrivable" or "notInspected" or "none" or "NAN" 
                if type(bodySide) is not str:
                    if numpy.isnan(bodySide):
                        bodySide = 'notinspected'
                    else:
                        raise Exception('"BodySide" type ({}) NOT valid from Electrode-{}, channel = {}. It must be "str" or NaN¡¡'.format(type(bodySide), e['group_id'], e['rel_id']))

                if bodySide.lower() not in null_BodySide_RF:
                    # Check that RF attributes (BodyPart, Modality) are not NaN
                    for a in e_RF_attribute:
                        if pandas.isnull(e_row.loc[a]):
                            if skipMissing_RF:
                                print('Electrode-{}, channel = {}, has receptive field attribute "{}" missing ({})¡¡'.format(e['group_id'], e['rel_id'], a, e_row.loc[a]))
                            else:
                                raise Exception('Electrode-{}, channel = {}, has receptive field attribute "{}" missing ({})¡¡'.format(e['group_id'], e['rel_id'], a, e_row.loc[a]))
                        else:
                            # Ensure is String
                            e.update({a: str(e_row.loc[a])})
                    nRF += 1
                else:
                    # Copy any original note (type=string) made when BodySide was null_BodySide:
                    for a in e_RF_attribute:
                        if type(e_row.loc[a]) is str:
                            e.update({a: e_row.loc[a]})

            e['receptiveField_n'] = float(nRF)
    
        return electrodeInfo_YAML

    ###########################################################
    # Get the List of Electrodes enabled for microstimulation
    ###########################################################
    @classmethod
    def getMicroStimElectrodeList(cls, dictYAML):

        electrodeDicts = [value for key, value in dictYAML['Subject Info'].items() if key.startswith('Electrode')]
        electrodesKeys = [key for key, value in dictYAML['Subject Info'].items() if key.startswith('Electrode')]
        nElectrodesGroups = len(electrodeDicts)

        electrodeList = []
        nFrontEndsMicroStim = 0
        frontEndsMicroStim = []

        for eD in range(nElectrodesGroups):

            electrodeDF = electrodeDevices[electrodeDicts[eD]['Name']]
            nChans = len(electrodeDF)

            if ripple_port_Labels.count(electrodeDicts[eD]['Port ID'])!=1:
                raise Exception('{} has an incorrect Port ID: {}.\nDeviceName: {}, valid PortIDs: {})'.format(
                    electrodesKeys[eD], electrodeDicts[eD]['Port ID'], electrodeDicts[eD]['Name'], ripple_port_Labels
                ))
            
            port_index = ripple_port_Labels.index(electrodeDicts[eD]['Port ID'])
                    
            nChansYAML = 0
            electrodeID_Name = []

            nFrontEnds = len(electrodeDicts[eD]['Front End'])
            
            for f in range(nFrontEnds):

                frontEnd_info = [int(c) for c in str(electrodeDicts[eD]['Front End'][f]).split(' ')]

                if frontEnd_info[1]==1:
                    nFrontEndsMicroStim += 1
                    frontEndsMicroStim.append('{} : {}{}'.format(electrodesKeys[eD], electrodeDicts[eD]['Port ID'], frontEnd_info[0]))

                for e in range(frontEnd_info[2], frontEnd_info[3]+1):

                    port_id = electrodeDicts[eD]['Port ID']
                    frontEnd_id = int(frontEnd_info[0])
                    frontEnd_electrode_id = int(e)

                    if frontEnd_info[1]==1:

                        electrodeList.append({
                            'id': int(e + (128*port_index) + (32*(frontEnd_id-1))),
                            'port_id': port_id,
                            'frontEnd_id': frontEnd_id,
                            'frontEnd_electrode_id': frontEnd_electrode_id,
                        })

                    nChansYAML += 1
                    electrodeID_Name.append('{}{}-{}'.format(port_id, frontEnd_id, frontEnd_electrode_id))
                    

            if nChansYAML!=nChans:
                raise Exception(
                    '\nYAML chanNum (n={}) must match Device-ElectrodeGroup map (device: {}, expected # of Chans={})\n'.format(
                    nChansYAML, electrodeDicts[eD]['Name'], nChans) +
                    'YAML Group: {}\nYAML-electrode channels: {}'.format(
                    electrodesKeys[eD], electrodeID_Name
                ))
            
        if nFrontEndsMicroStim>1:
            print('WARNING¡¡ {} FrontEnd(s) were enabled for microstimulation\nThe code has NOT been updated to handle more than one\nYAML-FrontEnd(s) enabled:\n'.format(
                nFrontEndsMicroStim
                ))
            for i in range(nFrontEndsMicroStim):
                print(frontEndsMicroStim[i])
            print('\n')
        
        # Check for unique MicroStim Channels in the YAML (first Rep : trial by Trial)
        channelID_trials = []
        for i in range(cls.getNumTrialsRep(dictYAML, repID=1)):

            trial = cls.getTrial_by_Index(dictYAML, repID=1, trialIndex=i)

            # For-loop of Stim List of Params
            stimList = [value for key, value in trial.items() if key.startswith('Stim ')]

            for dictStim in stimList:
                # Get MicroStim (Check Amplitude, Duration, Frequency)
                microStimParams = cls.getMicroStimParams(dictStim, dictYAML=dictYAML, verbose=False)
                if microStimParams['valid']:
                    for chanID in microStimParams['microStim']['Channel']:
                        if chanID>0 and channelID_trials.count(chanID)==0:
                            channelID_trials.append(chanID)

        # Check that there is only one microStimChannel in the list 
        electrodesMicroStim = []
        for  chanID in channelID_trials:
            chanIDInfo = [electDict for electDict in electrodeList if electDict['frontEnd_electrode_id']==chanID]
            if len(chanIDInfo)==1:
                electrodesMicroStim.append(chanIDInfo[0])
            else:
                print('ElectrodeID = {}, was found in {}-frontEnd(s) : '.format(chanID, len(chanIDInfo)))
                for chan_i in chanIDInfo:
                    print(chan_i)
                raise Exception('It should NOT be more than one microStimulation electrode with the same ID')
                
        
        return electrodesMicroStim               

    ########################################################### 
    # fix Mode option : eye, eyeNoPostChoice, foot, mouse
    ###########################################################
    @classmethod
    def getFixMode(cls, dictYAML):

        fixMode = []

        dictExp = dictYAML['Experimental Visual Settings']
        keys = list(dictExp.keys())

        searchEyeFix = True
        if keys.count('Eye Fixation Without PostChoice')==1:
            if dictExp['Eye Fixation Without PostChoice']==1:                
                fixMode.append('eyeNoPostchoice')
                searchEyeFix = False
                    

        if keys.count('Foot Fixation Mode')==1:
            if dictExp['Foot Fixation Mode']==1:
                fixMode.append('foot')

        if keys.count('Eye Fixation Mode')==1:
            if dictExp['Eye Fixation Mode']==1 and searchEyeFix:
                fixMode.append('eye')

        # Assume that when nothing was chosen, mouse was used
        if len(fixMode)==0:
            fixMode.append('mouse')

        return '-'.join(fixMode)
    
    ###########################################################
    # response Mode option : eye, foot, noResponse, mouse 
    ###########################################################
    @classmethod
    def getReponseMode(cls, dictYAML):

        dictExp = dictYAML['Experimental Visual Settings']
        keys = list(dictExp.keys())

        footResp  = False
        if keys.count('Foot Response Mode')==1:
            if dictExp['Foot Response Mode']==1 and keys.count('Target On')==1:
                footResp = True
                footID = 'foot'
            elif dictExp['Foot Response Mode']==1 and keys.count('Target Off')==1:
                footResp = True
                footID = 'noResponse'
        
        eyeResp = False
        if keys.count('Eye Response Mode')==1:
            if dictExp['Eye Response Mode']==1:
                eyeResp = True
        
        if footResp and eyeResp:
            raise Exception('Eye and Foot response is not a valid Response Mode, check the code for updates or YAML file for errors')
        elif footResp and not eyeResp:
            responseMode = footID
        elif not footResp and eyeResp:
            responseMode = 'eye'
        else:
            fixMode = cls.getFixMode(dictYAML)
            if fixMode=='mouse':
                responseMode = 'mouse'
            elif fixMode=='eyeNoPostchoice-foot' or fixMode=='eyeNoPostchoice':
                responseMode = 'foot'
            else:
                responseMode = fixMode

        return responseMode
    
    ################################################################################################
    # Get the max number of Tactile, MicroStim and Visual stim & ChoiceTargetsShown per Trial 
    # It will use the first Repetition
    ################################################################################################
    @classmethod
    def getMaxStimTypes(cls, dictYAML):

        nStim = 0
        nTactileStim = 0
        nMicroStim = 0
        nVisualStim = 0
        nChoiceTargetsShown = 0

        for i in range(cls.getNumTrialsRep(dictYAML, repID=1)):

            trial = cls.getTrial_by_Index(dictYAML, repID=1, trialIndex=i)

            choiceTargetsShown = len(str(trial['Showing Target IDs']).split(' '))

            stim = 0
            tactStim = 0
            microStim = 0
            visualStim = 0

            stimList = [value for key, value in trial.items() if key.startswith('Stim ')]

            # For-loop of Stim List of Params
            for dictStim in stimList:

                stim += 1

                # Check TACTILE Amplitude and Duration are higher than 0
                leftExists = dictStim[0]['Duration']>0 and dictStim[0]['Amp']>0 and dictStim[0]['Freq']>0
                rightExists = dictStim[1]['Duration']>0 and dictStim[1]['Amp']>0 and dictStim[1]['Freq']>0
                if leftExists or rightExists:
                    tactStim += 1
                
                # Get MicroStim (Check Amplitude, Duration, Frequency)
                microStimParams = cls.getMicroStimParams(dictStim, dictYAML=dictYAML, verbose=False)
                if microStimParams['valid']:
                    microStim += 1

                # Chek Visual CUES (it assumes is the last dictionary)
                cueIDs = [int(c) for c in str(dictStim[-1]['ID']).split(' ')]
                if max(cueIDs)>0:
                    visualStim += len([i for i in cueIDs if i >0])

            if stim>nStim:
                nStim = stim
            if tactStim>nTactileStim:
                nTactileStim = tactStim
            if microStim>nMicroStim:
                nMicroStim = microStim
            if visualStim>nVisualStim:
                nVisualStim = visualStim
            if choiceTargetsShown>nChoiceTargetsShown:
                nChoiceTargetsShown = choiceTargetsShown
        
        if cls.getReponseMode(dictYAML) == 'noResponse':
            nChoiceTargetsShown = 0

        return {'nStim': nStim,'nTactileStim': nTactileStim, 'nMicroStim': nMicroStim, 
                'nVisualStim': nVisualStim, 'nChoiceTargetsShown': nChoiceTargetsShown}
    
    #################################################################################
    # GET CHANNEL ID for MICROSTIMULATION 
    #################################################################################
    @classmethod
    def getGlobal_microStim_channelID(cls, dictYAML):
        # check if XIPP Stimulus channel exists
        if 'XIPP Stimulus Channel' in dictYAML:
            channels_global = [int(i) for i in str(dictYAML['XIPP Stimulus Channel']).split(' ')]
        else:
            # Search for Channel info on a trial by trial basis
            channels_global = None
        
        return channels_global

    #################################################################################
    # Get Tactile parameters from YAML stim-dictionary
    #################################################################################
    @classmethod
    def getTactileStimParams(cls, dictStim):
        
        # Search for LEFT & RIGHT parameters from Stim-dictionary
        tactStimParams = {'leftValid': False, 'rightValid': False}

        # Default indices:
        # dictStim[0] = LEFT
        # dictStim[1] = RIGHT

        ###################
        # Add Info
        shakerID = ['left', 'right']
        
        for s in range(0, 2):

            # Check if exist Stim based on amplitude, freq, Duration
            amplitudeCheck = dictStim[s]['Amp']>0
            durationCheck = dictStim[s]['Duration']>0
            freqCheck = dictStim[s]['Freq']>0

            tactStimParams[shakerID[s]+'Valid'] = all([amplitudeCheck, durationCheck, freqCheck]) 

            if tactStimParams[shakerID[s]+'Valid']:
                tactStimParams.update( {shakerID[s]: dictStim[s]})
            # if tactile nor valid reset values to default TactParams
            else:
                tactStimParams.update( {shakerID[s]: labInfo['StimDefaults']['Tactile']})

        return tactStimParams

    #################################################################################
    # GET RIPPLE Micro-Stim Paramters from YAML stim-dictionary
    #################################################################################
    @classmethod
    def getMicroStimParams(cls, dictStim, dictYAML=None, microStimChannel=None, expStartTime=None, verbose=True):

        checkChannels_in_Stim = True # Default search for XIPP Stimulus channel in Stim dictionary

        # Check if dictYAML is an input and search for XIPP Stimulus Channel
        if dictYAML is not None:
            channels_global = cls.getGlobal_microStim_channelID(dictYAML)
            if channels_global is not None:
                checkChannels_in_Stim  = False
            if expStartTime is None:
                if verbose:
                    print('WARNING¡ YAML-startTime will be use as a default for "XIPP Stimulus Channel Times"')
                expStartTime = cls.getStartTimeSecs(dictYAML)
        
        # Check if stimChannel is already an input (i.e., read it from dictYAML in advance)
        # WARINING: This input will supersede/overwrite dictYAML['XIPP Stimulus Channel']
        if microStimChannel is not None:
            channels_global = microStimChannel
            checkChannels_in_Stim  = False

        # Search for MicroStimulation parameters from Stim-dictionary
        microStimParams = {'valid': False}

        existsMicroStim = False

        for dictParams in dictStim:
            
            if 'XIPP Stimulus' in dictParams:

                existsMicroStim = True

                if checkChannels_in_Stim:
                    if 'XIPP Stimulus Channel' in dictParams:
                        channelIDs = [int(i) for i in str(dictParams['XIPP Stimulus Channel']).split(' ')]
                    else:
                        raise Exception('[XIPP Stimulus Channel] info was not found ¡¡\n{}'.format(
                            dictParams
                        ))
                else:
                    channelIDs = channels_global
                
                if 'XIPP Stimulus Channel Times' in dictParams:
                    start_chans = dictParams['XIPP Stimulus Channel Times']['startTime']
                    stop_chans = dictParams['XIPP Stimulus Channel Times']['stopTime']
                else:
                    if expStartTime is None:
                        raise Exception('No MicroStim TimeStamps were found')
                                        
                    noMicroStim_time = expStartTime + float(labInfo['StimDefaults']['NoTime'])                        

                    start_chans = [noMicroStim_time, noMicroStim_time, noMicroStim_time, noMicroStim_time]
                    stop_chans = [noMicroStim_time, noMicroStim_time, noMicroStim_time, noMicroStim_time]


                microStimParams['microStim'] = {
                    'Stimulus': dictParams['XIPP Stimulus'],
                    'StartTime': dictParams['XIPP Stim Start Time'],
                    'Channel': channelIDs,
                    'ChannelStart_time': start_chans,
                    'ChannelStop_time': stop_chans,
                    'ReturnChannel': int(dictParams['XIPP Return Channel']),
                    'Duration': [float(i) for i in str(dictParams['XIPP Duration (Sec)']).split(' ')],
                    'Frequency': [float(i) for i in str(dictParams['XIPP Frequency (HZ)']).split(' ')],
                    'InterphaseInterval': [float(i) for i in str(dictParams['XIPP Interphase Interval (uSec)']).split(' ')],
                    'Phase1_Width': [float(i) for i in str(dictParams['XIPP Phase 1 Width (uSec)']).split(' ')],
                    'Phase1_Amp': [float(i) for i in str(dictParams['XIPP Phase 1 Amp (uA)']).split(' ')],
                    'Phase2_Width': [float(i) for i in str(dictParams['XIPP Phase 2 Width (uSec)']).split(' ')],
                    'Phase2_Amp': [float(i) for i in str(dictParams['XIPP Phase 2 Amp (uA)']).split(' ')],        
                }

        if not existsMicroStim:
            microStimParams['microStim'] = labInfo['StimDefaults']['MicroStim']
        
        # Check if exist MicroStim based on amplitude, freq, Duration
        widthCheck1 = any([value>0 for value in microStimParams['microStim']['Phase1_Width']])
        amplitudeCheck1 = any([value!=0 for value in microStimParams['microStim']['Phase1_Amp']])
        widthCheck2 = any([value>0 for value in microStimParams['microStim']['Phase2_Width']])
        amplitudeCheck2 = any([value!=0 for value in microStimParams['microStim']['Phase2_Amp']])
        durationCheck = any([value>0 for value in microStimParams['microStim']['Duration']])
        freqCheck = any([value>0 for value in microStimParams['microStim']['Frequency']])
        chanCheck = any([value>0 for value in microStimParams['microStim']['Channel']])
        stimCheck = microStimParams['microStim']['Stimulus']>0
        widthCheck = any([widthCheck1, widthCheck2])
        amplitudeCheck = any([amplitudeCheck1, amplitudeCheck2])

        microStimParams['valid'] = all([stimCheck, chanCheck, widthCheck, amplitudeCheck, durationCheck, freqCheck]) 

        return microStimParams

    #################################################################################
    # GET VisualCue Paramters from YAML stim-dictionary
    #################################################################################
    @classmethod
    def getVisualStimParams(cls, dictStim):

        visualStimParams = {'valid': False}

        # Last dict from STIM should be Visual 
        visualStimParams['visualStim'] = {
            'StartTime': float(dictStim[-1]['Cue Start Time']),
            'Duration': float(dictStim[-1]['Cue Duration']),
            'ID': [int(visID) for visID in str(dictStim[-1]['ID']).split(' ') if int(visID)>0],
        }

        visualStimParams['visualStim']['nStim'] = len(visualStimParams['visualStim']['ID'])

        visualStimParams['valid'] = visualStimParams['visualStim']['Duration']>0 and visualStimParams['visualStim']['nStim']>0

        return visualStimParams
    
    ######################################################################################################
    # GET SHAKER INFORMATION BodyPlacement, Calibration coefficients & accelerometer sensitivity
    # from YAML dictionary
    ######################################################################################################
    @classmethod
    def getTactorInfo(cls, dictYAML):

        accelerometer = [float(c) for c in str(dictYAML['Experimental Visual Settings']['Accelerometer Sensitivity']).split(' ')]

        leftCoeff = [float(i) for i in str(dictYAML['Left Vibe Stim Coeffs']).split(' ')]
        rightCoeff = [float(i) for i in str(dictYAML['Right Vibe Stim Coeffs']).split(' ')]
        
        # Get how many freqs were tested with calibration
        shakerInfo = {
            'freqCalibration': [],
            'leftCoeffA': [],
            'leftCoeffB': [],
            'rightCoeffA': [],
            'rightCoeffB': [],
        }
        for f in range(0, len(leftCoeff), 3):
            if leftCoeff[f]==rightCoeff[f]:
                shakerInfo['freqCalibration'].append(leftCoeff[f])
                shakerInfo['leftCoeffA'].append(leftCoeff[f+1])
                shakerInfo['leftCoeffB'].append(leftCoeff[f+2])
                shakerInfo['rightCoeffA'].append(rightCoeff[f+1])
                shakerInfo['rightCoeffB'].append(rightCoeff[f+2])

        shakerInfo.update(labInfo['ShakerInfo'])
        
        return {
            'leftBodyPart' : str(dictYAML['Subject Info']['Left Placement']['Body Placement']),
            'leftSegment' : str(dictYAML['Subject Info']['Left Placement']['Segment']),
            'leftIndentation' : float(dictYAML['Subject Info']['Left Placement']['Indentation Depth']),
            'leftAcclSensitivity': accelerometer[0],
            'rightBodyPart' : str(dictYAML['Subject Info']['Right Placement']['Body Placement']),
            'rightSegment' : str(dictYAML['Subject Info']['Right Placement']['Segment']),
            'rightIndentation' : float(dictYAML['Subject Info']['Right Placement']['Indentation Depth']),
            'rightAcclSensitivity': accelerometer[1],
            'device': shakerInfo,
        }

    #################################################################################
    # GET ALL VISUALCUE(s) INFORMATION (nVisualCues, IDs, shapes, 'RGBA', 'Position') 
    # from YAML dictionary
    #################################################################################
    @classmethod
    def getVisualCueInfo(cls, dictYAML):

        visualCuesDict = dictYAML['Experimental Visual Settings']['Visual Cue Settings']
        visualCuesInfo = {'n': len(visualCuesDict), 'ID': [], 'Shape':[], 'Size': [], 'RGBA': [], 'Position': []}

        for _, visualCue in visualCuesDict.items():
                visualCuesInfo['ID'].append(int(visualCue['ID']))
                visualCuesInfo['Shape'].append(visualCue['Shape'])
                visualCuesInfo['Size'].append([float(s) for s in str(visualCue['Size']).split(' ')])
                visualCuesInfo['RGBA'].append([float(s) for s in str(visualCue['Color']).split(' ')])
                visualCuesInfo['Position'].append([float(s) for s in str(visualCue['Pos']).split(' ')])
        
        # Check when Visual Cues END
        if 'Cue Off at Post Choice End' in dictYAML['Experimental Visual Settings']['Control Settings'].keys():
            if dictYAML['Experimental Visual Settings']['Control Settings']['Cue Off at Post Choice End']:
                visualCuesInfo['visualENDwith'] = 'fixationOFF'
            else:
                visualCuesInfo['visualENDwith'] = 'choiceTargetON'
        else:
            visualCuesInfo['visualENDwith'] = 'choiceTargetON'

        return visualCuesInfo

    #####################################################################################
    # GET ALL CHOICETARGETs INFORMATION (nChoiceTargets, ID, shape, 'RGBA', 'Position', ''ChoiceTargets_Window'')
    # from YAML dictionary
    #####################################################################################
    @classmethod
    def getChoiceTargetInfo(cls, dictYAML):
        
        choiceTargetDict = dictYAML['Experimental Visual Settings']['Choice Target']
        choiceTargetsInfo = {'n': len(choiceTargetDict), 'ID': [], 'Shape':[], 'Size': [], 'RGBA': [], 'Position': [],
                             'ChoiceTargets_Window': [
                                float(dictYAML['Experimental Visual Settings']['Choice Target Window Size']['X']),
                                float(dictYAML['Experimental Visual Settings']['Choice Target Window Size']['Y'])
                                ]}

        for _, choiceTarget in choiceTargetDict.items():
                choiceTargetsInfo['ID'].append(int(choiceTarget['ID']))
                choiceTargetsInfo['Shape'].append(choiceTarget['Shape'])
                choiceTargetsInfo['Size'].append([float(s) for s in str(choiceTarget['Size']).split(' ')])
                choiceTargetsInfo['RGBA'].append([float(s) for s in str(choiceTarget['Color']).split(' ')])
                choiceTargetsInfo['Position'].append([float(s) for s in str(choiceTarget['Pos']).split(' ')])

        return choiceTargetsInfo
    
    #####################################################################################
    # GET INFORMATION From FIXATION POINT (shape, 'RGBA', 'Position', FixTargetWindow)
    #####################################################################################
    @classmethod
    def getFixTargetInfo(cls, dictYAML):
        fixTarget = dictYAML['Experimental Visual Settings']['Fixation Target']
        fixWindow = dictYAML['Experimental Visual Settings']['Fixation Window Size']
        return {
            'fixTarget_Shape': fixTarget['Shape'],
            'fixTarget_Size': [float(val) for val in fixTarget['Size'].split(' ')],
            'fixTarget_RGBA': [float(val) for val in fixTarget['Color'].split(' ')],
            'fixTarget_Position': [float(val) for val in fixTarget['Pos'].split(' ')],
            'fixTarget_Window': [float(fixWindow['X']), float(fixWindow['Y'])],
        }
    
    ###########################################################
    # GET START DATETIME
    ###########################################################
    @classmethod
    def getStartDateTime(cls, dictYAML, TimeZone = 'America/Chicago'):
        
        yamlDateTime = dt.datetime.strptime(dictYAML['Experiment Started'][0:-9] + 'T' + 
                dictYAML['Experiment Started'][-8:], 
                        "%Y-%m-%dT%H:%M:%S"
            )
        
        yamlDateTime = pytz.timezone(TimeZone).localize(yamlDateTime)
        yamlDate = dt.datetime.date(yamlDateTime)

        startTimeSecs = cls.getStartTimeSecs(dictYAML)
        
        if (startTimeSecs%1)==0:
            startTimeSecs_str = str(dt.timedelta(seconds=startTimeSecs)) + '.000'
        else:
            startTimeSecs_str = str(dt.timedelta(seconds=startTimeSecs))

        startTime = dt.datetime.strptime(startTimeSecs_str, "%H:%M:%S.%f").time()

        trial1Start = dt.datetime.combine(yamlDate, startTime)
        startDateTime = pytz.timezone(TimeZone).localize(trial1Start)

        return startDateTime
    
    @classmethod
    def getStartTimeSecs(cls, dictYAML):

        firstMarker = cls.getTrial_by_Num(dictYAML, repID = 1, trialID = 1)['Marker Signal Sequence'][0]

        # check first Marker is "1" = Fixation Target ON (first trial Start)
        if firstMarker['Marker'] == 1:
            startTimeSecs = firstMarker['Time']
        else:
            raise Exception("First Marker was {}.. it is not recognized to start Experiment, it has to be 1 ".format(
                firstMarker['Marker']))
        
        return startTimeSecs  
    
    @classmethod
    def getStopTimeSecs(cls, dictYAML):

        expFinished = list(dictYAML.keys()).count('Experiment Finished')==1

        if expFinished:
            expEnd = dt.datetime.strptime(dictYAML['Experiment Finished'][0:-9] + 'T' + 
                        dictYAML['Experiment Finished'][-8:], 
                        "%Y-%m-%dT%H:%M:%S")
        else:
            
            # Get last marker
            lastRep = cls.getNumReps(dictYAML)
            lastTrialID = cls.getNumTrialsRep(dictYAML, lastRep)

            trial = cls.getTrial_by_Num(dictYAML, lastRep, lastTrialID)

            # Get last Time of Marker Signal Sequence
            expEndSeconds = max([marker['Time'] for marker in trial['Marker Signal Sequence']])
            expEndTime = dt.datetime.strptime(str(dt.timedelta(seconds=expEndSeconds)), "%H:%M:%S.%f").time()

            yamlDateTime = dt.datetime.strptime(dictYAML['Experiment Started'][0:-9] + 'T' + 
                dictYAML['Experiment Started'][-8:], 
                        "%Y-%m-%dT%H:%M:%S"
                )
            yamlDate = dt.datetime.date(yamlDateTime)
            expEnd = dt.datetime.combine(yamlDate, expEndTime)

        expDayZero = expEnd.replace(hour=0, minute=0, second=0, microsecond=0)

        return (expEnd-expDayZero).seconds
    
    @classmethod
    def getNumReps(cls, dictYAML):
        reps = len(dictYAML['Experiment Log'])
        if reps <1:
            raise Exception('experimentalYAML session does not have any repetition .. ')
        else:
            return reps
    
    @classmethod
    def getNumTrialsRep(cls, dictYAML, repID):
        if repID > cls.getNumReps(dictYAML):
            print(" repID ({}) must be within tested reps (nReps = {})\n".format(repID, cls.getNumReps(dictYAML)))
            return 0
        else:
            return len(dictYAML['Experiment Log']['Repeat ' + str(int(repID))])
        
    @classmethod
    def getTrial_by_Num(cls, dictYAML, repID, trialID):
        if trialID > cls.getNumTrialsRep(dictYAML, repID):
            print('trialID {} is out of bound (repID {} has {} trials)\n'.format(trialID, repID, cls.getNumTrialsRep(dictYAML, repID)))
            return None
        else:
            return dictYAML['Experiment Log']['Repeat ' + str(int(repID))]['Trial ' + str(int(trialID))]
        
    @classmethod   
    def getTrial_by_Index(cls, dictYAML, repID, trialIndex):
        maxTrials = cls.getNumTrialsRep(dictYAML, repID) - 1
        if trialIndex>maxTrials:
            print('trialIndex {} is out of bound (repID {} has {} trials)\n'.format(max(trialIndex), repID, maxTrials+1))
            return None
        else:
            return cls.getTrial_by_Num(dictYAML, repID, trialIndex+1)
    
    ###################################################################################################
    # split Marker Signal Sequence in case the trial was on correct Trial Mode
    # extract the outcome and the sequence for each individual trial in a list
    # Handle with last trials in case the experiment was stopped before all reps were completed
    ###################################################################################################
    @classmethod  
    def getTrial(cls, dictYAML, repID, trialID, isLastTrial=None):

        expStopped = False

        if isLastTrial is None:
            lastRep = cls.getNumReps(dictYAML)
            lastTrialID = cls.getNumTrialsRep(dictYAML, lastRep)
            isLastTrial = lastRep==repID and lastTrialID==trialID

        if isLastTrial:
            expStopped = list(dictYAML.keys()).count('Experiment Stopped')==1
        
        trial = cls.getTrial_by_Num(dictYAML, trialID=trialID, repID=repID)

        # Repeat all KEYS except OutCome and MarkerSignalSequence        
        commonDictTrial = {key:value for key, value in trial.items() if not trialKeys2remove.count(key)}

        markerID = [dictMarker['Marker'] for dictMarker in trial['Marker Signal Sequence']]
        markerTime = [dictMarker['Time'] for dictMarker in trial['Marker Signal Sequence']]

        nTrials = markerID.count(1)
        trial_i = [i for i in range(0, len(markerID)) if markerID[i]==1]
        trial_e = [i for i in range(0, len(markerID)) if markerID[i]==9]

        # Is possible that marker9 was not saved in the last trial
        # that can happen only if it is the last trial and the experiment was stopped
        if len(trial_e)+1==len(trial_i) and expStopped and isLastTrial:
            # IF Stop-Marker was not saved, APPEND marker16 and marker9
            if markerID.count(16)==0:
                endTimeSecs = cls.getStopTimeSecs(dictYAML)
                markerID.append(16)
                markerTime.append(endTimeSecs)
                markerID.append(9)
                markerTime.append(endTimeSecs)
            # IF Stop-Marker exist, only APPEND marker9
            else:
                markerID.append(9)
                markerTime.append(markerTime[markerID.index(16)])

            # update start & end Index
            trial_i = [i for i in range(0, len(markerID)) if markerID[i]==1]
            trial_e = [i for i in range(0, len(markerID)) if markerID[i]==9]

        if len(trial_i)!=len(trial_e):
            raise Exception('Number of startTrialIndex ({}) does not match the number of endTrialIndex ({})'.format(len(trial_i), len(trial_e)))

        # Split into trials 
        outcomesIDs = list(dictOutcomes.keys())
        listTrial = []

        for t in range(0, nTrials):

            markerSeq = markerID[trial_i[t]:trial_e[t]+1]
            markerTimeSeq = markerTime[trial_i[t]:trial_e[t]+1]

            outcome = [marker for marker in markerSeq if marker in outcomesIDs]
            if len(outcome)>1:
                if outcome.count(16)==0:
                    raise Exception ('{} outcomes ({}) founded in trial {}, rep {}. It has to be only 1 outcome'.format(len(outcome), outcome, trialID, repID))
                else:
                    outcome = [marker for marker in outcome if marker!=16]  

            if len(outcome)!=1:
                raise Exception ('{} outcomes ({}) founded in trial {}, rep {}. It has to be only 1 outcome'.format(len(outcome), outcome, trialID, repID))

            listTrial.append({**commonDictTrial, ** {
                'repID': repID,
                'trialNum': t+1,
                'repTrial': t>0,
                'outcomeID': outcome[0],
                'outcomeLabel': dictOutcomes[outcome[0]],
                'outcomeTime': markerTimeSeq[markerSeq.index(outcome[0])],
                'markerID': markerSeq,
                'markerTime': markerTimeSeq
            }})
        
        return listTrial
    
    ###################################################################################################
    # Estract all trials in a list
    ###################################################################################################
    @classmethod
    def getALLtrials(cls, dictYAML):

        listTrials = []
        nReps = cls.getNumReps(dictYAML)

        for r in range(nReps):
            nTrials = cls.getNumTrialsRep(dictYAML, repID=r+1)
            for t in range(nTrials):
                isLastTrial = False
                if r+1==nReps and t+1==nTrials:
                    isLastTrial = True
                
                trialList = cls.getTrial(dictYAML, repID=r+1, trialID=t+1, isLastTrial=isLastTrial)

                for rt in trialList:
                    listTrials.append(rt)
        
        return listTrials
    
    ###################################################################################################
    # Get Column Names & descriptions based on the number of Stim
    ###################################################################################################
    @classmethod
    def getTrialColNames(cls, dictYAML, analogTemp=None):

        maxStimTypes = cls.getMaxStimTypes(dictYAML)
        secsBeforeNext = labInfo['SecsToStopPreviousTrial'][0]

        addTempCol = False
        tempDescription = 'Temperature (centigrades) of the Thermistor from ['
        if analogTemp is not None:
            if analogTemp['exists']:
                addTempCol = True
                for themID in analogTemp['thermistorIDs']:
                    tempDescription += '{},'.format(themID)
        tempDescription += ']'

        # key = colName
        # values = Description
        names = {
            'start_time': 'trial start time (default name from NWB), TimeStamp when the fixation target was presented ON the screen (Marker1)',
            'stop_time': 'trial end time (default name from NWB), it corresponds to {} secs before the next Trial start_time (if is the last Trial, it will be = ExperimentStop)'.format(secsBeforeNext),
            'roundID': 'Repetition number of the current trial',
            'trialID': 'Trial condition (trialID from the configure file)',
            'repeatedTrial': 'whether or not the trial is a repetition of the previous one (correctTrialMode = ON)',
            'outcomeID': 'Marker ID indicating behavioral outcome : 10, 11, 12, 13, 14, 140, 141, 16, 30+X, 60+X',
            'outcomeDescription': 'Text description of the behavioral outcome',
            'outcome_time': 'TimeStamp when the outcome occurred',
            'fixTarget_OFF_time': 'TimeStamp when the fixation target is OFF the screen (Marker 18)',
            'fixTarget_ON_analog_time': 'TimeStamp when the fixation is ON the screen (Marker 1000, extracted from Left-Photodiode signal)',            
            'fixTarget_OFF_analog_time': 'TimeStamp when the fixation is OFF the screen (Marker 18000, extracted from Left-Photodiode signal)',
            'fix_FixTarget_time': 'TimeStamp when the subject fix (foot or eye or both) the fixation target (hold both bars for foot). (Marker 2)',
            'fixMode': 'eye, eyeNoPostChoice, foot or mouse',
            'fixTarget_Window': "[width, height] (units=degrees) of the tolerance window for the fixation (dimensions are relative to the center of the fixationTarget Position)",
            'fixTarget_Shape': 'fixation target Shape',
            'fixTarget_Size': "[width, height] (units=degrees) of the fixationTarget",
            'fixTarget_RGBA': 'fixation target Color, vector of 4 values between 0 - 1: [r g b alpha]',
            'fixTarget_Position': '[X, Y] postition in degrees relative to the interocular center',          
        }

        if maxStimTypes['nStim']>0:
            names['nStim'] = 'Number of Stimulation Events (either Tactile, MicroStim and/or Visual)'
            for s in range(1, maxStimTypes['nStim']+1):
                names.update({
                    'stim'+str(s)+'_ON_time': 'TimeStamp when stim{} event started'.format(s),
                    'stim'+str(s)+'_OFF_time': 'TimeStamp when stim{} event ended'.format(s)
                })
        
        
        if maxStimTypes['nTactileStim']>0:
            names['nTactile'] = 'Number of Tactile Stim on each trial'
            # Append TACTILE Stim
            for s in range(1, maxStimTypes['nTactileStim']+1):
                names.update({
                    'tact'+str(s)+'_Left_ON_time': 'TimeStamp LEFT tact{}_ON (time extrapolated from Marker 3 plus the tact{}_Left-StartTime parameter)'.format(s, s),
                    'tact'+str(s)+'_Left_OFF_time': 'TimeStamp LEFT tact{}_OFF (time extrapolated from "tact{}_Left_ON" plus "tact{}_Left_DUR")'.format(s, s, s),
                    'tact'+str(s)+'_Left_ON_analog_time': 'TimeStamp LEFT tact{}_ON (Marker 3001, extracted from Left-Accelerometer signal)'.format(s),
                    'tact'+str(s)+'_Left_OFF_analog_time': 'TimeStamp LEFT tact{}_OFF (Marker 4001, extracted from Left-Accelerometer signal)'.format(s),
                    'tact'+str(s)+'_Left_FR': 'Frequency (Hz) of the left shaker stimuli',
                    'tact'+str(s)+'_Left_AMP': 'Amplitude (micrometers) of the left shaker stimuli',
                    'tact'+str(s)+'_Left_DUR': 'Duration (seconds) of the left shaker stimuli',
                    'tact'+str(s)+'_Right_ON_time': 'TimeStamp RIGHT tact{}_ON (time extrapolated from Marker 3 plus the tact{}_Right-StartTime parameter)'.format(s, s),
                    'tact'+str(s)+'_Right_OFF_time': 'TimeStamp RIGHT tact{}_OFF (time extrapolated from "tact{}_Right_ON" plus "tact{}_Right_DUR")'.format(s, s, s),
                    'tact'+str(s)+'_Right_ON_analog_time': 'TimeStamp RIGHT tact{}_ON (Marker 3002, extracted from Right-Accelerometer signal)'.format(s),
                    'tact'+str(s)+'_Right_OFF_analog_time': 'TimeStamp RIGHT tact{}_OFF (Marker 4002, extracted from Right-Accelerometer signal)'.format(s),
                    'tact'+str(s)+'_Right_FR': 'Frequency (Hz) of the right shaker stimuli',
                    'tact'+str(s)+'_Right_AMP': 'Amplitude (micrometers) of the right shaker stimuli',
                    'tact'+str(s)+'_Right_DUR': 'Duration (seconds) of the right shaker stimuli',
                })
        
        # Information about tactile Placement
        names.update({
            'leftPlacement' : 'Body part & segment part where the LEFT shaker probe was placed',
            'leftIndentation': 'Indentantion depth of the LEFT shaker probe (microns)',
            'rightPlacement' : 'Body part & segment part where the RIGHT shaker probe was placed',
            'rightIndentation': 'Indentantion depth of the RIGHT shaker probe (microns)',
        })

        # Information about Probe Temperature
        if addTempCol:
            names.update({
                'trialTemp': tempDescription
            })
        
        if maxStimTypes['nMicroStim']>0:
            # APPEND MICROSTIM
            names['nMicroStim'] = 'Number of MicroStimulation stimuli on each trial. Each stimuli is compose by three consecutive trains'
            for s in range(1, maxStimTypes['nMicroStim']+1):
                names.update( {
                    'XIPP_Stim'+ str(s)+ '_ON_time': 'Timestamp when MicroStimulation-{} START (Marker 3 + startTime parameter)'.format(s),
                    'XIPP_Stim'+ str(s)+ '_OFF_time': 'Timestamp when MicroStimulation-{} ENDS (Marker 91)'.format(s),
                    'XIPP_Stim'+ str(s)+ '_ON_analog_time': 'Timestamp when MicroStimulation-{} START (Marker 91001, extracted from Ripple stim-TimeStamps)'.format(s),
                    'XIPP_Stim'+ str(s)+ '_OFF_analog_time': 'Timestamp when MicroStimulation-{} ENDS (Marker 91002, extracted from Ripple stim-TimeStamps)'.format(s),
                    'XIPP_Stim'+ str(s)+ '_Channel': 'Electrodes used to deliver MicroStimulation-{} (up to four simultaneous channels simultaneously)'.format(s),
                    'XIPP_Stim'+ str(s)+ '_ChannelStart_time': 'Onset time of MicroStimulation-{} from each electrode (extracted from Ripple stim-TimeStamps)'.format(s),
                    'XIPP_Stim'+ str(s)+ '_ChannelStop_time': 'Onset time of MicroStimulation-{} from each electrode (extracted from Ripple stim-TimeStamps)'.format(s),
                    'XIPP_Stim'+ str(s)+ '_ReturnChannel': 'Return channel for MicroStimulation-{}'.format(s),
                    'XIPP_Stim'+ str(s)+ '_Duration': 'Duration (secs) of each of the three consecutive trains.',
                    'XIPP_Stim'+ str(s)+ '_Frequency': 'Frequency (Hz) of the bi-phasic pulse. One value for each of the three consecutive trains',
                    'XIPP_Stim'+ str(s)+ '_InterphaseInterval': 'Gap (uSecs) between each phase of the bi-phasic pulse. One value for each of the three consecutive trains',
                    'XIPP_Stim'+ str(s)+ '_Phase1_Width': 'Duration (uSecs) of the first phase of the bi-phasic pulse. One value for each of the three consecutive trains',
                    'XIPP_Stim'+ str(s)+ '_Phase1_Amp': 'Amplitude (uA) of the first phase of the bi-phasic pulse. One value for each of the three consecutive trains', 
                    'XIPP_Stim'+ str(s)+ '_Phase2_Width': 'Duration (uSecs) of the second phase of the bi-phasic pulse. One value for each of the three consecutive trains',
                    'XIPP_Stim'+ str(s)+ '_Phase2_Amp': 'Amplitude (uA) of the second phase of the bi-phasic pulse. One value for each of the three consecutive trains', 

                })

        # Append VISUAL Stim
        if maxStimTypes['nVisualStim']>0:
            names['nVisual'] = 'Number of Visual Stim on each trial'
            for s in range(1, maxStimTypes['nVisualStim']+1):
                names.update( {
                    'vis'+str(s)+'_ON_time': 'TimeStamp when the Visual Stim{} START (Marker 15)'.format(s),
                    'vis'+str(s)+'_OFF_time': 'TimeStamp when the Visual Stim{} END (marker 15 plus Visual Stim duration)'.format(s),
                    'vis'+str(s)+'_ON_analog_time': 'TimeStamp when the Visual Stim{} START (Marker 15000, extracted from Right-Photodiode signal)'.format(s),
                    'vis'+str(s)+'_OFF_analog_time': 'TimeStamp when the Visual Stim{} END (marker 15000 plus Visual Stim duration)'.format(s),
                    'vis'+str(s)+'_ID': 'Identifier of Visual Stim{} out of all the possible visual stimuli'.format(s),
                    'vis'+str(s)+'_Shape': 'Visual Stim{} shape'.format(s),
                    'vis'+str(s)+'_Size': '[width, height] (units=degrees) of Visual Stim{}'.format(s),
                    'vis'+str(s)+'_RGBA': 'Visual Stim{} Color, vector of 4 values between 0 - 1: [r g b alpha]'.format(s),
                    'vis'+str(s)+'_Position': '[X, Y] Visual Stim{} postition in degrees relative to the interocular center'.format(s),
                })

        
        # APPEND Go-cue LABELS
        names.update({
            'responseMode': 'Response type: eye, foot, noResponse (passive condition) or mouse',
        })

        responseMode = cls.getReponseMode(dictYAML)
        # When foot-Off mode is selected, No choice targets are shown (passive conditions)
        if responseMode != 'noResponse':
            names.update({
                'choiceTarget_ON_time': 'TimeStamp when the visual Choice Targets are ON the screen (Marker 5)',
                'choiceTarget_OFF_time': 'TimeStamp when the visual Choice Targets are OFF the screen (i.e., Reward Start or outcomeTimeStamp)',
                'choiceTarget_ON_analog_time': 'TimeStamp when the visual Choice Targets are ON the screen (Marker 5000, extracted from Right-Photodiode signal)',
                'choiceTarget_OFF_analog_time': 'TimeStamp when the visual Choice Targets are OFF the screen (i.e. analog Reward Start or outcomeTimeStamp)',
                'correctChoiceTarget_ID': 'Identifier of the correct ChoiceTarget (number relative to all the possible visual ChoiceTargets)',
                'selectedChoiceTarget_ID': 'Identifier of the chosen ChoiceTarget (number relative to all the possible visual ChoiceTargets)',
                'showingCorrectChoiceTarget': 'Boolean indicating if only the correct ChoiceTarget was shown'
            })

            names['nChoiceTargets'] = 'Number of possible visual ChoiceTargets shown on each trial (if showingOnlyCorrectTarget is TRUE, only the correct one was shown)'
            names['choiceTargets_Window'] =  "[width, height] (units=degrees) of the tolerance window for the ChoiceTarget fixation (dimensions are relative to the center of each ChoiceTarget Position)"
            for s in range(1, maxStimTypes['nChoiceTargetsShown']+1):
                names.update({
                    'choice'+str(s)+'_ID': 'Identifier of ChoiceTarget{} from all the possible visual ChoiceTargets'.format(s),
                    'choice'+str(s)+'_Shape': 'visual ChoiceTarget{} shape'.format(s),
                    'choice'+str(s)+'_Size': "[width, height] (units=degrees) of the visual ChoiceTarget{}".format(s),
                    'choice'+str(s)+'_RGBA': 'visual ChoiceTarget{} Color, vector of 4 values between 0 - 1: [r g b alpha]'.format(s),
                    'choice'+str(s)+'_Position': '[X, Y] visual ChoiceTarget{} postition in degrees relative to the interocular center'.format(s),
                })
        
        # Mov-Response Markers
        names.update({
            'responseStart_time': 'eyeResponse: saccade Start (eyePosition out of fixationTarget window). footResponse: first foot release the bar. (Marker 6)',
            'fix_ChoiceTarget_time': 'eyeResponse: fixation to ChoiceTarget start. footResponse: start to count Response time from footResponse (should be almost the same as responseStart, except for ReleaseBoth). (Marker 7)',
            'rewardStart_time': 'reward valve open (Marker 8)',
            'rewardType': 'reward type (water, juice, etc)',
            'trialEnd_time': 'reward Stop (correct trials) or penalty time ends (incorrect/abort trials) (Marker 9). Also, experiment can be stopped before repetitions finish (Marker 16)'
        })

        return names
    
    @classmethod
    def parserTrial(cls, trial, expStartTime, starTime_nextTrial, maxStimTypes, fixMode, fixTargetInfo, 
                      visualCuesInfo, responseMode, choiceTargetInfo, shakerInfo, microStimChannel):
        
        # print('.......parsing Trial: {}'.format(trial['trialNum']))

        stop_Time = float(starTime_nextTrial - labInfo['SecsToStopPreviousTrial'][0])
        
        defaultNoTime = float(labInfo['StimDefaults']['NoTime'])
        defaultNoStimParam = float(labInfo['StimDefaults']['NoStimParam'])
        toleranceTactileStim = float(labInfo['MarkerOffsetTolerance']['TactileStim'][0])
        toleranceMicroStim = float(labInfo['MarkerOffsetTolerance']['MicroStim'][0])
        toleranceVisualCue = float(labInfo['MarkerOffsetTolerance']['VisualCue'][0])
        tolerancePhotodiode = float(labInfo['MarkerOffsetTolerance']['PhotodiodeON'][0])

        # Be sure Markers are sorted by Time:
        nMarkers = len(trial['markerID'])
        markerIndex = list(numpy.argsort(numpy.array(trial['markerTime'])))
        markerID = [trial['markerID'][i] for i in markerIndex]
        markerTime = [trial['markerTime'][i] for i in markerIndex]
        
        #########################################################
        # Get Tactile, MicroStim & Visual STIM-params info 
        tactStimParams = [cls.getTactileStimParams(value) for key, value in trial.items() if key.startswith('Stim ')]
        microStimParams = [cls.getMicroStimParams(value, microStimChannel=microStimChannel, expStartTime=expStartTime) for key, value in trial.items() if key.startswith('Stim ')]
        visualStimParams = [cls.getVisualStimParams(value) for key, value in trial.items() if key.startswith('Stim ')]

        ##########################################################
        # Get shown ChoiceTargetIDs
        correctChoiceTarget_ID = int(trial['Correct Target'])
        showingCorrectChoiceTarget = trial['Show Correct Target Only']==1
        if showingCorrectChoiceTarget:
            choiceTargetsList = correctChoiceTarget_ID
        else:
            choiceTargetsList = sorted([int(c) for c in str(trial['Showing Target IDs']).split(' ')])          
        nChoiceShown = len(choiceTargetsList)

        ####################################################
        # GET selected ChoiceTarget
        rewardType = 'None'
        if trial['outcomeID']==10:
            selectedChoiceTarget_ID = correctChoiceTarget_ID
            rewardType = trial['Reward Type']
        elif trial['outcomeID']>30 and trial['outcomeID']<61:
            selectedChoiceTarget_ID = int(trial['outcomeID']-30)
        elif trial['outcomeID']>60 and trial['outcomeID']<91:
            selectedChoiceTarget_ID = int(trial['outcomeID']-60)
        else:
            selectedChoiceTarget_ID = int(0)

        ###################################################################################
        # get MARKERS related to TIME events
        # markers higher than "marker 1" might not exists on aborted trials
        ###################################################################################
        
        ###################################################################################
        # Marker 1000 : Analog Fix ON the screen
        if markerID.count(1000):
            fixTimeON_analog = markerTime[markerID.index(1000)]
        else:
            fixTimeON_analog = expStartTime + defaultNoTime
            
        ###################################################################################
        # Marker 2 : Fixation-Fix Started
        if markerID.count(2): 
            fix_FixTarget_Start = markerTime[markerID.index(2)]
        else:
            fix_FixTarget_Start = expStartTime + defaultNoTime

        ###################################################################################
        # markers related to each stimID
        ###################################################################################

        ###################################
        # Marker 3 : StimStart
        nStimON = markerID.count(3)
        stimTimeON = [markerTime[indx] for indx in range(nMarkers) if markerID[indx]==3]

        ###################################
        # Marker 4 : StimEnd
        nStimEND = markerID.count(4)
        stimTimeEND = [markerTime[indx] for indx in range(nMarkers) if markerID[indx]==4]

        ####################################################################
        # Marker 3001, 4001. Analog-markers will appears for each marker-3
        leftAnalog = markerID.count(3001)
        leftTimeON_analog = [markerTime[indx] for indx in range(nMarkers) if markerID[indx]==3001]
        leftTimeOFF_analog = [markerTime[indx] for indx in range(nMarkers) if markerID[indx]==4001]

        ####################################################################
        # Marker 3002, 4002. Analog-markers will appears for each marker-3
        rightAnalog = markerID.count(3002)
        rightTimeON_analog = [markerTime[indx] for indx in range(nMarkers) if markerID[indx]==3002]
        rightTimeOFF_analog = [markerTime[indx] for indx in range(nMarkers) if markerID[indx]==4002]

        ###################################
        # Marker 91 : MicroStimEnd
        nMicroStimOFF = markerID.count(91)
        microStimTimeOFF = [markerTime[indx] for indx in range(nMarkers) if markerID[indx]==91]

        ###################################
        # Marker 15: VisualCueStart
        nVisON = markerID.count(15)
        visTimeON = [markerTime[indx] for indx in range(nMarkers) if markerID[indx]==15]

        #######################################
        # Marker 15000: Analog VisualCueStart
        nVisON_analog = markerID.count(15000)
        visTimeON_analog = [markerTime[indx] for indx in range(nMarkers) if markerID[indx]==15000]

        ####################################################
        # Marker 5: visual Choice Targets are ON the screen
        if markerID.count(5)>0: 
            choiceTargetON = markerTime[markerID.index(5)]
            nChoiceShownTrial = nChoiceShown
        else:
            choiceTargetON = expStartTime + defaultNoTime
            nChoiceShownTrial = 0

        ###################################################################################
        # Marker 5000 : Analog visual Choice Targets are ON the screen
        existChoiceTargetON_analog = False
        if markerID.count(5000)>0:
            choiceTargetON_analog = markerTime[markerID.index(5000)]
            existChoiceTargetON_analog = True
        else:
            choiceTargetON_analog = expStartTime + defaultNoTime

        ####################################################
        # Marker 18: fixationTarget is OFF the screen
        if markerID.count(18): 
            fix_FixTarget_End = markerTime[markerID.index(18)]
        else:
            if trial['outcomeLabel'].count('abort') or trial['outcomeLabel'].count('experiment stopped'):
                fix_FixTarget_End = trial['outcomeTime']
            else:
                fix_FixTarget_End = expStartTime + defaultNoTime

        ###################################################################################
        # Marker 1800 : Analog fixationTarget is OFF the screen
        existfixTimeOFF_analog = False
        if markerID.count(18000):
            fixTimeOFF_analog = markerTime[markerID.index(18000)]
            existfixTimeOFF_analog = True
        else:
            fixTimeOFF_analog = expStartTime + defaultNoTime
        
        ###################################################################################
        # markers related to Response Events
        ###################################################################################
        
        ####################################################
        # Marker 6: saccade/footResponse start
        if markerID.count(6):
            responseMovStart = markerTime[markerID.index(6)]
        else:
            responseMovStart = expStartTime + defaultNoTime

        ####################################################
        # Marker 7: fix ChoiceTarget start
        if markerID.count(7):
            fixChoiceTargetStart = markerTime[markerID.index(7)]
        else:
            fixChoiceTargetStart = expStartTime + defaultNoTime

        ####################################################
        # Marker 8: Reward start
        choiceTargetOFF_list = [trial['outcomeTime']]
        if markerID.count(8):
            rewardStart = markerTime[markerID.index(8)]
            choiceTargetOFF_list.append(rewardStart)
        else:
            rewardStart = expStartTime + defaultNoTime

        ####################################################
        # visual Choice Targets OFF the screen
        if nChoiceShownTrial>0: 
            choiceTargetOFF = min(choiceTargetOFF_list)
        else:
            choiceTargetOFF = expStartTime + defaultNoTime

        ####################################################
        # "analog" visual Choice Targets OFF the screen
        if existChoiceTargetON_analog: 
            choiceTargetOFF_analog = min(choiceTargetOFF_list)
        else:
            choiceTargetOFF_analog = expStartTime + defaultNoTime

        ####################################################
        # Marker 9: trial Ends (penalty-End or Reward-End)
        if markerID.count(9):
            trialEnd = markerTime[markerID.index(9)]
        else:
            if trial['outcomeLabel'].count('experiment stopped'):
                trialEnd = trial['outcomeTime']
            else:
                trialEnd = expStartTime + defaultNoTime

        #############################################################################################################
        # If StimStart was recorded, then Tactile & Microstim Events will happen regarless if the monkey aborts
        # Visual Cues might not occurr if the monkey aborts before CueON-time. 
        # TactileStim & MicroStim will be completed even if the monkey aborts in the middle of the stimulation
        # Marker 4 will not exists if the monkey aborts before tactileStim finishes.
        # Marker 4 indicates the end of the TactileStim regardless of MicroStim and Visual durations
        # If the monkey aborts, all VisualCues will be off immediately

        # it is assume that trial was sorted in sequence
        # Construct timeON-OFF for each type of Stim
        stimTimeOFF = []
        stim_tactile = []
        stim_microStim = []
        stim_Visual = []
        
        for s in range(nStimON):

            ############################
            # Get Tactile STIM
            # 3001 = leftON_analog, 
            # 4001 = leftOFF_analog
            # 3002 = rightON_analog, 
            # 4002 = rightOFF_analog
            if tactStimParams[s]['leftValid'] or tactStimParams[s]['rightValid']:

                if tactStimParams[s]['leftValid']:
                    leftON = stimTimeON[s] + (tactStimParams[s]['left']['Start Time']/1000) # leftShaker startTime
                    leftOFF = stimTimeON[s] + (tactStimParams[s]['left']['Start Time']/1000) + (tactStimParams[s]['left']['Duration']/1000) # leftShaker endTime
                else:
                    leftON = expStartTime + defaultNoTime
                    leftOFF = expStartTime + defaultNoTime

                if leftAnalog and tactStimParams[s]['leftValid']:
                    leftON_analog = leftTimeON_analog[numpy.abs(numpy.asarray(leftTimeON_analog)-leftON).argmin()] # ClosestMarker3001 
                    leftOFF_analog = leftTimeOFF_analog[numpy.abs(numpy.asarray(leftTimeOFF_analog)-leftOFF).argmin()] # ClosestMarker3002
                    if abs(leftON_analog-leftON)>toleranceTactileStim:
                        print('marker 3001 (Tactile Left-shaker ON) was not found closer to the expected time by the configFile. \n\
                            closer marker3001: {},\n\
                            configFile leftON: {},\n\
                            absDifference (secs): {},\n\
                            Rep: {}, TrialID: {}, TrialNum: {}\n\
                            Tolerance (secs): {}'.format(leftON_analog, leftON, 
                                                         abs(leftON_analog-leftON),
                                                         trial['repID'], trial['ID'], trial['trialNum'], toleranceTactileStim))
                else:
                    leftON_analog = expStartTime + defaultNoTime
                    leftOFF_analog = expStartTime + defaultNoTime

                if tactStimParams[s]['rightValid']:
                    rightON = stimTimeON[s] + (tactStimParams[s]['right']['Start Time']/1000) # rightShaker startTime
                    rightOFF = stimTimeON[s] + (tactStimParams[s]['right']['Start Time']/1000) + (tactStimParams[s]['right']['Duration']/1000) # rightShaker endTime
                else:
                    rightON = expStartTime + defaultNoTime
                    rightOFF = expStartTime + defaultNoTime

                if rightAnalog and tactStimParams[s]['rightValid']:
                    rightON_analog = rightTimeON_analog[numpy.abs(numpy.asarray(rightTimeON_analog)-rightON).argmin()] # ClosestMarker4001 
                    rightOFF_analog = rightTimeOFF_analog[numpy.abs(numpy.asarray(rightTimeOFF_analog)-rightOFF).argmin()] # ClosestMarker4002 
                    if abs(rightON_analog-rightON)>toleranceTactileStim:
                        print('marker 4001 (Tactile Right-shaker ON) was not found closer to the expected time by the configFile. \n\
                            closer marker4001: {},\n\
                            configFile rightON: {},\n\
                            absDifference (secs): {},\n\
                            Rep: {}, TrialID: {}, TrialNum: {}\n\
                            Tolerance (secs): {}'.format(rightON_analog, rightON, 
                                                         abs(rightON_analog-rightON),
                                                         trial['repID'], trial['ID'], trial['trialNum'], toleranceTactileStim))
                else:
                    rightON_analog = expStartTime + defaultNoTime
                    rightOFF_analog = expStartTime + defaultNoTime

                stim_tactile.append({
                        'left': {**tactStimParams[s]['left'], 
                                 **{'on':leftON, 'off': leftOFF, 
                                    'on_analog': leftON_analog,
                                    'off_analog': leftOFF_analog,
                                    }},
                        'right': {**tactStimParams[s]['right'], 
                                  **{'on':rightON, 'off':rightOFF,
                                    'on_analog': rightON_analog,
                                    'off_analog': rightOFF_analog,
                                    }}
                        })  
                stimStop = max([leftOFF, leftOFF_analog, rightOFF, rightOFF_analog]) 
            else:
                stimStop = stimTimeON[s]
            
            ############################
            # Get StimEND
            if s>=nStimEND:
                #replace the marker by the end of Tactile Params
                stimTimeOFF.append(stimStop)
            else:
                stimTimeOFF.append(stimTimeEND[s])  

            ################################################################
            # Get MicroStim
            if microStimParams[s]['valid']:
                    
                microStimON = stimTimeON[s] + microStimParams[s]['microStim']['StartTime']

                # Use Marker 91 for end of MicroStim if it is within the timeTolerance
                microStimEnd = stimTimeON[s] + microStimParams[s]['microStim']['StartTime'] + sum(microStimParams[s]['microStim']['Duration'])           
                    
                if nMicroStimOFF>0 and (microStimEnd-toleranceMicroStim)<trial['outcomeTime']:
                    microStimOFF = microStimTimeOFF[numpy.abs(numpy.asarray(microStimTimeOFF)-microStimEnd).argmin()]
                    if abs(microStimOFF-microStimEnd)>toleranceMicroStim:
                        print('marker 91 (microStimEnd) was not found closer to the expected time by the configFile. \n\
                            closer marker91: {},\n\
                            configFile MicroStimEnd: {},\n\
                            absDifference (secs): {},\n\
                            Rep: {}, TrialID: {}, TrialNum: {}\n\
                            Tolerance (secs): {}'.format(microStimOFF, microStimEnd, 
                                                         abs(microStimOFF-microStimEnd),
                                                         trial['repID'], trial['ID'], trial['trialNum'], toleranceMicroStim))
                else:
                    microStimOFF = microStimEnd
                    
                stim_microStim.append({
                    **microStimParams[s]['microStim'], 
                    **{
                    'on':microStimON, 
                    'off':microStimOFF, 
                    'on_analog': min([microStimParams[s]['microStim']['ChannelStart_time'][c] for c in range(len(microStimParams[s]['microStim']['Channel'])) if microStimParams[s]['microStim']['Channel'][c]>0]),
                    'off_analog': max(microStimParams[s]['microStim']['ChannelStop_time']),
                    }
                })

            ##########################################################################
            # Get Visual Cue, If there is a Visual Cue ID, regarless of the duration, 
            # it will create a marker=15
            if visualStimParams[s]['visualStim']['nStim']:

                # Check if VisualCue started (abort/experimentStopped)
                # Use Marker 15 for Start of VisualCueStim if it is within the timeTolerance
                visualCueStart = stimTimeON[s] + (visualStimParams[s]['visualStim']['StartTime']/1000)

                if nVisON>0:

                    visStimON = visTimeON[numpy.abs(numpy.asarray(visTimeON)-visualCueStart).argmin()]
                    
                    if abs(visStimON-visualCueStart)>toleranceVisualCue:
                        print('marker 15 (VisualCue start) was not found closer to the expected time by the configFile. \n\
                            closer marker15: {},\n\
                            configFile VisualCueStartTime: {},\n\
                            absDifference (secs): {},\n\
                            Rep: {}, TrialID: {}, TrialNum: {}\n\
                            Tolerance (secs): {}'.format(visStimON, visualCueStart,
                                                         abs(visStimON-visualCueStart),
                                                         trial['repID'], trial['ID'], trial['trialNum'], toleranceVisualCue))
                        
                    visualCueDuration = visualStimParams[s]['visualStim']['Duration']/1000
                    
                    visualCueEnd = visStimON + visualCueDuration
                    endVisual = [visualCueEnd, trial['outcomeTime']]

                    # Check if Visual ends with choiceTarget ON
                    if visualCuesInfo['visualENDwith']=='choiceTargetON' and nChoiceShownTrial>0:
                        endVisual.append(choiceTargetON)
                    # Check if Visual ends with Fixation
                    if visualCuesInfo['visualENDwith']=='fixationOFF' and fix_FixTarget_End>0:
                        endVisual.append(fix_FixTarget_End)

                    visStimOFF = min(endVisual)

                    # Check for ANALOG TimeStamps
                    if nVisON_analog>0:

                        visStimON_analog = visTimeON_analog[numpy.abs(numpy.asarray(visTimeON_analog)-visStimON).argmin()]
                        if abs(visStimON_analog-visStimON)>tolerancePhotodiode:
                            print('marker 1500 (VisualCue start) was not found closer to the expected time by the configFile.\
                                \ncloser marker1500: {},\
                                \nconfigFile VisualCueStartTime: {},\
                                \nabsDifference (secs): {},\
                                \nRep: {}, TrialID: {}, TrialNum: {}\
                                \nTolerance (secs): {}'.format(visStimON_analog, visStimON,
                                                         abs(visStimON_analog-visStimON),
                                                         trial['repID'], trial['ID'], trial['trialNum'], tolerancePhotodiode))

                        visualCueEnd_analog = visStimON_analog + visualCueDuration

                        endVisual_analog = [visualCueEnd_analog, trial['outcomeTime']]

                        # Check if Visual ends with choiceTarget ON
                        if visualCuesInfo['visualENDwith']=='choiceTargetON' and nChoiceShownTrial>0:
                            if existChoiceTargetON_analog:
                                endVisual_analog.append(choiceTargetON_analog)                                
                            else:
                                endVisual_analog.append(choiceTargetON)
                        # Check if Visual ends with Fixation
                        if visualCuesInfo['visualENDwith']=='fixationOFF' and fix_FixTarget_End>0:
                            if existfixTimeOFF_analog:
                                endVisual_analog.append(fixTimeOFF_analog)
                            else:
                                endVisual_analog.append(fix_FixTarget_End)
                        # # Check if Analog Reward was recorded (Hit Trials with VisualCues ON the screen the entire trial)
                        # if existRewardStart_analog:
                        #     endVisual_analog.append(rewardStart_analog)

                        visStimOFF_analog = min(endVisual_analog)
                    else:
                        visStimON_analog = expStartTime + defaultNoTime
                        visStimOFF_analog = expStartTime + defaultNoTime

                    stim_Visual.append({
                        'nStim': visualStimParams[s]['visualStim']['nStim'], 
                        'ID': sorted(visualStimParams[s]['visualStim']['ID']),
                        'on': visStimON,
                        'off': visStimOFF,
                        'on_analog': visStimON_analog,
                        'off_analog': visStimOFF_analog,
                    })

        
        ##########################################################################################
        #                       Construct trialRow dictionary
        ##########################################################################################
        # All data should match "float" type already, but it was added again to be 100% sure
        
        trialRow = {
            'start_time': float(markerTime[markerID.index(1)]-expStartTime),
            'stop_time': float(stop_Time-expStartTime),
            'roundID': trial['repID'],
            'trialID': trial['ID'],
            'repeatedTrial': trial['repTrial'],
            'outcomeID': trial['outcomeID'],
            'outcomeDescription': trial['outcomeLabel'],
            'outcome_time': float(trial['outcomeTime']-expStartTime),
            'fixTarget_OFF_time': float(fix_FixTarget_End-expStartTime),
            'fixTarget_ON_analog_time': float(fixTimeON_analog-expStartTime),
            'fixTarget_OFF_analog_time': float(fixTimeOFF_analog-expStartTime),
            'fix_FixTarget_time': float(fix_FixTarget_Start-expStartTime),
            'fixMode': fixMode,
            'fixTarget_Window': fixTargetInfo['fixTarget_Window'],
            'fixTarget_Shape': fixTargetInfo['fixTarget_Shape'],
            'fixTarget_Size': fixTargetInfo['fixTarget_Size'],
            'fixTarget_RGBA': fixTargetInfo['fixTarget_RGBA'],
            'fixTarget_Position': fixTargetInfo['fixTarget_Position'],    

        }

        # Stimulation Events
        if maxStimTypes['nStim']>0:
            trialRow['nStim'] = nStimON
            for s in range(maxStimTypes['nStim']):
                if s>=nStimON:
                    trialRow.update({
                        'stim'+str(s+1)+'_ON_time': defaultNoTime,
                        'stim'+str(s+1)+'_OFF_time': defaultNoTime
                    })
                else:
                    trialRow.update({
                        'stim'+str(s+1)+'_ON_time': float(stimTimeON[s]-expStartTime),
                        'stim'+str(s+1)+'_OFF_time': float(stimTimeOFF[s]-expStartTime)
                    })

        # Tactile Stimulation
        if maxStimTypes['nTactileStim']>0:
            nTactile = len(stim_tactile)
            trialRow['nTactile'] = nTactile
            # Append TACTILE Stim
            for s in range(maxStimTypes['nTactileStim']):
                # In case this trial has less Stim than the MAX-Tactile then ADD Default Values
                if s>=nTactile:
                    trialRow.update( {
                        'tact'+str(s+1)+'_Left_ON_time': defaultNoTime,
                        'tact'+str(s+1)+'_Left_OFF_time': defaultNoTime,
                        'tact'+str(s+1)+'_Left_ON_analog_time': defaultNoTime,
                        'tact'+str(s+1)+'_Left_OFF_analog_time': defaultNoTime,
                        'tact'+str(s+1)+'_Left_FR': defaultNoStimParam,
                        'tact'+str(s+1)+'_Left_AMP': defaultNoStimParam,
                        'tact'+str(s+1)+'_Left_DUR': defaultNoStimParam,
                        'tact'+str(s+1)+'_Right_ON_time': defaultNoTime,
                        'tact'+str(s+1)+'_Right_OFF_time': defaultNoTime,
                        'tact'+str(s+1)+'_Right_ON_analog_time': defaultNoTime,
                        'tact'+str(s+1)+'_Right_OFF_analog_time': defaultNoTime,
                        'tact'+str(s+1)+'_Right_FR': defaultNoStimParam,
                        'tact'+str(s+1)+'_Right_AMP': defaultNoStimParam,
                        'tact'+str(s+1)+'_Right_DUR': defaultNoStimParam,
                        })
                else:
                    trialRow.update( {
                        'tact'+str(s+1)+'_Left_ON_time': float(stim_tactile[s]['left']['on']-expStartTime),
                        'tact'+str(s+1)+'_Left_OFF_time': float(stim_tactile[s]['left']['off']-expStartTime),
                        'tact'+str(s+1)+'_Left_ON_analog_time': float(stim_tactile[s]['left']['on_analog']-expStartTime),
                        'tact'+str(s+1)+'_Left_OFF_analog_time': float(stim_tactile[s]['left']['off_analog']-expStartTime),
                        'tact'+str(s+1)+'_Left_FR': float(stim_tactile[s]['left']['Freq']),
                        'tact'+str(s+1)+'_Left_AMP': float(stim_tactile[s]['left']['Amp']),
                        'tact'+str(s+1)+'_Left_DUR': float(stim_tactile[s]['left']['Duration']/1000),
                        'tact'+str(s+1)+'_Right_ON_time': float(stim_tactile[s]['right']['on']-expStartTime),
                        'tact'+str(s+1)+'_Right_OFF_time': float(stim_tactile[s]['right']['off']-expStartTime),
                        'tact'+str(s+1)+'_Right_ON_analog_time': float(stim_tactile[s]['right']['on_analog']-expStartTime),
                        'tact'+str(s+1)+'_Right_OFF_analog_time': float(stim_tactile[s]['right']['off_analog']-expStartTime),
                        'tact'+str(s+1)+'_Right_FR': float(stim_tactile[s]['right']['Freq']),
                        'tact'+str(s+1)+'_Right_AMP': float(stim_tactile[s]['right']['Amp']),
                        'tact'+str(s+1)+'_Right_DUR': float(stim_tactile[s]['right']['Duration']/1000),
                        })

        # Information about tactile Placement  
        trialRow.update({
            'leftPlacement' : shakerInfo['leftBodyPart'] + ' - ' + shakerInfo['leftSegment'],
            'leftIndentation': float(shakerInfo['leftIndentation']),
            'rightPlacement' : shakerInfo['rightBodyPart'] + ' - ' + shakerInfo['rightSegment'],
            'rightIndentation': float(shakerInfo['rightIndentation']),
        })

        if 'trialTemp' in trial:
            trialRow.update({
                'trialTemp': trial['trialTemp']
            })


        # Microstimulation
        if maxStimTypes['nMicroStim']>0:
            nMicroStim = len(stim_microStim)
            trialRow['nMicroStim'] = nMicroStim
            # Append MicroStim
            for s in range(maxStimTypes['nMicroStim']):
                if s>=nMicroStim:
                    trialRow.update( {
                        'XIPP_Stim'+ str(s+1)+ '_ON_time': defaultNoTime,
                        'XIPP_Stim'+ str(s+1)+ '_OFF_time': defaultNoTime,
                        'XIPP_Stim'+ str(s+1)+ '_ON_analog_time': defaultNoTime,
                        'XIPP_Stim'+ str(s+1)+ '_OFF_analog_time': defaultNoTime,
                        'XIPP_Stim'+ str(s+1)+ '_Channel': [int(-1), int(-1), int(-1), int(-1)],
                        'XIPP_Stim'+ str(s+1)+ '_ChannelStart_time': [defaultNoTime, defaultNoTime, defaultNoTime, defaultNoTime],
                        'XIPP_Stim'+ str(s+1)+ '_ChannelStop_time': [defaultNoTime, defaultNoTime, defaultNoTime, defaultNoTime],
                        'XIPP_Stim'+ str(s+1)+ '_ReturnChannel': int(-1),
                        'XIPP_Stim'+ str(s+1)+ '_Duration': [defaultNoStimParam, defaultNoStimParam, defaultNoStimParam],
                        'XIPP_Stim'+ str(s+1)+ '_Frequency': [defaultNoStimParam, defaultNoStimParam, defaultNoStimParam],
                        'XIPP_Stim'+ str(s+1)+ '_InterphaseInterval': [defaultNoStimParam, defaultNoStimParam, defaultNoStimParam],
                        'XIPP_Stim'+ str(s+1)+ '_Phase1_Width': [defaultNoStimParam, defaultNoStimParam, defaultNoStimParam],
                        'XIPP_Stim'+ str(s+1)+ '_Phase1_Amp': [defaultNoStimParam, defaultNoStimParam, defaultNoStimParam], 
                        'XIPP_Stim'+ str(s+1)+ '_Phase2_Width': [defaultNoStimParam, defaultNoStimParam, defaultNoStimParam],
                        'XIPP_Stim'+ str(s+1)+ '_Phase2_Amp': [defaultNoStimParam, defaultNoStimParam, defaultNoStimParam], 
                    })
                else:
                    trialRow.update( {
                        'XIPP_Stim'+ str(s+1)+ '_ON_time': float(stim_microStim[s]['on']-expStartTime),
                        'XIPP_Stim'+ str(s+1)+ '_OFF_time': float(stim_microStim[s]['off']-expStartTime),
                        'XIPP_Stim'+ str(s+1)+ '_ON_analog_time': float(stim_microStim[s]['on_analog']-expStartTime),
                        'XIPP_Stim'+ str(s+1)+ '_OFF_analog_time': float(stim_microStim[s]['off_analog']-expStartTime),
                        'XIPP_Stim'+ str(s+1)+ '_Channel': stim_microStim[s]['Channel'],
                        'XIPP_Stim'+ str(s+1)+ '_ChannelStart_time': [ts-expStartTime for ts in stim_microStim[s]['ChannelStart_time']],
                        'XIPP_Stim'+ str(s+1)+ '_ChannelStop_time': [ts-expStartTime for ts in stim_microStim[s]['ChannelStop_time']],
                        'XIPP_Stim'+ str(s+1)+ '_ReturnChannel': stim_microStim[s]['ReturnChannel'],
                        'XIPP_Stim'+ str(s+1)+ '_Duration': stim_microStim[s]['Duration'],
                        'XIPP_Stim'+ str(s+1)+ '_Frequency': stim_microStim[s]['Frequency'],
                        'XIPP_Stim'+ str(s+1)+ '_InterphaseInterval': stim_microStim[s]['InterphaseInterval'],
                        'XIPP_Stim'+ str(s+1)+ '_Phase1_Width': stim_microStim[s]['Phase1_Width'],
                        'XIPP_Stim'+ str(s+1)+ '_Phase1_Amp': stim_microStim[s]['Phase1_Amp'],
                        'XIPP_Stim'+ str(s+1)+ '_Phase2_Width': stim_microStim[s]['Phase2_Width'],
                        'XIPP_Stim'+ str(s+1)+ '_Phase2_Amp': stim_microStim[s]['Phase2_Amp'],
                    })

        # Visual Stimulation
        if maxStimTypes['nVisualStim']>0:
            nVisual = len(stim_Visual)                        
            countV = 0
            trialRow['nVisual'] = 0
            for s in range(nVisual):
                currentVisStimON = stim_Visual[s]['on']-expStartTime
                currentVisStimOFF = stim_Visual[s]['off']-expStartTime
                currentVisStimON_analog = stim_Visual[s]['on_analog']-expStartTime
                currentVisStimOFF_analog = stim_Visual[s]['off_analog']-expStartTime
                for v in range(stim_Visual[s]['nStim']):
                    indexVisual = visualCuesInfo['ID'].index(stim_Visual[s]['ID'][v])
                    trialRow.update( {
                        'vis'+str(countV+1)+'_ON_time': float(currentVisStimON),
                        'vis'+str(countV+1)+'_OFF_time': float(currentVisStimOFF),
                        'vis'+str(countV+1)+'_ON_analog_time': float(currentVisStimON_analog),
                        'vis'+str(countV+1)+'_OFF_analog_time': float(currentVisStimOFF_analog),
                        'vis'+str(countV+1)+'_ID': stim_Visual[s]['ID'][v],
                        'vis'+str(countV+1)+'_Shape': visualCuesInfo['Shape'][indexVisual],
                        'vis'+str(countV+1)+'_Size': visualCuesInfo['Size'][indexVisual],
                        'vis'+str(countV+1)+'_RGBA': visualCuesInfo['RGBA'][indexVisual],
                        'vis'+str(countV+1)+'_Position': visualCuesInfo['Position'][indexVisual],
                    })
                    countV += 1

            trialRow['nVisual'] = countV
                        
            # Fill the remaining visuals with default values
            for s in range(countV, maxStimTypes['nVisualStim']):
                trialRow.update( {
                    'vis'+str(s+1)+'_ON_time': defaultNoTime,
                    'vis'+str(s+1)+'_OFF_time': defaultNoTime,
                    'vis'+str(s+1)+'_ON_analog_time': defaultNoTime,
                    'vis'+str(s+1)+'_OFF_analog_time': defaultNoTime,
                    'vis'+str(s+1)+'_ID': int(0),
                    'vis'+str(s+1)+'_Shape': 'None',
                    'vis'+str(s+1)+'_Size': [defaultNoStimParam, defaultNoStimParam],
                    'vis'+str(s+1)+'_RGBA': [defaultNoStimParam, defaultNoStimParam, defaultNoStimParam, defaultNoStimParam],
                    'vis'+str(s+1)+'_Position': [defaultNoStimParam, defaultNoStimParam],
                })

        # Go-cue LABELS
        trialRow.update({
            'responseMode': responseMode,
        })

        # ChoiceTargetsINFO
        if responseMode != 'noResponse':
            trialRow.update({
                'choiceTarget_ON_time': float(choiceTargetON-expStartTime),
                'choiceTarget_OFF_time': float(choiceTargetOFF-expStartTime),
                'choiceTarget_ON_analog_time': float(choiceTargetON_analog-expStartTime),
                'choiceTarget_OFF_analog_time': float(choiceTargetOFF_analog-expStartTime),
                'correctChoiceTarget_ID': correctChoiceTarget_ID,
                'selectedChoiceTarget_ID': selectedChoiceTarget_ID,
                'showingCorrectChoiceTarget': showingCorrectChoiceTarget,
                'nChoiceTargets': nChoiceShownTrial,
                'choiceTargets_Window':  choiceTargetInfo['ChoiceTargets_Window']
            })

            for s in range(nChoiceShownTrial):
                indexChoice = choiceTargetInfo['ID'].index(choiceTargetsList[s])
                trialRow.update({
                    'choice'+str(s+1)+'_ID': int(choiceTargetsList[s]),
                    'choice'+str(s+1)+'_Shape': choiceTargetInfo['Shape'][indexChoice],
                    'choice'+str(s+1)+'_Size': choiceTargetInfo['Size'][indexChoice],
                    'choice'+str(s+1)+'_RGBA': choiceTargetInfo['RGBA'][indexChoice],
                    'choice'+str(s+1)+'_Position': choiceTargetInfo['Position'][indexChoice],
                })

            # Fill the remaining ChoiceTargets with default values
            for s in range(nChoiceShownTrial, maxStimTypes['nChoiceTargetsShown']):
                trialRow.update({
                    'choice'+str(s+1)+'_ID': int(0),
                    'choice'+str(s+1)+'_Shape': 'None',
                    'choice'+str(s+1)+'_Size': [defaultNoStimParam, defaultNoStimParam],
                    'choice'+str(s+1)+'_RGBA': [defaultNoStimParam, defaultNoStimParam, defaultNoStimParam, defaultNoStimParam],
                    'choice'+str(s+1)+'_Position': [defaultNoStimParam, defaultNoStimParam],
                })
                    
        # Movement-Response Markers
        trialRow.update({
            'responseStart_time': float(responseMovStart-expStartTime),
            'fix_ChoiceTarget_time': float(fixChoiceTargetStart-expStartTime),
            'rewardStart_time': float(rewardStart-expStartTime),
            'rewardType': rewardType,
            'trialEnd_time': float(trialEnd-expStartTime),
        })

        return trialRow