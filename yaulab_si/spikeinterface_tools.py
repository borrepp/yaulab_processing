from ..yaulab_extras import supported_probes_manufacturer, get_tempdir, clear_tempdir

from spikeinterface.core import concatenate_recordings, order_channels_by_depth, get_random_data_chunks, get_noise_levels
from spikeinterface.extractors import read_nwb
from spikeinterface.preprocessing import common_reference as spre_common_reference

from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks

from probeinterface.plotting import plot_probe
from probeinterface import Probe

from pynwb import NWBHDF5IO

import matplotlib.pyplot as plt

import os
import shutil
import numpy


####################################################################################################################################################################################
#                                                                    SOME DEFAULT PARAMETERS
####################################################################################################################################################################################

n_cpus = os.cpu_count()
n_jobs = n_cpus - 2

job_kwargs = dict(chunk_duration="1s", n_jobs=n_jobs, progress_bar=True)

# si.set_global_job_kwargs(**job_kwargs)

######################################################################################################
# GENERAL PARAMETER FOR "PEAKS LOCATIONS"
ms_before_default = 0.6
ms_after_default = 1.5
peak_sign_default = 'both' # (“neg” | “pos” | “both”)
nearest_chans_default = 3 # Number of neighbor channels to search for the same waveform

###########################################################
# DETECT PEAKS: 
peak_detect_threshold_default = 5 #  MAD: Median Amplitude Deviations

######################################################################################################
# Figures size when is saved
fig_width_global = 12 # inches
fig_height_global = 8 # inches
fig_dpi = 120


####################################################################################################################################################################################
####################################################################################################################################################################################
#                                                                   MAIN FUNCTIONS
####################################################################################################################################################################################
####################################################################################################################################################################################

##############################################################################################################
# GET GENERAL INFORMATION ABOUT THE PROBE FROM A GIVEN ELECTRODE GROUP
##############################################################################################################
def getProbeInfo(electrodeGroup_dataFrame, nwbFile):

    groupName_u = electrodeGroup_dataFrame.group_name.unique()
    # Double check if these electrodes were recorded with the same probe:
    if groupName_u.size>1:
        raise Exception('ElectrodeGroup should have a unique name, but more than one name was found {}'.format(
           groupName_u 
        ))
    
    groupName = groupName_u[0]
    nameDevice = nwbFile.electrode_groups[groupName].device.name.upper()
    group_names = electrodeGroup_dataFrame.group_name.to_numpy().astype("str", copy=False)

    if 'PLX' in nameDevice:
        # PLX-UP-16-15SE(150)
        # [0:3] = PLX prefix
        # [4:6] = Tip Profile 
        # [7:9] = Number of Channels
        # [10:12] = electrode diameter in microns
        # [12:14] = Electrode configuration (SE=single, ST=stereotrode, TT=tetrode)
        # [15:-1] = electrode spacing e.g.: 15/100 intra/inter spacing
        diameterElec = int(nameDevice[10:12])/1000

        num_shanks = 1
        shank_ids = numpy.zeros(int(nameDevice[7:9])).astype(int)

    elif 'FHC' in nameDevice:
        # FHC-UE1234567890
        # UE = metal microelectrode
        # 1 = Material 
        # 2= lenght range
        # 3= shank diameter (microns)
        FHC_diameter = {
            'C': 75,
            'D': 100,
            'E': 125,
            'F': 200,
            'G': 250,
            'H': 500,
            'X': 'special'
            }
        
        diameterElec = FHC_diameter[nameDevice[8]]
        if type(diameterElec)==str:
            diameterElec = 50
        
        diameterElec = diameterElec/1000

        num_shanks = 1
        shank_ids = numpy.zeros(1).astype(int)

    else:
        print('WARNING¡¡¡\nDevice name "{}" was not recognized\nelectrodeProbes should start with:\n{}'.format(nameDevice, supported_probes_manufacturer))
        diameterElec = 50
        num_shanks = 0
        shank_ids = numpy.zeros(1).astype(int)

    return {'probeName': nameDevice, 'radius': diameterElec, 'radius_units': 'mm', 'num_shanks': num_shanks, 'shank_ids': shank_ids, 'group_names': group_names}

########################################################################################################################
# CREATE A SPIKEINTERFACE-PROBE USING ELECTRODE-GROUP INFORMATION
########################################################################################################################
def constructProbe_2d(si_recordingObj_nwb, probeInfo_dict, showPlot=False):
    
    """
    It will construct a 2D probe (probeInterface) using:
        From NWB : locations, re_id, frontEnd_electrode_id
        From probeInfo: manufacturer, radius, 

    Coordinates extracted from NWB:

        x = AnteriorPosterior (ap) ZERO = Bregma (+x is posterior)
        y = DorsalVentral (dv)  ZERO = top of the brain (+y is inferior)
        z = MedialLateral (ml) ZERO = middline (+z is right)

        rel_x = x-axis (width) locations of the contacts on the Probe
        rel_y = y-axis (length) locations of the contacts on the Probe
        rel_z = z-axis (zero-values) 
    
    When loading an NWB into a sipkeinterface RecordingObject, relative coordinates (rel_x, rel_y, rel_z)
    are used to create the "property" "location". 
    Using the "location" will provide the contact positions in the "probe"-coordinate system
    Probe plane will be X (ap), Y(dv) coordinates


    """

    # Check the electrode/probe is supported by this version:
    #   PLEXON-SE(single shank with multiple electrode/contacts)
    #   FHC-UE (single electrode)
    probeExist = any([p in probeInfo_dict['probeName'] for p in supported_probes_manufacturer])
    if not probeExist:
        raise Exception(
            'Device name was not recognized:{}\
            \nCurrent electrodeProbes should start with: {}'.format
            (probeInfo_dict['probeName'], 
            supported_probes_manufacturer)
            )
    
    # GET CONTACTS_IDs: 
    # These correspond to the labels of the contacts on the probe (defined by Probe-manufacturer)
    # In the YAULAB's NWB-scheme. The probe-contactID corresponds to the property "rel_id"
    # Contacts ids are converted to strings. Contact ids must be **unique** for the **Probe** and also for the **ProbeGroup**
    contact_ids_nwb = si_recordingObj_nwb.get_property('rel_id')


    # GET DEVICE CHANNEL INDICES:
    # Correspond to the channel indexes in the recording system (i.e., Ripple)
    # The connections between contact_ids (Probe) with the recording device depends on the Headstage manufacturer (i.e., Ripple)
    # In the YAULAB's NWB-scheme. The device-channel ID corresponds to the property "frontEnd_electrode_id"-1 (zero index for pyhton)
    # CHECK TODO
    # WARNING¡¡¡¡
    # For multiple contact probes, it will assume that no more than one electrodeProbe was connected in a the same fronEnd
    if si_recordingObj_nwb.get_num_channels()>1:
        channel_indices_nwb = si_recordingObj_nwb.get_property('frontEnd_electrode_id') -1
    else:
    # For single contact electrodes (e.g., FHC), force the channel_index to ZERO
        channel_indices_nwb = numpy.array([0])

    # GET LOCATIONS IN "um"
    # Check if the recording Object has the location_units defined
    if 'location_units' in si_recordingObj_nwb.get_annotation_keys():
        if si_recordingObj_nwb.get_annotation('location_units') == 'mm':
            locationsProbe_nwb = si_recordingObj_nwb.get_property('location')*1000 
        elif si_recordingObj_nwb.get_annotation('location_units') == 'um':
            locationsProbe_nwb = si_recordingObj_nwb.get_property('location')
        else:
            raise Exception('Units of the Locations in the RecordingObject is not recognized: {}'.format(
                si_recordingObj_nwb.get_annotation('location_units') 
            ))
    else:
    # If not Assume units in 'mm' (default units from NWB format)
        locationsProbe_nwb = si_recordingObj_nwb.get_property('location')*1000  

    # Get Radius of the contacts
    if probeInfo_dict['radius_units']=='mm':
        radiusContact = probeInfo_dict['radius']*1000
    elif probeInfo_dict['radius_units']=='um':
        radiusContact = probeInfo_dict['radius']
    else:
        raise Exception('Units of the ElectrodeContact radius is not recognized: {}'.format(
           probeInfo_dict['radius_units'] 
        ))
    
    # PolygonProbe for a single linear probe.
    if probeInfo_dict['num_shanks']==1:
        marginPol = radiusContact*5
        # (x, y) coordinates of the points that will define the contour of the 2D probe
        #   
        #   E---D
        #   |   |
        #   A   C
        #    \ /
        #     B
        #
        polygonProbe2d = [
            (-marginPol, 0), # A
            (0, -marginPol), # B
            (marginPol, 0),  # C
            (marginPol, max(locationsProbe_nwb[:, 1])+marginPol), # D
            (-marginPol, max(locationsProbe_nwb[:, 1])+marginPol) # E
            ]
    else:
        raise Exception('Probe with {} shanks is not supported: {}'.format(
           probeInfo_dict['shankNum'] 
        ))
    
    probeObj = Probe(ndim=2, si_units='um', model_name=probeInfo_dict['probeName'])
    probeObj.set_contacts(
        positions=locationsProbe_nwb[:, 0:2], 
        contact_ids=contact_ids_nwb,
        shapes='square', 
        shape_params={'width':radiusContact},
    )
    probeObj.set_shank_ids(probeInfo_dict['shank_ids'])
    probeObj.set_planar_contour(contour_polygon=polygonProbe2d)
    probeObj.set_device_channel_indices(channel_indices=channel_indices_nwb)

    if showPlot:
        plot_probe(probe=probeObj, with_contact_id=True, with_device_index=True)
        plt.show()
            
    return probeObj



########################################################################################################################################################
# Set up NWB file Paths to be sorted. Sorting must be done for each raw-ElectrodeGroup 
# from all sessions where the probe(s) was not move (i.e., the same neurons were recorded). To choose those sessions, 
# it will search for NWB files with the same "expYEAR-MONTH-DAY" prefix and will check that coordinates and electrode 
# configuration match across sessions. If a session  has a mismatch on coordinates or electrode configuration, that 
# session will be named as -subX.
########################################################################################################################################################
def getUnique_electrodeGroups(parentRecordingFolder, sessionYear, sessionMonth, sessionDay):

    # Get filePaths of all NWB files recorded on the same Day
        
    filesDate = []

    for root, _, files in os.walk(parentRecordingFolder):

        for name in files:

            nameSplit = os.path.splitext(name)

            if nameSplit[1]=='.nwb' and '-noNEV' not in nameSplit[0]:

                fileName = nameSplit[0]
                yearFile = int(fileName[3:7])
                monthFile = int(fileName[8:10])
                dayFile = int(fileName[11:13])

                if yearFile==sessionYear and monthFile==sessionMonth and dayFile==sessionDay:

                    filesDate.append(os.path.join(root, name))

    # Check for electrodeGroups on each file
    elecGroups = {}

    for f in filesDate:

        # Open the file in read mode "r"
        io = NWBHDF5IO(f, mode="r")
        nwbfile = io.read()
        elecGroups_f = [k for k in nwbfile.acquisition.keys() if 'raw-' in k]

        fParts = os.path.split(f)

        for e  in elecGroups_f:
            electrodes_df = nwbfile.acquisition[e].electrodes.to_dataframe()
            probeInfo = getProbeInfo(electrodeGroup_dataFrame=electrodes_df, nwbFile = nwbfile)
            elecGroups_keys = list(elecGroups.keys())
            if elecGroups_keys.count(e)==1:
                elecGroups[e]['pathFiles'].append(f)
                elecGroups[e]['fileNames'].append(fParts[1])
                elecGroups[e]['electrodes'].append(electrodes_df)
                elecGroups[e]['probeInfo'].append(probeInfo)

            else:
                elecGroups[e] = {
                    'pathFiles': [f],
                    'fileNames': [fParts[1]],
                    'electrodes': [electrodes_df],
                    'probeInfo': [probeInfo],
                }
            
        io.close()

    uniqueElectrodesNames = []
    uniqueElectrodesPaths = []
    uniqueProbeInfo = []
    uniqueElectrodesFilePaths = []
    uniqueElectrodesFileNames = []

    for g_name, g in elecGroups.items():

        # Check that across files this ElectrodeGroup has the same:
        # coordinates, and electrode configuration
         
        nFiles = len(g['fileNames'])
        electrodesDF_unique = [g['electrodes'][0]]
        probes_unique = [g['probeInfo'][0]]
        filesMatch = [[0]]
        groupNames = [g_name]
        groupPath = [g_name]
        count = 1

        for t in range(1, nFiles):
            
            testDF = g['electrodes'][t]
            testProbe = g['probeInfo'][t]
            test_is_NEW = True

            n_u = len(electrodesDF_unique)

            for u in range(n_u):

                uDF = electrodesDF_unique[u]
                probe_equal = probes_unique[u]['probeName']==testProbe['probeName']

                x_equal = testDF['x'].equals(uDF['x'])
                y_equal = testDF['y'].equals(uDF['y'])
                z_equal = testDF['z'].equals(uDF['z'])
                port_equal = testDF['port_id'].equals(uDF['port_id'])
                FE_equal = testDF['frontEnd_id'].equals(uDF['frontEnd_id'])
                FEchannel_equal = testDF['frontEnd_electrode_id'].equals(uDF['frontEnd_electrode_id'])
                relID_equal = testDF['rel_id'].equals(uDF['rel_id'])
                
                if x_equal and y_equal and z_equal and port_equal and FE_equal and FEchannel_equal and relID_equal and probe_equal:
                    test_is_NEW = False
                    filesMatch[u].append(t)

            if test_is_NEW:
                electrodesDF_unique.append(testDF)
                probes_unique.append(testProbe)
                filesMatch.append([t])
                groupNames.append(g_name + '-sub' + str(count))
                groupPath.append(g_name)
                count +=1
        
        for u in range(len(groupNames)):
            
            uniqueElectrodesNames.append(groupNames[u])
            uniqueElectrodesPaths.append(groupPath[u])
            uniqueProbeInfo.append(probes_unique[u])
            nFiles_u = len(filesMatch[u])
            filePaths_u = []
            files_u = []

            for f_u in range(nFiles_u):
                filePaths_u.append(g['pathFiles'][filesMatch[u][f_u]])
                files_u.append(g['fileNames'][filesMatch[u][f_u]])

            uniqueElectrodesFilePaths.append(filePaths_u)
            uniqueElectrodesFileNames.append(files_u)
    
    electrodeGroups = []

    for g in range(len(uniqueElectrodesNames)):

        ########################################################################################################################
        # Ensure Files are sorted by time
        file_index = numpy.argsort(numpy.array([float(os.path.splitext(f)[0][14:20]) for f in uniqueElectrodesFileNames[g]]))

        fileNames_sorted = []
        filePaths_sorted = []
        fileSamples_sorted = []
        fileDuration_sorted = []

        for i in file_index:

            fileNames_sorted.append(uniqueElectrodesFileNames[g][i])
            fP = [fP for fP in uniqueElectrodesFilePaths[g] if uniqueElectrodesFileNames[g][i] in fP][0]
            filePaths_sorted.append(fP)

            # Open the file in read mode "r"
            io = NWBHDF5IO(fP, mode="r")
            nwbfile = io.read()

            nSamples = nwbfile.acquisition[uniqueElectrodesPaths[g]].data.shape[0] # first dimension is time
            fs = nwbfile.acquisition[uniqueElectrodesPaths[g]].rate # fSampling

            fileSamples_sorted.append(nSamples)
            fileDuration_sorted.append(nSamples/fs)

            io.close()

        ########################################################################################################################

        electrodeGroups.append({
            'electrodeName': uniqueElectrodesNames[g],
            'electrodePath': uniqueElectrodesPaths[g],
            'probeInfo': uniqueProbeInfo[g],
            'fileNames': fileNames_sorted,
            'fileNamePaths': filePaths_sorted,
            'fileSamples': numpy.array(fileSamples_sorted),
            'fileDurations': numpy.array(fileDuration_sorted)
        })

    return electrodeGroups



#####################################################################################################################################################
# PRINT INFORMATION ABOUT HOW MANY ELECTRODE GROUPS AND SESSIONS WERE FOUND FOR A GIVEN EXPERIMENTAL DAY
#####################################################################################################################################################
def print_recordingInfo(parentRecordingFolder, sessionYear, sessionMonth, sessionDay):

    electrodeGroups = getUnique_electrodeGroups(parentRecordingFolder, sessionYear, sessionMonth, sessionDay)

    # Electrode Group Preprocessing 
    nGroups = len(electrodeGroups)

    print('\nThere are {} electrode group(s)\n'.format(nGroups))

    # Print some information from each electrodeGroup
    for eg_ in electrodeGroups:

        print('ElectrodeGroup Name: "{}" (nwbPath: {})'.format(eg_['electrodeName'], 'acquisition/' + eg_['electrodePath']))
        
        print('\tProbe Info: ')
        for k, v, in eg_['probeInfo'].items():
            print('\t\t{} : {}'.format(k, v))
        
        print('\tNumber of sessions = {}\n\t\tSessionsInfo:'.format(len(eg_['fileNames'])))

        for n in range(len(eg_['fileNames'])):

            print('\t\t\tSessIndx: {} (SessName: {})'.format(n, eg_['fileNames'][n]))

            ######################################################################################################
            # Print some Recording information about the session:
            print('\t\t\tRecInfo: \n\t\t\t', 
                  read_nwb(
                    file_path=eg_['fileNamePaths'][n], 
                    electrical_series_path='acquisition/' + eg_['electrodePath'],
                    load_recording=True,
                    load_sorting=False,
                    ),
                '\n')
        print('\n')
 
####################################################################################################################################################################################
# Construct Dictionary with general information about the ElectrodeGroup & Sessions
####################################################################################################################################################################################
def select_electrodeGroup_and_session_info(electrodeGroups, sessionYear, sessionMonth, sessionDay, electrodeGroup_Name, session_index=None):
    
    #######################################################################################################################
    # Get selected ElectrodeGroup & Sessions
    #######################################################################################################################
    eg_i = [i for i in range(len(electrodeGroups)) if electrodeGroups[i]['electrodeName']==electrodeGroup_Name]

    if len(eg_i)==0:
        raise Exception("electrodeGroup Name : '{}' was not found\nAvailable electrodeGroups:\n\t{}".format(electrodeGroup_Name, [electrodeGroups[i]['electrodeName'] for i in range(len(electrodeGroups))]))
    else:
        # Check indexes to create sufix of the selected sessions
        sess_indexes = [n for n in range(len(electrodeGroups[eg_i[0]]['fileNames']))]
        consecutiveSessions = True
        if session_index is None:
            session_index_in = sess_indexes
        else:
            if len(session_index)==0:
                session_index_in = sess_indexes
            else:
                session_index_in = session_index 

            if len(session_index)>len(sess_indexes):
                raise Exception('session index of length = {} is out of bounds (nSession = {})\nSessions index available = {}'.format(len(session_index), len(sess_indexes), sess_indexes))
            elif len(session_index)>1:
                # Check all indexes are within boundary
                prev_i = 0
                start_i = True
                for i in session_index:
                    if i not in sess_indexes:
                        raise Exception('session index = {} is out of bounds\nSessions index available: {}'.format(i, sess_indexes))
                    if prev_i>i:
                        print('\n\nWARNING¡¡¡ Multiple Sessions selected, but indexes are not in consecutive order: {} should go before {}\nCheck your session-index = {}\nIt will concatenate non-consecutive recordings¡¡¡\n\n'.format(i, prev_i, session_index))
                        consecutiveSessions = False
                    if not start_i and (i-prev_i)>1:
                        print('\n\nWARNING¡¡¡ Multiple Sessions selected, but there is a gap between  session "{}" and "{}"\nCheck your session-index = {}\nIt will concatenate non-consecutive recordings¡¡¡\n\n'.format(prev_i, i, session_index))
                        consecutiveSessions = False

                    prev_i = i
                    start_i = False
                    
        if session_index_in[0]==session_index_in[-1]:
            consecutiveSessions = False

        sessID = 'exp{}-{:02d}-{:02d}'.format(sessionYear, sessionMonth, sessionDay) + '-sess'
        if consecutiveSessions:
            sessID += '-{}_{}'.format(session_index_in[0], session_index_in[-1])
        else:
            for i in session_index_in:
                sessID += '-' + str(i)

        electrodeGroup_sessions = {
            'electrodeName': electrodeGroups[eg_i[0]]['electrodeName'],
            'electrodePath': electrodeGroups[eg_i[0]]['electrodePath'],
            'probeInfo': electrodeGroups[eg_i[0]]['probeInfo'],
            'sessID': sessID,
            'fileNames': [electrodeGroups[eg_i[0]]['fileNames'][i] for i in session_index_in],
            'fileNamePaths': [electrodeGroups[eg_i[0]]['fileNamePaths'][i] for i in session_index_in],
            'fileNamePaths_read': [],
            'fileSamples': electrodeGroups[eg_i[0]]['fileSamples'][numpy.array(session_index_in)],
            'fileDurations': electrodeGroups[eg_i[0]]['fileDurations'][numpy.array(session_index_in)]
        }
    return electrodeGroup_sessions


####################################################################################################################################################################################
# Return SpikeInterface Recording Object for a given ElectrodeGroup & Sessions
####################################################################################################################################################################################
def get_si_recording(parentRecordingFolder, sessionYear, sessionMonth, sessionDay, electrodeGroup_Name, session_index=None, electrodeGroup_sessions=None, 
            localProcess_NWB=False, folder_temporal= None, reset_tempdir=True):
    
    if electrodeGroup_sessions is None:
        electrodeGroups = getUnique_electrodeGroups(parentRecordingFolder, sessionYear, sessionMonth, sessionDay)
        electrodeGroup_sessions = select_electrodeGroup_and_session_info(electrodeGroups, sessionYear, sessionMonth, sessionDay, electrodeGroup_Name, session_index=session_index)
    
    #######################################################################################################################
    # Copy files if need it & get updated FILE paths to read
    #######################################################################################################################
    # Check if the NWB files exists in the same Disk as this "code". Otherwise copy NWB-files to the same disk if localProcess_NWB=True
    keepOriginal_NWBpath = True
    if localProcess_NWB:
        for f in electrodeGroup_sessions['fileNamePaths']:
            filePath, _ = os.path.split(os.path.abspath(f))
            if filePath[0:2].lower()!=os.path.abspath(__file__)[0:2].lower():
                keepOriginal_NWBpath = False
    
    # If need it, copy NWBfiles. Update paths to read files
    for f in electrodeGroup_sessions['fileNamePaths']:
        fPath, fName = os.path.split(os.path.abspath(f))
        if not keepOriginal_NWBpath:
            # SET & clear temporal folder (it will be used to save peaks, peaks locations, and NWB files if localProcess_NWB=True)
            if folder_temporal is None:
                folder_temporal = get_tempdir(processName='SInwb')
            if reset_tempdir:
                clear_tempdir(folder_temporal)

            print('..... copying file {}'.format(fName))

            shutil.copy2(os.path.join(fPath, fName), os.path.join(folder_temporal, fName))
            electrodeGroup_sessions['fileNamePaths_read'].append(os.path.join(folder_temporal, fName))

        else:
            electrodeGroup_sessions['fileNamePaths_read'].append(os.path.join(fPath, fName))

    ###################################################
    # LOAD session(s)
    ###################################################
    if len(electrodeGroup_sessions['fileNamePaths_read'])>1:

        si_recording_List = []
        for f in electrodeGroup_sessions['fileNamePaths_read']:

            print('Concatenating sessions :\n', f, '\n')
            
            si_recording_List.append(read_nwb(
                file_path=f, 
                electrical_series_path='acquisition/' + electrodeGroup_sessions['electrodePath'],
                load_recording=True,
                load_sorting=False,
            ))
                        
        si_recording_raw = concatenate_recordings(si_recording_List)

    else:

        f = electrodeGroup_sessions['fileNamePaths_read'][0]

        print('loading session : ', f)

        si_recording_raw = read_nwb(
                file_path=f, 
                electrical_series_path='acquisition/' + electrodeGroup_sessions['electrodePath'],
                load_recording=True,
                load_sorting=False,
        )
    
    if not si_recording_raw.has_probe():

        print('Attaching probe to the recording ......')
        probe_from_nwb = constructProbe_2d(si_recordingObj_nwb = si_recording_raw, probeInfo_dict = electrodeGroup_sessions['probeInfo'])

        si_recording_rawProbe = si_recording_raw.set_probe(probe_from_nwb)
    else:
        si_recording_rawProbe = si_recording_raw
    
    del si_recording_raw

    ###################################################
    # Order channels by location
    ###################################################
    print('Sorting channels by depth ......')

    chan_ordered_index, chan_orig_index = order_channels_by_depth(recording=si_recording_rawProbe, dimensions=("x", "y"))
    si_recording = si_recording_rawProbe.channel_slice(channel_ids=[si_recording_rawProbe.channel_ids[i] for i in chan_ordered_index])
    si_recording.set_property(key='to_raw_index', values = chan_orig_index)

    del si_recording_rawProbe

    # CONFIRM LOCATIONS ARE IN "um"
    if si_recording.has_probe():
        units_probe = numpy.unique(si_recording._properties['contact_vector']['si_units'])
        if len(units_probe)>1:
            raise Exception('more than one type of units were found in the location of the contacts within the probe : {}', units_probe)
        else:
            si_recording.annotate(location_units=units_probe[0])

    # Check if the recording Object has the location_units defined
    if 'location_units' in si_recording.get_annotation_keys():
        if si_recording.get_annotation('location_units') == 'mm':
            si_recording._properties['location'] *=1000
            si_recording._annotations['location_units'] = 'um'

        
        if si_recording.get_annotation('location_units') != 'mm' and si_recording.get_annotation('location_units') != 'um':
            raise Exception('Units of the Locations in the RecordingObject is not recognized: {}'.format(
                si_recording.get_annotation('location_units') 
                ))
    else:
        # If not Assume units in 'mm' (default units from NWB format)
        si_recording._properties['location'] *=1000
        si_recording.annotate(location_units='um')

    #################################################################################################################
    # GET CHANNEL-CONTACT SPACING ("y" coordinate)
    contact_locations = si_recording.get_property('location')
    if len(contact_locations)>1:
        step_chan = numpy.mean(numpy.absolute(numpy.diff(contact_locations[:, 1])))
    else:
        step_chan = 50 # Default from SipkeInterface

    si_recording.annotate(y_contacts_distance_um = step_chan)

    #################################################################################################################
    # ADD CONCATENATION INFORMATION
    nSessions = len(electrodeGroup_sessions['fileNames'])
    nConcatenations = nSessions-1
    if nConcatenations>0:
        concatenationSamples = numpy.array(electrodeGroup_sessions['fileSamples'][:-1]).cumsum()
        concatenationTimes = concatenationSamples/si_recording.sampling_frequency
    else:
        concatenationSamples = numpy.array([])
        concatenationTimes = numpy.array([])
    
    si_recording.annotate(nSessions = nSessions)
    si_recording.annotate(sessionSamples = numpy.array(electrodeGroup_sessions['fileSamples']))
    si_recording.annotate(nConcatenations = nConcatenations)
    si_recording.annotate(concatenationSamples = concatenationSamples)
    si_recording.annotate(concatenationTimes = concatenationTimes)
    si_recording.annotate(elecGroupSessName = '{}-{}'.format(electrodeGroup_sessions['sessID'], electrodeGroup_sessions['electrodeName']))
    si_recording.annotate(recording_sufix = '')
    si_recording.annotate(fileNamePaths = electrodeGroup_sessions['fileNamePaths'])

    return si_recording, electrodeGroup_sessions, keepOriginal_NWBpath


#########################################################################################################################################
# PLOT TO INSPECT THE RECORDING AROUND CONCATENATION OF DIFFERENT SESSIONS
#########################################################################################################################################
def plot_concatenations(si_recording_dict, plot_windows_secs=0.01, sampleChans=False, showPlots=True, savePlots=False, folderPlots=None):

    recNames = list(si_recording_dict.keys())
    nRecs = len(recNames)
    si_recording = si_recording_dict[recNames[0]]

    nConcatenations = si_recording.get_annotation('nConcatenations')
    samples_radius = int(si_recording.sampling_frequency*plot_windows_secs)
    median_recs = int(numpy.median(numpy.arange(nRecs)))
    color_line_concat = (0.25, 0.25, 0.25, 1)

    if nConcatenations>0:

        if folderPlots is None:
            if 'savePath' in si_recording._annotations:
                preproPlots_folder = os.path.join(si_recording.get_annotation('savePath'), 'preproPlots')
            else:
                savePlots = False
                preproPlots_folder = None
                showPlots = True
                print('Concatenation Plots will not be saved')
        else:
            preproPlots_folder = folderPlots

        # maxChans = 9
        concatenationSamples = si_recording.get_annotation('concatenationSamples')
        concatenationTimes = si_recording.get_annotation('concatenationTimes')
        nChans = si_recording.get_num_channels()
        if nChans>9 and sampleChans:
            chans_range = numpy.arange(nChans)
            median_ch = numpy.median(chans_range).round().astype(dtype='int')-1
            chans_index = numpy.concatenate([chans_range[0:3], chans_range[median_ch-1:median_ch+2], chans_range[-3::]])
        else:
            chans_index = range(nChans)

        steps = []
        for c in range(nConcatenations):
            traces = si_recording.get_traces(start_frame=concatenationSamples[c]-samples_radius, end_frame=concatenationSamples[c]+samples_radius)
            meanTraces = traces.mean(axis=0)
            stdTraces = traces.std(axis=0)
            steps.append(max([abs(max(meanTraces + stdTraces)), abs(min(meanTraces - stdTraces))]))
            del traces, meanTraces, stdTraces

        step = max(steps)    

        for c in range(nConcatenations):

            sample_start = int(concatenationSamples[c]-samples_radius)
            sample_end = int(concatenationSamples[c]+samples_radius)
                
            traces_time = si_recording.sample_index_to_time(numpy.array(range(sample_start, sample_end))) - concatenationTimes[c]

            fig, axis = plt.subplots(ncols=nRecs, sharex=True, sharey=True)

            if nRecs>1:

                for r in range(nRecs):

                    traces = si_recording_dict[recNames[r]].get_traces(start_frame=sample_start, end_frame=sample_end)
                
                    axis[r].plot((0, 0), (-step, step*len(chans_index)), color=color_line_concat, linestyle='dotted')
                    
                    count_ch = 0
                    for ch in chans_index:
                        axis[r].plot(traces_time, traces[:, ch]+(step*count_ch), alpha=0.75)
                        count_ch +=1

                    axis[r].set_title(recNames[r])
                    
                    axis[median_recs].set_xlabel('time[s]')
                    axis[r].set_xlim(-plot_windows_secs, plot_windows_secs)
                    axis[r].set_rasterized(True)
                    
                fig.suptitle('Concatenation: {} [{} s]'.format(c+1, concatenationTimes[c])) 
                
            else:
                
                traces = si_recording.get_traces(start_frame=sample_start, end_frame=sample_end)
                
                axis.plot((0, 0), (-step, step*len(chans_index)), color=color_line_concat, linestyle='dotted')

                count_ch = 0     
                for ch in chans_index:
                    axis.plot(traces_time, traces[:, ch]+(step*count_ch), alpha=0.75)
                    count_ch +=1

                axis.set_title(recNames[0])

                axis.set_xlabel('time[s]')

                axis.set_xlim(-plot_windows_secs, plot_windows_secs)

                axis.set_rasterized(True)
                    
                fig.suptitle('Concatenation: {} [{} s]'.format(c+1, concatenationTimes[c])) 

            if savePlots: 
                sess_electrodeGroup_Name = si_recording.get_annotation('elecGroupSessName') + si_recording.get_annotation('recording_sufix')
                fig.set_figheight(fig_height_global)
                fig.set_figwidth(fig_width_global)
                fig.savefig(os.path.join(preproPlots_folder, '{}-Concat-{}.eps'.format(sess_electrodeGroup_Name, c+1)), dpi=fig_dpi, format='eps')
            if showPlots:
                plt.show()
            else:
                plt.close(fig=fig)
    else:
        print('This recording do not have concatenated sessions (nSess = {})'.format(nConcatenations))



#########################################################################################################################################
# PLOT TO INSPECT FOR NOISE AT SPECIFIC FREQUENCIES
#########################################################################################################################################
def plotPSD_randomChunks(si_recording, compare_CMR=True, plot_by_channel=True, chan_radius=None, showPlots=True, savePlots=False, folderPlots=None):

    nChans = si_recording.get_num_channels()

    # Create PSD folder if savePlots & plot_by_channel
    if folderPlots is None:
        if 'savePath' in si_recording._annotations:
            preproPlots_folder = os.path.join(si_recording.get_annotation('savePath'), 'preproPlots')
        else:
            savePlots = False
            preproPlots_folder = None
            showPlots = True
            print('PSD-Plots will not be saved')
    else:
        preproPlots_folder = folderPlots
    
    sess_electrodeGroup_Name = si_recording.get_annotation('elecGroupSessName') + si_recording.get_annotation('recording_sufix')

    if plot_by_channel and savePlots and nChans>1 and preproPlots_folder is not None:
        if compare_CMR:
            psd_chans_folder = os.path.join(preproPlots_folder, 'CMR-PSD_by_chan')
        else:
            psd_chans_folder = os.path.join(preproPlots_folder, 'PSD_by_chan')
        if not os.path.isdir(psd_chans_folder):
            os.mkdir(psd_chans_folder)
    
    # Get Chunks of data
    data_chunk = get_random_data_chunks(
        recording = si_recording,
        num_chunks_per_segment=50,
        chunk_size=10000,
        seed=0
    )

    ##########################################################
    # Apply Common Median Reference to compare
    if compare_CMR and nChans>1:

        if chan_radius is None:
            recObj_cmr = spre_common_reference(recording=si_recording, reference='global', operator='median')
        else:
            step_chan = si_recording.get_annotation('y_contacts_distance_um') # GET CHANNEL-CONTACT SPACING ("y" coordinate)
            recObj_cmr = spre_common_reference(recording=si_recording, reference='local', operator='median', local_radius=(chan_radius[0] * step_chan, chan_radius[1] * step_chan ))

        # Get Chunks of data
        data_chunk_cmr = get_random_data_chunks(
            recording = recObj_cmr,
            num_chunks_per_segment=50,
            chunk_size=10000,
            seed=0
        )
                
    # Plot the power spectral density for each channel to inspect specific noise frequencies 
    if nChans==1:

        fig, ax = plt.subplots(ncols=1, figsize=(10, 7))
        p, f = ax.psd(data_chunk[:, 0], Fs=si_recording.sampling_frequency, color="b")
        ax.set_rasterized(True)
        if savePlots: 
            fig.set_figheight(fig_height_global)
            fig.set_figwidth(fig_width_global)
            fig.savefig(os.path.join(preproPlots_folder, '{}-PSD.eps'.format(sess_electrodeGroup_Name)), dpi=fig_dpi, format='eps')
        if showPlots:
            plt.show()
        else:
            plt.close(fig=fig)

    else:

        if compare_CMR:

            fig, axis = plt.subplots(ncols=2)
            for tr in data_chunk.T:
                p, f = axis[0].psd(tr, Fs=si_recording.sampling_frequency, color="b")
            axis[0].set_title('filtered-PSD')
            axis[0].set_rasterized(True)

            for tr in data_chunk_cmr.T:
                p, f = axis[1].psd(tr, Fs=si_recording.sampling_frequency, color="r")
            axis[1].set_title('CMR-PSD')
            axis[1].set_rasterized(True)

            fig.suptitle('All channels')

            if savePlots: 
                fig.set_figheight(fig_height_global)
                fig.set_figwidth(fig_width_global)
                fig.savefig(os.path.join(preproPlots_folder, '{}-CMR-PSD.eps'.format(sess_electrodeGroup_Name)), dpi=fig_dpi, format='eps')
            if showPlots:
                plt.show()
            else:
                plt.close(fig=fig)

        else:

            fig, axis = plt.subplots(ncols=1)
            for tr in data_chunk.T:
                p, f = axis.psd(tr, Fs=si_recording.sampling_frequency, color="b")
            axis.set_title('filtered-PSD')
            axis.set_rasterized(True)

            fig.suptitle('All channels')

            if savePlots: 
                fig.set_figheight(fig_height_global)
                fig.set_figwidth(fig_width_global)
                fig.savefig(os.path.join(preproPlots_folder, '{}-PSD.eps'.format(sess_electrodeGroup_Name)), dpi=fig_dpi, format='eps')
            if showPlots:
                plt.show()
            else:
                plt.close(fig=fig)
    
        if plot_by_channel:

            chanIDs = si_recording.get_channel_ids()
            deviceIDs = [si_recording.get_channel_property(chanID, 'contact_vector')['device_channel_indices'] for chanID in chanIDs]
            
            for ch in range(nChans):
                
                if compare_CMR:
                    fig, axis = plt.subplots(ncols=2)
                    for tr in data_chunk.T:
                        p, f = axis[0].psd(tr, Fs=si_recording.sampling_frequency, color="k", alpha=0.05)
                    p, f = axis[0].psd(data_chunk[:, ch], Fs=si_recording.sampling_frequency, color="b", linewidth=1.5)
                    axis[0].set_title('filtered-PSD')
                    axis[0].set_rasterized(True)
                    for tr in data_chunk_cmr.T:
                        p, f = axis[1].psd(tr, Fs=si_recording.sampling_frequency, color="m", alpha=0.05)
                    p, f = axis[1].psd(data_chunk_cmr[:, ch], Fs=si_recording.sampling_frequency, color="r", linewidth=1.5)
                    axis[1].set_title('CMR-PSD')
                    axis[1].set_rasterized(True)

                    fig.suptitle('ch={} (devID={})'.format(chanIDs[ch], deviceIDs[ch]))

                    if savePlots: 
                        fig.set_figheight(fig_height_global)
                        fig.set_figwidth(fig_width_global)
                        fig.savefig(os.path.join(psd_chans_folder, '{}-CMR-PSD-dev{}-id{}.eps'.format(sess_electrodeGroup_Name, deviceIDs[ch], chanIDs[ch])), dpi=fig_dpi, format='eps')
                    if showPlots:
                        plt.show()
                    else:
                        plt.close(fig=fig)

                else:

                    fig, axis = plt.subplots(ncols=1)
                    for tr in data_chunk.T:
                        p, f = axis.psd(tr, Fs=si_recording.sampling_frequency, color="k", alpha=0.05)
                    p, f = axis.psd(data_chunk[:, ch], Fs=si_recording.sampling_frequency, color="b", linewidth=1.5)
                    axis.set_title('filtered-PSD')
                    axis.set_rasterized(True)

                    fig.suptitle('ch={} (devID={})'.format(chanIDs[ch], deviceIDs[ch]))

                    if savePlots: 
                        fig.set_figheight(fig_height_global)
                        fig.set_figwidth(fig_width_global)
                        fig.savefig(os.path.join(psd_chans_folder, '{}-PSD-dev{}-id{}.eps'.format(sess_electrodeGroup_Name, deviceIDs[ch], chanIDs[ch])), dpi=fig_dpi, format='eps')
                    if showPlots:
                        plt.show()
                    else:
                        plt.close(fig=fig)

       

#########################################################################################################################################
# FUNCTION TO PLOT A PEAK LOCATIONS AS A FUNCTION OF TIME
# IN CASE OF SINGLE CHANNEL RECORDINGS, THE PLOT WILL SHOW THE AMPLITUDE OF THE PEAKS AS A FUNCTION OF TIME
#########################################################################################################################################
def plot_peakLocations(si_recording, folderPeaks, peaks_options=None, locations_options=None, rewrite=False, locationsSubSampled=True,
                       showPlots=True, savePlots=False, folderPlots=None):
    
    if savePlots or showPlots:

        if folderPlots is None:
            folderPlots = folderPeaks

        concatenationTimes = None
        color_line_concat = (1, 0, 0, 1)
        if si_recording.get_annotation('nConcatenations')>0:
            concatenationTimes = si_recording.get_annotation('concatenationTimes')
            
        step_chan = si_recording.get_annotation('y_contacts_distance_um')
        nChans = si_recording.get_num_channels()

        contact_locations = si_recording.get_property('location')

        if peaks_options is None:
            peaks_options = {
                    'method': "locally_exclusive",
                    'peak_sign': peak_sign_default, 
                    'detect_threshold': peak_detect_threshold_default, #  MAD: Median Amplitude Deviations
                    'radius_um': step_chan*nearest_chans_default,
                }
        
        if locations_options is None:
            locations_options = {     
                    'ms_before': ms_before_default,
                    'ms_after': ms_after_default,
                    'location_method': 'monopolar_triangulation', # Paninski Lab
                    'location_kwargs': {'max_distance_um': step_chan*(nearest_chans_default+1), 'optimizer': 'least_square'}
                }
            
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
        if nChans>1:
            
            if locations_options['location_method']=='center_of_mass':
                loc_label = '_mass'
            elif locations_options['location_method']=='monopolar_triangulation':
                loc_label = '_mono'
            elif locations_options['location_method']=='grid_convolution':
                loc_label = '_grid'
            else:
                raise Exception('Peaks Location method "{}" not recognized\nAvailable options: {}'.format(peaks_options['method'], ['center_of_mass', 'monopolar_triangulation', 'grid_convolution'])) 
        else:
            loc_label =  ''   
        
        # If Peak Detection & Location methods match with prefix, remove it to avoid long naming
        recording_sufix = si_recording.get_annotation('recording_sufix').replace(peaks_label+loc_label + '_', '')

        ###############################################################################
        # Detect PEAKS 
        if not os.path.exists(os.path.join(folderPeaks, 'peaks_' + peaks_label + recording_sufix + '.npy')) or rewrite:

            noise_levels = get_noise_levels(si_recording, return_scaled=False)
            
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
                    
            numpy.save(os.path.join(folderPeaks, 'peaks_' + peaks_label + recording_sufix + '.npy'), peaks)

            del peaks, noise_levels
        else:
            print('Subsampled peaks file was found ¡')
        
        if locationsSubSampled:

            peaksSampled_label = 'Sampled_'

            if not os.path.exists(os.path.join(folderPeaks, 'peaksSampled_' + peaks_label + recording_sufix + '.npy')) or rewrite:

                noise_levels = get_noise_levels(si_recording, return_scaled=False)

                peaks = numpy.load(os.path.join(folderPeaks, 'peaks_' + peaks_label + recording_sufix + '.npy'))

                peaksSampled = select_peaks(peaks, method='smart_sampling_amplitudes', n_peaks=5000, select_per_channel=True, noise_levels = noise_levels, return_indices=False) 

                numpy.save(os.path.join(folderPeaks, 'peaksSampled_' + peaks_label + recording_sufix + '.npy'), peaksSampled)

                del peaksSampled, peaks, noise_levels
            
            else:
                print('Subsampled peaks file was found ¡')

        else:
            peaksSampled_label = '_'

        peaks = numpy.load(os.path.join(folderPeaks, 'peaks' + peaksSampled_label + peaks_label + recording_sufix + '.npy'))
        
                
        if nChans>1:
                
            ##################################################################
            # Localize PEAKS 
            ##################################################################
            if not os.path.exists(os.path.join(folderPeaks, 'peaks' + peaksSampled_label + peaks_label + '_locations' + loc_label + recording_sufix + '.npy')) or rewrite:

                peaksSampled_locations = localize_peaks(
                    recording=si_recording,
                    peaks=peaks,
                    ms_before= locations_options['ms_before'],
                    ms_after= locations_options['ms_after'],
                    radius_um = peaks_options['radius_um'],
                    method= locations_options['location_method'],
                    **locations_options['location_kwargs'],
                    **job_kwargs
                    )

                numpy.save(os.path.join(folderPeaks, 'peaks' + peaksSampled_label + peaks_label + '_locations' + loc_label + recording_sufix + '.npy'), peaksSampled_locations)

                del peaksSampled_locations
            
            else:
                print('File with Locations of {} was found.... '.format( 'peaks' + peaksSampled_label[:-1]))
                                
            peaks_locations = numpy.load(os.path.join(folderPeaks, 'peaks' + peaksSampled_label + peaks_label + '_locations' + loc_label + recording_sufix + '.npy'))

            x = peaks['sample_index'] / si_recording.get_sampling_frequency()
            y = peaks_locations['y']

            y_label = 'Probe Location (\u03BCm)'
            minY_loc = -50
            maxY_loc = max(contact_locations[:, 1])+50
            alpha_val = 0.05
            

        else:

            x = peaks['sample_index'] / si_recording.get_sampling_frequency()
            y = peaks['amplitude']

            y_label = 'Peaks Amp (\u03BCV)'
            minY_loc = numpy.ceil(min(peaks['amplitude'])/10)*10
            maxY_loc = numpy.ceil(max(peaks['amplitude'])/10)*10
            alpha_val = 0.25


        ##################################################################
        # PLOT LOCATION vs TIME
        fig, ax = plt.subplots()
        ax.scatter(x, y, s=1, color='k', alpha=alpha_val)
        #Plot concatenation
        if concatenationTimes is not None:
            ax.plot((concatenationTimes, concatenationTimes), (minY_loc, maxY_loc), color=color_line_concat, linestyle='solid', linewidth=0.25)

        ax.set_ylim(minY_loc, maxY_loc)
        ax.set_ylabel(y_label)
        ax.set_rasterized(True)

        fig.suptitle(peaks_label + loc_label + recording_sufix)

        if savePlots: 

            sess_electrodeGroup_Name = si_recording.get_annotation('elecGroupSessName')

            fig.set_figheight(fig_height_global)
            fig.set_figwidth(fig_width_global)
            fig.savefig(os.path.join(folderPlots, sess_electrodeGroup_Name + '_peaks' + peaksSampled_label + peaks_label + loc_label + recording_sufix + '.eps'), dpi=fig_dpi, format='eps')
        
        if showPlots:
            plt.show()
        else:
            plt.close(fig=fig)


#########################################################################################################################################
# FUNCTION TO PLOT THE OUTPUTS OF MOTION ESTIMATION, ALL INPUTS MUST BE PROVIDED (I.E., ALREADY COMPUTED)
#########################################################################################################################################
def plot_motion_outputs(peaks, peaks_locations, sampling_frequency, motion, extra_check, peaks_label, loc_label, motion_label, minY_loc, maxY_loc, 
        prefixRec, showPlots=True, savePlots=False, folderPlots=None, concatenationTimes=None, verbose=False):
    
    if savePlots and folderPlots is None:
        raise Exception('To save Motion correction plots you need to provide a folderPath')
    
    if savePlots or showPlots:

        motion_dict = motion.to_dict()
        
        displacement = motion_dict['displacement'][0]
        temporal_bins_s = motion_dict['temporal_bins_s'][0]
        spatial_bins_um = motion_dict['spatial_bins_um']
        interpolation_method = motion_dict['interpolation_method']
        direction = motion_dict['direction']
        color_line_concat = (1, 0, 0, 1)

        if verbose:

            print('Motion shape: ', displacement.shape, '\n')
            print('Temporal_bins shape: ', temporal_bins_s.shape, '\n')
            print('Spatial_bins shape: ', spatial_bins_um.shape,'\n')
            print('Interpolate Method : ', interpolation_method, '\n')
            print('direction: :', direction, '\n')
            print('Extra info: ',extra_check.keys(), '\n')

            if 'motion_histogram' in extra_check:
                print('motion_histogram shape: ', extra_check['motion_histogram'].shape, '\n')
                print('pairwise_displacement_list lentgh: ', len(extra_check['pairwise_displacement_list']), '\n')
                print('pairwise_displacement_list [0] shape: ',extra_check['pairwise_displacement_list'][0].shape, '\n')

        ##################################################################
        # FIGURE 1
        ##################################################################
        # Heat map of motion displacement
        if 'noRigid' in motion_label:
            nRows = 2
        else:
            nRows = 1
        
        if 'motion_histogram' in extra_check:
            nCols = 2
        else:
            nCols = 1

        fig, ax = plt.subplots(ncols=nCols, nrows=nRows)

        if nCols==2 and nRows==2:
            extent = (temporal_bins_s[0], temporal_bins_s[-1], spatial_bins_um[0], spatial_bins_um[-1])
            im = ax[1, 0].imshow(
                extra_check['motion_histogram'].T,
                interpolation='nearest',
                origin='lower',
                aspect='auto',
                extent=extent,
            )
            im.set_clim(0, 50)
            if concatenationTimes is not None:
                ax[1, 0].plot((concatenationTimes, concatenationTimes), (minY_loc, maxY_loc), color=color_line_concat, linestyle='solid', linewidth=0.25)
            ax[1, 0].set_xlabel('time[s]')
            ax[1, 0].set_ylabel('depth[um]')
            fig.colorbar(im, location='bottom')
            ax[1, 0].set_title('Motion Histogram')
            ax[1, 0].set_ylim(minY_loc, maxY_loc)
            ax[1, 0].set_rasterized(True)
        
        if nCols==2:
            if nRows==2:
                ax_pw = ax[0, 0]
            else:
                ax_pw = ax[0]
                
            # PLOT pairwise displacement
            extent = (temporal_bins_s[0], temporal_bins_s[-1], temporal_bins_s[0], temporal_bins_s[-1])
            im2 = ax_pw.imshow(
                numpy.mean(extra_check['pairwise_displacement_list'], axis=0),
                interpolation='nearest',
                cmap='PiYG',
                origin='lower',
                aspect='auto',
                extent=extent,
                )
            im2.set_clim(-40, 40)
            ax_pw.set_aspect('equal')
            fig.colorbar(im2, location='bottom')
            ax_pw.set_title('Pairwise Displacement')
            ax_pw.set_rasterized(True)
        
        ##################################################################
        # PLOT Motions 
        if nRows==2:
            if nCols==2:
                ax_m = ax[0, 1]
            else:
                ax_m = ax[0]
        else:
            if nCols==2:
                ax_m = ax[1]
            else:
                ax_m = ax
        ax_m.plot(temporal_bins_s, displacement)
        if concatenationTimes is not None:
            ax_m.plot((concatenationTimes, concatenationTimes), (numpy.min(displacement)*1.05, numpy.max(displacement)*1.05), color=color_line_concat, linestyle='solid', linewidth=0.25)
        ax_m.set_title('Motion')
        ax_m.set_rasterized(True)

        if nRows==2:
            if nCols==2:
                ax_m2 = ax[1, 1]
            else:
                ax_m2 = ax[1]
            im = ax_m2.imshow(displacement.T,
                interpolation='nearest',
                cmap='PiYG',
                origin='lower',
                aspect='auto',
            )
            im.set_clim(-40, 40)
            ax_m2.set_aspect('equal')
            ax_m2.set_title('Motion (heatmap)')
            ax_m2.set_rasterized(True)
        
        fig.suptitle(peaks_label + '-' + loc_label + '-' + motion_label)
        
        if savePlots: 
            fig.set_figheight(fig_height_global)
            fig.set_figwidth(fig_width_global)
            fig.savefig(os.path.join(folderPlots, '{}_{}_{}_{}_motion.eps'.format(prefixRec, peaks_label, loc_label, motion_label)), dpi=fig_dpi, format='eps')
            
        if showPlots:
            plt.show()
        else:
            plt.close(fig=fig)

        ##################################################################
        # Figure 2:  
        ##################################################################
        # PLOT motion on top of the peak-localization
        fig, ax = plt.subplots(ncols=nRows+1)

        x = peaks['sample_index'] / sampling_frequency
        y = peaks_locations['y']

        ax[0].scatter(x, y, s=1, color='k', alpha=0.01)
        if concatenationTimes is not None:
            ax[0].plot((concatenationTimes, concatenationTimes), (minY_loc, maxY_loc), color=color_line_concat, linestyle='solid', linewidth=0.25)
        ax[0].set_title('Original')
        ax[0].set_ylim(minY_loc, maxY_loc)
        ax[0].set_rasterized(True)


        ax[nRows].scatter(x, y, s=1, color='k', alpha=0.01)

        for b in range(displacement.shape[1]):
            ax[nRows].plot(temporal_bins_s, displacement[:, b] + spatial_bins_um[b], alpha=0.5, color='r')
            if nRows>1:
                ax[nRows-1].plot(temporal_bins_s, displacement[:, b] + spatial_bins_um[b], alpha=0.5, color='r')
                ax[nRows-1].set_rasterized(True)

        if nRows>1:
            if concatenationTimes is not None:
                ax[1].plot((concatenationTimes, concatenationTimes), (minY_loc, maxY_loc), color=color_line_concat, linestyle='solid', linewidth=0.25)
            ax[1].set_title('Motion')
            ax[1].set_ylim(minY_loc, maxY_loc)
            ax[1].set_rasterized(True)
        
        if concatenationTimes is not None:
            ax[nRows].plot((concatenationTimes, concatenationTimes), (minY_loc, maxY_loc), color=color_line_concat, linestyle='solid', linewidth=0.25)
            
        ax[nRows].set_title('Original + Motion')
        ax[nRows].set_ylim(minY_loc, maxY_loc)
        ax[nRows].set_rasterized(True)

        fig.suptitle(peaks_label + '-' + loc_label + '-' + motion_label)

        if savePlots: 
            fig.set_figheight(fig_height_global)
            fig.set_figwidth(fig_width_global)
            fig.savefig(os.path.join(folderPlots, '{}_{}_{}_{}_motionPeaks.eps'.format(prefixRec, peaks_label, loc_label, motion_label)), dpi=fig_dpi, format='eps')
            
        if showPlots:
            plt.show()
        else:
            plt.close(fig=fig)

