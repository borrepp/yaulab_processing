import yaml
import os
import pandas
import numpy
import shutil
from warnings import warn

##############################################################################################################
# Class to track messages printed on the command window
##############################################################################################################
class Unbuffered:
    def __init__(self, stream, fwriteMode):
        self.stream = stream
        self.fwriteMode = fwriteMode
    def write(self, data):
        self.fwriteMode.write(data)    # Write the data of stdout here to a text file:
        self.stream.write(data)
        self.stream.flush()

##############################################################################################################
version = '0.0.1'
version1_date ='2024-02-14' # YAML or RIpple Files previous to this date, might not be compatible with this new format

# Date when Thermistor-Probe was added
date_temperature = '2024-06-18' 

# Possible Stim 2 remove and/or Resample: 
# ['fixON', 'visualON', 'rewardON', 'leftAccelerometer', 'leftCommand', 'rightAccelerometer', 'rightCommand', 'thermistors']
removeRipple_stimulus_list_default = ['fixON', 'visualON', 'rewardON', 'leftCommand', 'rightCommand', 'thermistors']
resample_stimulus_list_default = ['leftAccelerometer', 'rightAccelerometer', 'thermistors']

##############################################################################################################
# LOAD EXTRA FILES
_currentPath, _ = os.path.split(os.path.abspath(__file__))

# Extract monkeyIDs-YAML as nested Dictionaries
with open(os.path.join(_currentPath, 'monkeyIDs.yaml'), 'r') as _stream:
    monkeysDict = yaml.safe_load(_stream)
del _stream


# Extract yauLabInfo-YAML as nested Dictionaries
with open(os.path.join(_currentPath, 'yauLabInfo.yaml'), 'r') as _stream:
    labInfo = yaml.safe_load(_stream)
del _stream

# Extract excel "ElectrodeDevices" as pandas dataframe
electrodeDevices = pandas.read_excel(os.path.join(_currentPath, 'ElectrodeDevices.xlsx'), sheet_name=None)
supported_probes_manufacturer = ['PLX', 'FHC']

##############################################################################################################
# WAVECLUS matlab toolbox should be part of the current directory
waveclus_path = os.path.join(_currentPath, 'wave_clus-master')

########################################################################################################################
# SET UP A TEMPORARY DIRECTORY TO PROCESS DATA
########################################################################################################################
environ_tempName_prefix = 'YaulabTemp'
processName_default = 'processing'

def get_tempdir(processName=processName_default, resetDir=True):

    # CONFIRM or ADD TEMPORAL DIR ('YAULAB_PROCESSOR_TEMPDIR') into the variables of the environment
    environVarName = environ_tempName_prefix + '_' + processName

    if not os.environ.get(environVarName):
        root_env = os.path.abspath("/")
        os.environ[environVarName] = os.path.join(root_env, environ_tempName_prefix, processName)
        print(environVarName + ' variable was created')
	
    # CONFIRM THAT EXISTS or CREATE THE TEMPORAL DIR ('SI_PROCESSOR_TEMPDIR')
    if not os.path.exists(os.environ.get(environVarName)):
        os.makedirs(os.environ.get(environVarName))
        print('Temporal folder was created ')
    
    if resetDir:
        clear_tempdir(os.environ.get(environVarName), verbose=False)
        os.makedirs(os.environ.get(environVarName))

    return os.environ.get(environVarName)


########################################################################################################################
# CLEAR ALL THE TREE FROM A TEMPORARY DIRECTORY
########################################################################################################################
def clear_tempdir(tempFolderPath, verbose=True):

    for root, _, files in os.walk(tempFolderPath, topdown=False):

        remove_dir = True

        for fname in files:
            tempFile_path = os.path.join(root, fname)
            if os.path.isfile(tempFile_path):
                try:
                    os.remove(tempFile_path)
                except:
                    remove_dir = False
                    if verbose:
                        print('Warning:\nUnable to delete the file:\n{}\nprobably is still open or in use\n\n'.format(tempFile_path))
        if remove_dir:
            try:
                os.rmdir(root)
            except:
                if verbose:
                    print('Warning:\nUnable to delete the folder:\n{}\nprobably it is not an empty folder\n\n'.format(root))


##############################################################################################################
# Check for YAML files with an NWB (exclude -noNEV.nwb)
##############################################################################################################
# get YAML file Paths to be extracted
def get_YAMLpaths_with_nwbRaw(folderName, dateStart='0000-00-00', dateEnd=None):
    
    yearStart = int(dateStart[0:4])
    monthStart = int(dateStart[5:7])
    dayStart = int(dateStart[8:])

    if dateEnd is None:
        dateEnd = '9999-99-99'

    yearEnd = int(dateEnd[0:4])
    monthEnd = int(dateEnd[5:7])
    dayEnd = int(dateEnd[8:])
        
    yaml_with_nwbRaw = []

    folderName = os.path.abspath(folderName)

    for root, _, files in os.walk(folderName):

        for name in files:

            nameSplit = os.path.splitext(name)

            if nameSplit[1]=='.yaml':

                fileName = nameSplit[0]
                yearFile = int(fileName[3:7])
                monthFile = int(fileName[8:10])
                dayFile = int(fileName[11:13])

                yearCompatible = False
                monthCompatible = False
                dayCompatible = False

                if yearFile>=yearStart and yearFile<yearEnd:
                    yearCompatible = True
                if monthFile>=monthStart and monthFile<monthEnd:
                    monthCompatible = True
                if dayFile>=dayStart and dayFile<dayEnd:
                    dayCompatible = True
                
                versionCompatible = False
                if yearCompatible and monthCompatible and dayCompatible:
                    versionCompatible = True

                filePathYAML = os.path.join(root, fileName + '.yaml')
                
                if versionCompatible:
                    # Check NWB
                    filePathNWB = os.path.join(root, fileName + '.nwb')

                    if os.path.isfile(filePathNWB):
                        yaml_with_nwbRaw.append(filePathYAML)


    if len(yaml_with_nwbRaw)==0:
        print('\nThere were no YAML files with NWB file within the same folder with the same name\nCheck dates to search for YAML-NWB files (start: {} - end: {})\n'.format(
            dateStart, dateEnd))
    
    return yaml_with_nwbRaw


##############################################################################################################
# CHECK IF THE DATE OF THE FILE CAN CONTAIN ANALOG INPOTS FROM THERMISTORS
##############################################################################################################
def check_temperature_date(fileName):

    yearFile = int(fileName[3:7])
    monthFile = int(fileName[8:10])
    dayFile = int(fileName[11:13])

    
    yearTemp= int(date_temperature[0:4])
    monthTemp = int(date_temperature[5:7])
    dayTemp = int(date_temperature[8:])

    if yearFile>yearTemp:
        versionCompatible = True
    elif yearFile==yearTemp and monthFile>monthTemp:
        versionCompatible = True
    elif yearFile==yearTemp and monthFile==monthTemp and dayFile>=dayTemp:
        versionCompatible = True
    else:
        versionCompatible = False
    
    return versionCompatible

##############################################################################################################
# get YAML file Paths to be extracted
##############################################################################################################
def get_filePaths_to_extract_raw(parentFolder, fileName=None, updateFiles = False, dateUpdate=version1_date):
        
    yamlFiles2convert = []

    yearUpdate = int(dateUpdate[0:4])
    monthUpdate = int(dateUpdate[5:7])
    dayUpdate = int(dateUpdate[8:])

    parentFolder = os.path.abspath(parentFolder)

    
    fileNameNWBexits = False
    for root, _, files in os.walk(parentFolder):

        for name in files:

            nameSplit = os.path.splitext(name)

            if nameSplit[1]=='.yaml':

                sessionName = nameSplit[0]
                yearFile = int(sessionName[3:7])
                monthFile = int(sessionName[8:10])
                dayFile = int(sessionName[11:13])

                if yearFile>yearUpdate:
                    versionCompatible = True
                elif yearFile==yearUpdate and monthFile>monthUpdate:
                    versionCompatible = True
                elif yearFile==yearUpdate and monthFile==monthUpdate and dayFile>=dayUpdate:
                    versionCompatible = True
                else:
                    versionCompatible = False
                
                validSession = True
                if fileName is not None:
                    if sessionName != fileName:
                        validSession = False
                
                if versionCompatible and validSession:
                    filePathYAML = os.path.join(root, sessionName + '.yaml')
                    if updateFiles:
                        yamlFiles2convert.append(filePathYAML)
                    else:
                        filePathNev = os.path.join(root, sessionName + '.nev')
                        if not os.path.isfile(filePathNev):
                            extNWB = '-noNEV'
                        else:
                            extNWB = ''
                            
                        filePathNWB = os.path.join(root, sessionName + extNWB + '.nwb')
                        if not os.path.isfile(filePathNWB):
                            yamlFiles2convert.append(filePathYAML)
                        elif fileName is not None:
                            fileNameNWBexits = True

    if len(yamlFiles2convert)==0:
        if fileName is None:
            warn('\nThere were no YAML files to convert ("updateFiles" was set to: {})\
                \nFiles created before {} are not compatible\n'.format(updateFiles, dateUpdate))
        elif fileNameNWBexits:
            warn('\nThe YAML file {} has been converted already\nCheck that "updateFiles" was set to: {})\
                \nFiles created before {} are not compatible\n'.format(fileName, updateFiles, dateUpdate))
        else:
            warn('\nThere was no YAML file that matches fileName = {}\n"updateFiles" was set to: {}\
                \nFiles created before {} are not compatible\n'.format(fileName, updateFiles, dateUpdate))
    
    return yamlFiles2convert


##############################################################################################################
# get raw NWB file Paths with potential ephys data to create a preproNWB
##############################################################################################################
def get_filePaths_to_extract_prepro(parentFolder, fileName=None):
        
    nwbFiles2convert = []

    parentFolder = os.path.abspath(parentFolder)

    for root, _, files in os.walk(parentFolder):

        for name in files:

            nameSplit = os.path.splitext(name)

            # Parent Folder must contain YAML & NWB. It will be requiered to update Receptive Field mapping 
            if nameSplit[1]=='.yaml':

                sessionName = nameSplit[0]

                filePathNev = os.path.join(root, sessionName + '.nev')
                filePathNWB = os.path.join(root, sessionName + '.nwb')

                # Confirm NEV file exists in the same folder with the same name as the NWB (this is the most strict criteria to confirm it is a raw-NWB)
                # it will exclude NWB with suffix added (i.e.: -noNEV.nwb)
                if os.path.isfile(filePathNev) and os.path.isfile(filePathNWB):
                    if fileName is None:
                        nwbFiles2convert.append(filePathNWB)
                    elif fileName == sessionName:
                        nwbFiles2convert.append(filePathNWB)   
                else:
                    if fileName is None:
                        warn('NEV file was NOT found within the same folder of the raw NWB file: {}\nContainer folder:{}'.format(sessionName+'.nwb', root))
                    elif fileName == sessionName:
                        warn('NEV file was NOT found within the same folder of the raw NWB file: {}\nContainer folder:{}'.format(fileName+'.nwb', root))


    if len(nwbFiles2convert)==0:
        if fileName is None:
            warn('\nThere were no raw NWB files to convert in the parent folder: {}'.format(parentFolder))
        else:
            warn('\nThere were no raw NWB files to match session : {}\nCheck if parent folder is correct: '.format(fileName, parentFolder))

    return nwbFiles2convert


def get_filePaths_to_extract_prepro_by_date(parentFolder,
        year_start, month_start, day_start, 
        year_stop, month_stop, day_stop
    ):

    nwbFiles2convert = []

    parentFolder = os.path.abspath(parentFolder)

    for root, _, files in os.walk(parentFolder):

        for name in files:

            nameSplit = os.path.splitext(name)

            # Parent Folder must contain YAML & NWB. It will be requiered to update Receptive Field mapping 
            if nameSplit[1]=='.yaml':

                sessionName = nameSplit[0]
                yearFile = int(sessionName[3:7])
                monthFile = int(sessionName[8:10])
                dayFile = int(sessionName[11:13])

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
                    filePathNev = os.path.join(root, sessionName + '.nev')
                    filePathNWB = os.path.join(root, sessionName + '.nwb')

                    # Confirm NEV file exists in the same folder with the same name as the NWB (this is the most strict criteria to confirm it is a raw-NWB)
                    # it will exclude NWB with suffix added (i.e.: -noNEV.nwb)
                    if os.path.isfile(filePathNev) and os.path.isfile(filePathNWB):
                        nwbFiles2convert.append(filePathNWB)   
                    else:
                        warn('NEV file was NOT found within the same folder of the raw NWB file: {}\nContainer folder:{}'.format(sessionName+'.nwb', root))

    if len(nwbFiles2convert)==0:
        print('\nThere were no raw NWB files to convert in the parent folder: {}\nCheck date range: \n\tstart: {}-{:02d}-{:02d}\n\tstop:  {}-{:02d}-{:02d}'.format(
            parentFolder, year_start, month_start, day_start, 
                          year_stop, month_stop, day_stop))
        warn('\nNo raw NWB to process¡')

    return nwbFiles2convert

##############################################################################################################
# check if container Folder has the same name as YAML, otherwise create the folder and copy:
# *.eye, *.nev, *nsX, *.nwb
##############################################################################################################
def check_folderSession_process(filePathYAML, processName_prefix, copy2disk=True):

    filePath, fileNameExt = os.path.split(os.path.abspath(filePathYAML))
    fileName, _ = os.path.splitext(fileNameExt)

    # Create Temporary directory to save intermediate processing files (hdf5)
    folder_temporal = get_tempdir(processName='{}-{}'.format(processName_prefix, fileName[3::]), resetDir=True)

    # Check if files need to be moved to a temporal folder
    copyFiles = False
    if copy2disk and filePath[0:2].lower()!=os.path.abspath(__file__)[0:2].lower():
        copyFiles = True

    dir_list = os.listdir(filePath)
    files2extract = [n for n in dir_list if fileName + '.' in n or fileName + '-noNEV.nwb' in n] 

    filesNWB = [n for n in files2extract if '.nwb' in n]

    if len(filesNWB)>0:
        print('Warning: NWB file aready exists it will be overwritten:\n{}'.format(filesNWB))

    _, containerFolder = os.path.split(filePath)
    if fileName==containerFolder:
        # print('Files are already within a session Folder, no move of files will ocurr')
        folder_save = filePath
    else:
        
        folder_save = os.path.join(filePath, fileName)

        if os.path.exists(folder_save):
            # check if Files also exist IN the folder
            dir_list2 = os.listdir(folder_save)
            filesExists = [n for n in dir_list2 if fileName + '.' in n or fileName + '-noNEV.nwb' in n]
            file2replace = [f for f in files2extract if f in filesExists]
            if len(file2replace)>0:
                raise Exception('\n\nParentFolder already has a folder for this session\n\nParentFolder:{}\
                                \nParentFolder files:\n{}\
                                \nExisting FolderSession:{}\nFolderSession files:\n{}\
                                \nYou will need to combined/replace manually'.format(
                                    filePath, files2extract, folder_save, filesExists)
            )
            print('Session-Folder already exists:\n{}\nFiles will be moved to this folder'.format(folder_save))
        else:
            print('A new folder for this session will be created:\n{}'.format(folder_save))
            os.mkdir(folder_save)

        print('The following files will be moved:\n{}'.format(files2extract))

        for f in files2extract:
            shutil.move(os.path.join(filePath, f), os.path.join(folder_save, f), copy_function = shutil.copy2)
    
    if copyFiles:

        print('\nOriginal files will be copied to temporal folder: {}...'.format(folder_temporal))

        folder_read = folder_temporal

        files2copy = [n for n in files2extract if '.nwb' not in n]

        for f in files2copy:
            print('..... copying file {}'.format(f))
            shutil.copy2(os.path.join(folder_save, f), os.path.join(folder_read, f))
    else:
        folder_read = folder_save
    
    return folder_save, folder_read, fileName, folder_temporal
