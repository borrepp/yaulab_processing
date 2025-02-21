'''
This package contains functions, classes, and objects to compute the following processes:

    1) Reading and combining Behavioral (*.yaml & *.eye; from YAU-Lab native format) & Electrophysiological (*.nev & *.ns5 from Ripple-Neuroshare format) into NWB format. 
        - package "yaulab_nwb": Based on "pynwb" toolbox.

    2) Create a processed NWB with LFPs from raw-NWB & ripple-EYE data downsampled to 1-KHz. 
        - package "yaulab_nwb": Based on "spikeinterface" & "pynwb" toolboxes.

    3) Reading raw-NWB electrophysiological data to preprocess and sort single unts. 
        - package "yaulab_si": Based on "spikeinterface" toolbox.

    4) Add preprocessed raw-signal & sorting results to the processed lfp-NWB. 
        - package "yaulab_si": Based on "spikeinterface" & "neuroconv" toolboxes.


------------------------------------------------------------------------------------
Questions and comments:
    José Vergara de la Fuente. 
    borre.pp@gmail.com
    github: borrepp

------------------------------------------------------------------------------------
Last review: 2024-Dec-01

'''

from .yaulab_extras import (
    version,
    get_tempdir, 
    clear_tempdir, 
    get_filePaths_to_extract_raw, 
    get_filePaths_to_extract_prepro,
    get_filePaths_to_extract_prepro_by_date
    )
from .yaulab_nwb.yaml_tools import yaml2dict, expYAML
from .yaulab_nwb.nwb_tools import (
    createNWB_raw, 
    createNWB_prepro,
    write_curatedFolder_into_nwb_sessions
    )
from .yaulab_si import (
    run_prepro_expDAY_in_range, 
    run_prepro_expDAY, 
    run_prepro, 
    run_prepro_and_sorting,
    lfp_params_spikeInterface_default,
    export_curatedFolder_to_sortingAnalyzer_sessions
    )

