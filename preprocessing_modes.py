# -*- coding: utf-8 -*-
"""
util_preprocessing.py
Created on Thu Jan 14 13:36:35 2021

@author: MONTGM11
"""

import numpy as np
import glob
import pydicom
import os
from pathlib import Path
from typing import Dict, List, Optional
import traceback

def get_mode(filepath):
    ''' Function to read dicom header and detect m-mode vs b-mode'''
    try:
        data = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)
    except Exception:
        return None
    mode = getattr(data, "OperatingMode", None)
    if mode is None:
        return None
    return str(mode)

def scan_dicoms(dirName: str) -> List[tuple[Path, Optional[str], Optional[str]]]:
    """
    Scan a directory for DICOMs (by .dcm extension) and read OperatingMode.

    Returns a list of (path, mode, error). If reading fails, mode is None and error is a traceback string.
    """
    dir_path = Path(dirName)
    dcm_files = sorted(p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".dcm")
    out: List[tuple[Path, Optional[str], Optional[str]]] = []
    for file_path in dcm_files:
        try:
            mode = get_mode(file_path)
            out.append((file_path, mode, None))
        except Exception:
            out.append((file_path, None, traceback.format_exc()))
    return out

def move_files(parentDir, fileList, subName):
    '''Function to relocate list of files from parentDir to sub directory'''
    parent_dir = Path(parentDir)
    sub_dir = parent_dir / subName
    sub_dir.mkdir(parents=True, exist_ok=True)
    for file in fileList:
        file_path = Path(file)
        if not file_path.is_absolute():
            file_path = parent_dir / file_path
        file_path.replace(sub_dir / file_path.name)

def classify_dicoms(dirName: str) -> Dict[str, List[Path]]:
    """
    Classify DICOM files in a directory by OperatingMode.

    Returns a dict with keys "B-Mode", "M-Mode", and "Unknown".
    """
    out: Dict[str, List[Path]] = {"B-Mode": [], "M-Mode": [], "Unknown": []}
    for file_path, mode, _err in scan_dicoms(dirName):
        if mode in ("B-Mode", "M-Mode"):
            out[mode].append(file_path)
        else:
            out["Unknown"].append(file_path)
    return out

def sort_modes(dirName):
    '''Function to sort all files within a directory into mmode and bmode subdirs'''
    
    # Get list of dcm files
    dcmFiles = [str(p) for p in Path(dirName).iterdir() if p.is_file() and p.suffix.lower() == ".dcm"]
    
    # Check each file's mode
    modes = []
    mmode_files = []
    bmode_files = []
    for file in dcmFiles:
        mode = get_mode(file)
        modes.append(mode or 'Undefined')
        if mode=='M-Mode':
            #Add just file name to mmode list
            mmode_files.append(os.path.basename(file))
        elif mode=='B-Mode':
            #Add just file name to bmmode list
            bmode_files.append(os.path.basename(file))
        else:
            #Do not analyze this file
            modes.append('Undefined')
    if len(mmode_files) + len(bmode_files) == 0:
        run_mode = 'None'
        return run_mode
                    
    
    # Check if multiple modes
    if len(np.unique(modes)) == 1:
        run_mode = modes[0]
    else:
        # Separate files into subfolders
        if 'M-Mode' in modes:
            move_files(dirName,mmode_files,'M-Mode')
        if 'B-Mode' in modes:
            move_files(dirName,bmode_files,'B-Mode')
        run_mode = 'Multiple'
        
    # Return info on which modes to run
    return run_mode

def write_command(inDir,runMode, testMode, verboseString):
    if testMode=='M-Mode':
        mainFunc = 'echoanalysis_mmode_main.py'
    elif testMode=='B-Mode':
        mainFunc = 'echoanalysis_main.py'
    else:
        return
    
    if runMode==testMode:
        subdir = inDir
        comm = ' & python ' + mainFunc + ' ' + verboseString + '--input_dir "' + subdir + '"'
    elif runMode=='Multiple' and os.path.isdir(os.path.join(inDir,testMode)):
        subdir = os.path.join(inDir,testMode)
        comm = ' & python ' + mainFunc + ' ' + verboseString + '--input_dir "' + subdir + '"'
    else:
        comm = ''
        
    return comm
