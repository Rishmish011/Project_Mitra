# MAMEP (Man-made Epidemic Predictor) Installation Guide

This document provides step-by-step instructions for installing the MAMEP application on both Windows and macOS systems.

## System Requirements

- Python (the installer will check and install if needed)
- 500MB free disk space
- Internet connection (required for downloading dependencies)

## Installation Files

The following files are included in your download:

1. **man_epidemic_model.py** - The core simulation model
2. **man_epidemic_gui.py** - The graphical user interface
3. **install_windows.bat** - Windows installer script
4. **install_mac.sh** - macOS installer script

## Windows Installation

1. **Prepare Installation Files**
   - Save all three required files to the same folder:
     - `man_epidemic_model.py`
     - `man_epidemic_gui.py`
     - `install_windows.bat`

2. **Run the Installer**
   - Double-click on `install_windows.bat`
   - Alternatively, open Command Prompt, navigate to the folder, and run:
     ```
     install_windows.bat
     ```
   - The installer will:
     - Check for Python and install if needed
     - Create a virtual environment to isolate dependencies
     - Install all required Python packages
     - Set up a launcher script for easy startup

3. **Launch the Application**
   - After installation completes, use `man_epidemic.bat` in the same folder to start the application
   - You can create a shortcut to this file on your desktop for convenience

## macOS Installation

1. **Prepare Installation Files**
   - Save all three required files to the same folder:
     - `man_epidemic_model.py`
     - `man_epidemic_gui.py`
     - `install_mac.sh`

2. **Run the Installer**
   - Open Terminal
   - Navigate to the folder containing the files:
     ```
     cd /path/to/your/folder
     ```
   - Make the installer executable:
     ```
     chmod +x install_mac.sh
     ```
   - Run the installer:
     ```
     ./install_mac.sh
     ```
   - The installer will:
     - Check for Python and install if needed
     - Create a virtual environment to isolate dependencies
     - Install all required Python packages
     - Set up a launcher script for easy startup

3. **Launch the Application**
   - After installation completes, use `man_epidemic.command` in the same folder to start the application
   - You can create an alias to this file on your desktop for convenience

## Troubleshooting

If you encounter any issues during installation:

- Ensure you have administrator privileges on your system
- Check your internet connection is active
- Make sure all three files are in the same directory
- For Windows: Try running Command Prompt as administrator
- For macOS: Ensure the script has executable permissions

## Support

For additional assistance, please contact technical support at support@example.com
