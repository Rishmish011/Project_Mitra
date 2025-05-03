@echo off
echo Installing Multi-scale Adaptive Network (MAN) Epidemic Model...
echo.

:: Check for Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.7 or higher
    pause
    exit /b 1
)

:: Create a virtual environment
echo Creating virtual environment...
python -m venv man_epidemic_env
call man_epidemic_env\Scripts\activate.bat

:: Install required packages
echo Installing required packages...
pip install numpy pandas networkx matplotlib scipy seaborn tqdm

:: Copy application files
echo Copying application files...
mkdir man_epidemic_app
copy man_epidemic_model.py man_epidemic_app\
copy man_epidemic_gui.py man_epidemic_app\

:: Create launcher
echo Creating launcher...
echo @echo off > man_epidemic.bat
echo call man_epidemic_env\Scripts\activate.bat >> man_epidemic.bat
echo cd man_epidemic_app >> man_epidemic.bat
echo python man_epidemic_gui.py >> man_epidemic.bat
echo pause >> man_epidemic.bat

echo.
echo Installation complete!
echo Run man_epidemic.bat to start the application
pause