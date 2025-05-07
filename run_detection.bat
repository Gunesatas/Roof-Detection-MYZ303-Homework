@echo off
echo Roof Detection Screen Capture
echo ===========================
echo.

:: Set default values (no arguments needed)
set MODEL_PATH=
set WIDTH_RATIO=0.5
set CONF_THRESHOLD=0.25
set SLICED=false
set SLICE_HEIGHT=512
set SLICE_WIDTH=512
set OVERLAP_HEIGHT=0.2
set OVERLAP_WIDTH=0.2
set SCALE=1.0
set SKIP_FRAMES=0

:: Override with command line args if provided
if not "%1"=="" set MODEL_PATH=%1
if not "%2"=="" set WIDTH_RATIO=%2
if not "%3"=="" set CONF_THRESHOLD=%3
if not "%4"=="" set SLICED=%4
if not "%5"=="" set SLICE_HEIGHT=%5
if not "%6"=="" set SLICE_WIDTH=%6
if not "%7"=="" set OVERLAP_HEIGHT=%7
if not "%8"=="" set OVERLAP_WIDTH=%8
if not "%9"=="" set SCALE=%9

:: Activate conda environment
echo Trying direct activation of ultralytics environment...
call conda activate ultralytics-env 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Successfully activated ultralytics-env
    goto ENVIRONMENT_READY
)

:: If direct activation failed, try to find and activate conda environment
echo Direct activation failed, searching for Anaconda installation...
IF EXIST %USERPROFILE%\anaconda3\Scripts\activate.bat (
    call %USERPROFILE%\anaconda3\Scripts\activate.bat
    call conda activate ultralytics-env
) ELSE IF EXIST C:\ProgramData\Anaconda3\Scripts\activate.bat (
    call C:\ProgramData\Anaconda3\Scripts\activate.bat
    call conda activate ultralytics-env
) ELSE (
    echo No Anaconda installation found in common locations.
)

:ENVIRONMENT_READY
:: Verify ultralytics is installed
python -c "import ultralytics" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: ultralytics module is not available.
    echo Installing required packages...
    pip install ultralytics sahi
)

echo.
echo Running YOLO screen capture...
echo Press 'q' in the detection window to quit

:: Build command
set CMD=python yolo_screen_capture.py

if not "%MODEL_PATH%"=="" (
    set CMD=%CMD% --model-path %MODEL_PATH%
)

set CMD=%CMD% --width-ratio %WIDTH_RATIO% --conf %CONF_THRESHOLD% --scale %SCALE% --skip-frames %SKIP_FRAMES%

if "%SLICED%"=="true" (
    set CMD=%CMD% --sliced --slice-height %SLICE_HEIGHT% --slice-width %SLICE_WIDTH% --overlap-height %OVERLAP_HEIGHT% --overlap-width %OVERLAP_WIDTH%
)

:: Run the command
%CMD%

echo.
echo Detection stopped.
pause