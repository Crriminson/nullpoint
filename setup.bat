@echo off
rem ============================================================================
rem  Nullpoint one-click setup for Windows 10/11
rem  Double-click this file. It will:
rem    1. install Git and Python 3.12 with winget if they are missing
rem    2. download Nullpoint into %USERPROFILE%\nullpoint (or use the folder this file is in)
rem    3. create a private Python environment and install everything
rem    4. start Nullpoint in your browser
rem  Run it again any time to update. Afterwards, start Nullpoint with run.bat.
rem ============================================================================
setlocal EnableExtensions
title Nullpoint setup
set "REPO=https://github.com/Crriminson/nullpoint.git"
set "DEST=%USERPROFILE%\nullpoint"
set "VENV=%LOCALAPPDATA%\nullpoint\venv"
echo.
echo   Nullpoint setup
echo   ---------------
echo.

rem --- use this folder if setup.bat sits inside a copy of the project -------------
if exist "%~dp0nullpoint\__main__.py" set "DEST=%~dp0"
if "%DEST:~-1%"=="\" set "DEST=%DEST:~0,-1%"

rem --- Git ---------------------------------------------------------------------
where git >nul 2>&1
if errorlevel 1 (
  if exist "%ProgramFiles%\Git\cmd\git.exe" (
    set "PATH=%ProgramFiles%\Git\cmd;%PATH%"
  ) else (
    echo [1/4] Installing Git with winget...
    winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
    set "PATH=%ProgramFiles%\Git\cmd;%PATH%"
  )
)
where git >nul 2>&1 || goto :nogit

rem --- Python 3.12 ---------------------------------------------------------------
set "PY="
py -3.12 --version >nul 2>&1 && set "PY=py -3.12"
if not defined PY (
  echo [2/4] Installing Python 3.12 with winget...
  winget install --id Python.Python.3.12 -e --source winget --accept-package-agreements --accept-source-agreements
  py -3.12 --version >nul 2>&1 && set "PY=py -3.12"
)
if not defined PY goto :nopython

rem --- download or update the project --------------------------------------------
if exist "%DEST%\nullpoint\__main__.py" (
  echo [3/4] Project found in %DEST%
  if exist "%DEST%\.git" git -C "%DEST%" pull --ff-only
) else (
  echo [3/4] Downloading Nullpoint into %DEST% ...
  git clone --depth 1 "%REPO%" "%DEST%" || goto :noclone
)

rem --- environment and packages (kept outside OneDrive, which breaks venvs) -------
if not exist "%VENV%\Scripts\python.exe" (
  echo [4/4] Creating the Python environment...
  %PY% -m venv "%VENV%" || goto :novenv
)
set "VPY=%VENV%\Scripts\python.exe"
echo       Installing packages (a few minutes the first time)...
"%VPY%" -m pip install --upgrade pip --quiet
"%VPY%" -m pip install -r "%DEST%\requirements.txt" --quiet || goto :nopip
"%VPY%" -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu --quiet || echo       (PyTorch skipped: only the slow-motion analysis needs it)
"%VPY%" -m pip install -e "%DEST%" --quiet || goto :nopip

echo.
echo   Done. Nullpoint is starting in your browser.
echo   Next time, double-click run.bat in %DEST%
echo   No webcam? Run:  run.bat --sim
echo.
if defined NULLPOINT_NO_RUN goto :eof
cd /d "%DEST%"
"%VPY%" -m nullpoint %*
goto :eof

:nogit
echo Could not install Git automatically. Install it from https://git-scm.com/download/win and run this again.
pause & exit /b 1
:nopython
echo Could not find Python 3.12. Install it from https://www.python.org/downloads/ (tick "Add to PATH") and run this again.
pause & exit /b 1
:noclone
echo Could not download the project from %REPO%. Check your internet connection.
pause & exit /b 1
:novenv
echo Could not create the Python environment in %VENV%.
pause & exit /b 1
:nopip
echo Package installation failed. Scroll up for the error, then run setup.bat again.
pause & exit /b 1
