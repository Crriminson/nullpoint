@echo off
rem Start Nullpoint (after setup.bat). Extra options pass through, e.g.  run.bat --sim   or   run.bat --lan
setlocal
set "VPY=%LOCALAPPDATA%\nullpoint\venv\Scripts\python.exe"
if not exist "%VPY%" (
  echo Nullpoint is not installed yet. Running setup.bat first...
  call "%~dp0setup.bat" %*
  exit /b
)
cd /d "%~dp0"
"%VPY%" -m nullpoint %*
