@echo off
cd /d "%~dp0"

:Isolation
call conda deactivate 2>NUL
set Path=%windir%\system32;%windir%;C:\Windows\System32\Wbem;%windir%\System32\WindowsPowerShell\v1.0\;%windir%\System32\OpenSSH\
SET CONDA_SHLVL=
SET PYTHONNOUSERSITE=1
SET PYTHONPATH=

Reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v "LongPathsEnabled" /t REG_DWORD /d "1" /f 2>nul
IF EXIST CONDA GOTO APP

:INSTALL
call update-runtime

:APP
call conda\condabin\activate.bat windows
%*
IF [%1] == [] TITLE Runtime Command Prompt && cmd /k