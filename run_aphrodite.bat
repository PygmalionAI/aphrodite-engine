@echo off
setlocal

:: Set the PYTHONPATH to include the current directory
set PYTHONPATH=%~dp0

:: Set the event loop policy and other Python settings
set PYTHONASYNCIODEBUG=1
set PYTHONDEVMODE=1
set PYTHONASYNCIO_EVENT_LOOP_POLICY=WindowsSelectorEventLoopPolicy

:: Run Aphrodite
dist\aphrodite\aphrodite.exe run EleutherAI/pythia-70m-deduped -gmu 0.5

pause