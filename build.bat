@echo off
setlocal

:: Install required packages
pip install pyinstaller

:: Run the build script
python build_exe.py

:: Optional: Create a ZIP of the dist folder
cd dist
tar -a -c -f aphrodite.zip aphrodite
cd ..

echo Build complete! Check the dist/aphrodite folder for the executable and dependencies.