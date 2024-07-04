# Create virtual environment and activate by cmd
python -m venv myenv
myenv\Scripts\activate.bat

# Open Pyqt5 designer
myenv\Lib\site-packages\qt5_applications\Qt\bin\designer.exe

# Convert file .ui to .py
pyuic5 -x MainWindow.ui -o MainWindow.py