@echo off
echo Installing API dependencies...
pip install -r requirements_api.txt

echo.
echo Starting Python API server...
echo API will be available at: http://localhost:5000
echo.
py -3.11 api_server.py