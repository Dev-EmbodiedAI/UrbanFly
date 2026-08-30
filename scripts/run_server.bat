@echo off
REM UrbanFly 后端服务器启动脚本
cd /d %~dp0..

echo Installing Python dependencies...
pip install -r requirements.txt -q

echo Starting UrbanFly server...
python -c "import sys; sys.path.insert(0,'.'); from backend.server.server import main; main()"
pause
