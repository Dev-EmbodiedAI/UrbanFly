@echo off
setlocal
cd /d D:\AI\UrbanFly
start "UrbanFly VLN Server" /min python -m http.server 8765 --bind 127.0.0.1
timeout /t 2 /nobreak >nul
start "" http://127.0.0.1:8765/demo/vln_dashboard/index.html
echo UrbanFly VLN demo: http://127.0.0.1:8765/demo/vln_dashboard/index.html
exit /b 0
