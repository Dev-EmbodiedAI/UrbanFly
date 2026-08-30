@echo off
REM UrbanFly 前端开发服务器启动脚本
cd /d %~dp0..\frontend

echo Installing npm packages...
call npm install

echo Starting Vite dev server...
call npx vite --host --port 5173
pause
