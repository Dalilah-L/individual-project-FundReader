@echo off
cd /d "%~dp0"

rem 优先使用上级目录里的虚拟环境，找不到再用系统 PATH 里的 python
set "PY=python"
if exist "..\.venv\Scripts\python.exe" set "PY=..\.venv\Scripts\python.exe"

echo.
echo   正在启动 FundReader 网页版...
echo   浏览器会自动打开 http://localhost:8501
echo   关闭这个黑色窗口 = 停止网页服务
echo.

%PY% -m streamlit run web_app.py --browser.gatherUsageStats=false

if errorlevel 1 (
    echo.
    echo   启动失败。请先在本目录执行: pip install -r requirements.txt
    echo.
    pause
)
