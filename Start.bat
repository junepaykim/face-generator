@echo off
rem facegen's one-click start on Windows (Phase 11). Double-click it.
rem
rem   1. If facegen is running already, show its window and stop here.
rem   2. Find uv (on PATH or in .bin\); else install it into .bin\ with Astral's installer.
rem   3. uv sync: Python 3.12 and the libraries (about 3 GB the first time, PyTorch most of it);
rem      packages a setup cut short left half-installed are installed again (venv_check.py).
rem   4. facegen launch: the prebuilt Rust extension (a failure is only a warning), then the
rem      model files: their size is shown and they are downloaded only after one "y".
rem   5. Start facegen's window with pythonw.exe (no console), and close this one.
rem
rem   Start.bat --shortcut   also puts a facegen shortcut on the desktop and in the Start Menu.
rem
rem The decisions live in Python (src/facegen/launch.py), where they are tested; this file
rem only runs the steps. A log is kept in data\logs\start.log. Errors stay on screen.

setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%~dp0"
set "LOG=%ROOT%data\logs\start.log"
set "PY=%ROOT%.venv\Scripts\python.exe"
set "PYW=%ROOT%.venv\Scripts\pythonw.exe"
set "SHORTCUT="
if /i "%~1"=="--shortcut" set "SHORTCUT=1"
title facegen
if not exist "%ROOT%data\logs" mkdir "%ROOT%data\logs"
call :log "Start.bat %*"

rem --- 1. facegen running already? ---------------------------------------------------------
if not exist "%PY%" goto find_uv
"%PY%" -m facegen.cli launch --open-running --log "%LOG%"
set "CODE=%ERRORLEVEL%"
if "%CODE%"=="0" goto done
if "%CODE%"=="4" goto fail_running

rem --- 2. uv --------------------------------------------------------------------------------
:find_uv
set "UV="
for /f "delims=" %%i in ('where uv 2^>nul') do if not defined UV set "UV=%%i"
if not defined UV if exist "%ROOT%.bin\uv.exe" set "UV=%ROOT%.bin\uv.exe"
if not defined UV if exist "%ROOT%.bin\bin\uv.exe" set "UV=%ROOT%.bin\bin\uv.exe"
if defined UV goto have_uv
echo Installing uv, the tool that sets up Python for facegen, into %ROOT%.bin ...
call :log "installing uv into .bin"
set "UV_INSTALL_DIR=%ROOT%.bin"
set "UV_NO_MODIFY_PATH=1"
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; irm https://astral.sh/uv/0.10.0/install.ps1 | iex"
if exist "%ROOT%.bin\uv.exe" set "UV=%ROOT%.bin\uv.exe"
if not defined UV if exist "%ROOT%.bin\bin\uv.exe" set "UV=%ROOT%.bin\bin\uv.exe"
if not defined UV goto fail_uv
:have_uv
call :log "uv: %UV%"

rem --- 3. Python and the libraries ----------------------------------------------------------
echo Getting Python and facegen's libraries ready. The first time this downloads about 3 GB
echo (PyTorch with CUDA is most of it) and takes a while; later starts take seconds.
"%UV%" sync --locked --inexact --no-group native
set "CODE=%ERRORLEVEL%"
call :log "uv sync: exit %CODE%"
if not "%CODE%"=="0" goto fail_sync
if not exist "%PYW%" goto fail_sync

rem A setup cut short (the console closed while uv unpacked a package) leaves the package
rem half-installed, and uv sync alone does not notice: install those again.
set "REPAIR="
for /f "delims=" %%a in ('call "%PY%" -m facegen.venv_check') do set "REPAIR=%%a"
if not defined REPAIR goto setup_done
echo An earlier setup was cut short; installing again: %REPAIR%
call :log "repair: %REPAIR%"
"%UV%" sync --locked --inexact --no-group native %REPAIR%
set "CODE=%ERRORLEVEL%"
call :log "uv sync (repair): exit %CODE%"
if not "%CODE%"=="0" goto fail_sync
set "REPAIR="
for /f "delims=" %%a in ('call "%PY%" -m facegen.venv_check') do set "REPAIR=%%a"
if defined REPAIR goto fail_sync
:setup_done

rem --- 4. the Rust extension and the models ----------------------------------------------------
"%PY%" -m facegen.cli launch --log "%LOG%"
set "CODE=%ERRORLEVEL%"
call :log "facegen launch: exit %CODE%"
if not "%CODE%"=="0" goto fail_launch
if not defined SHORTCUT goto start_app
"%PY%" -m facegen.cli shortcuts
call :log "facegen shortcuts: exit %ERRORLEVEL%"

rem --- 5. the window ---------------------------------------------------------------------------
:start_app
call :log "starting facegen app"
start "" "%PYW%" -m facegen.desktop app
goto done

:fail_running
call :log "facegen is running but did not answer"
echo.
echo facegen is running already but did not answer. If it is still starting, wait a moment
echo and start it again; if it seems stuck, end it in Task Manager.
goto pause_and_fail

:fail_uv
call :log "uv could not be installed"
echo.
echo uv could not be installed. Check the internet connection and start facegen again, or
echo install uv yourself: https://docs.astral.sh/uv/getting-started/installation/
goto pause_and_fail

:fail_sync
echo.
echo Python and facegen's libraries could not be set up (the messages above say why).
echo Check the internet connection and the free disk space, then start facegen again.
goto pause_and_fail

:fail_launch
echo.
echo facegen could not get ready (the messages above say why).
goto pause_and_fail

:pause_and_fail
echo The log is in %LOG%
echo.
pause
endlocal
exit /b 1

:done
endlocal
exit /b 0

:log
>>"%LOG%" echo %DATE% %TIME% %~1
exit /b 0
