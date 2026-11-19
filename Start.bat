@echo off
rem facegen's one-click start on Windows (Phase 11). Double-click it.
rem
rem   1. If facegen is running already, show its window and stop here.
rem   2. Find uv (on PATH or in .bin\); else install it into .bin\ with Astral's installer.
rem   3. uv sync: Python 3.12 and the libraries (about 3 GB the first time, PyTorch most of it);
rem      packages a setup cut short left half-installed are installed again (venv_check.py).
rem   4. facegen launch: the prebuilt Rust extension (a failure is only a warning), then the
rem      model files: their size is shown and they are downloaded only after one "y".
rem   5. Start facegen's window with no console, and close this one: facegen-app.exe (the
rem      venv's pythonw.exe under facegen's name and icon, made by step 4), else pythonw.exe.
rem
rem   Start.bat --shortcut   also puts a facegen shortcut on the desktop and in the Start Menu.
rem   Start.bat --update     first waits for facegen to quit and puts the newest release over
rem                          this folder (src/facegen/setup/update.py; data\ is kept), then
rem                          goes on as above with the new files. facegen's Update now runs it.
rem
rem The decisions live in Python (src/facegen/setup/launch.py and update.py), where they are
rem tested; this file only runs the steps. A log is kept in data\logs\start.log. Errors stay
rem on screen.

setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%~dp0"
set "LOG=%ROOT%data\logs\start.log"
set "PY=%ROOT%.venv\Scripts\python.exe"
set "PYW=%ROOT%.venv\Scripts\pythonw.exe"
set "APP=%ROOT%.venv\Scripts\facegen-app.exe"
set "SHORTCUT="
set "UPDATE="
if /i "%~1"=="--shortcut" set "SHORTCUT=1"
if /i "%~1"=="--update" set "UPDATE=1"
title facegen
if not exist "%ROOT%data\logs" mkdir "%ROOT%data\logs"
call :log "Start.bat %*"

rem --- 1. facegen running already? ---------------------------------------------------------
rem Recover interrupted file replacement before importing the installed Python package.
if exist "%ROOT%data\update-recovery\pending.json" (
  "%PY%" -B "%ROOT%data\update-recovery\repair.py" "%ROOT%."
  if errorlevel 1 goto pause_and_fail
)
if defined UPDATE goto update
:ordinary_start
if exist "%ROOT%.git" goto launch_ready
rem Keep a read handle through setup and launch. An updater needs exclusive access.
if not exist "%ROOT%data\install.lock" type nul >>"%ROOT%data\install.lock"
set "ENTERED="
call :launch_locked 9<"%ROOT%data\install.lock"
if not defined ENTERED (
  echo Cannot start facegen while its installation is in use. Try again after the update.
  exit /b 1
)
exit /b %ERRORLEVEL%

:launch_locked
set "ENTERED=1"
setlocal
:launch_ready
if exist "%ROOT%data\update-recovery\pending.json" (
  echo An update was interrupted. Start facegen again to complete the repair.
  goto pause_and_fail
)
if not exist "%PY%" goto find_uv
"%PY%" -m facegen.cli launch --open-running --log "%LOG%"
set "CODE=%ERRORLEVEL%"
if "%CODE%"=="0" goto done
if "%CODE%"=="4" goto fail_running
goto find_uv

rem --- 1b. Start.bat --update ------------------------------------------------------------
:update
rem Not set up yet: set up the files here first; the update can come later.
if not exist "%PY%" goto ordinary_start
echo Updating facegen. This window waits for facegen to quit, downloads the newest release
echo and replaces facegen's files; your pictures, settings and data folder are kept.
call :log "update"
rem cmd reads a batch file line by line as it runs it, and the update replaces this file. So
rem the rest runs from this block, which cmd has read in full: the update, then the new
rem Start.bat (from its first line), which sets up the new version's libraries and starts it.
(
  "%PY%" -m facegen.cli update --log "%LOG%"
  if errorlevel 1 pause
  endlocal
  call "%~f0"
  exit /b
)

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
"%UV%" sync --locked --inexact --no-group native --no-group dev
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
"%UV%" sync --locked --inexact --no-group native --no-group dev %REPAIR%
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
if not exist "%APP%" set "APP=%PYW%"
start "" "%APP%" -m facegen.desktop app
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
