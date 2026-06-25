@echo off
setlocal

set "ROOT=%~dp0"
set "NATIVE_DIR=%ROOT%image_processor\native_viewer"
set "APP=%NATIVE_DIR%\build\femtonics_image_processor.exe"

if not exist "%APP%" (
    echo Native image processor UI is not built yet; attempting native build...
    echo.
    call "%ROOT%tools\native_dev_env.bat" > nul
    if errorlevel 1 goto build_failed

    cmake -S "%NATIVE_DIR%" -B "%NATIVE_DIR%\build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="%QT_ROOT%" -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows
    if errorlevel 1 goto build_failed

    cmake --build "%NATIVE_DIR%\build"
    if errorlevel 1 goto build_failed
)

if exist "%APP%" (
    "%APP%" %*
    exit /b %ERRORLEVEL%
)

goto build_failed

:build_failed
echo.
echo Native image processor UI could not be built.
echo Expected executable:
echo   %APP%
echo.
echo You can retry manually from this folder with:
echo   call tools\native_dev_env.bat
echo   cmake -S image_processor\native_viewer -B image_processor\native_viewer\build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=%%QT_ROOT%% -DCMAKE_TOOLCHAIN_FILE=%%VCPKG_ROOT%%\scripts\buildsystems\vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows
echo   cmake --build image_processor\native_viewer\build
exit /b 1
