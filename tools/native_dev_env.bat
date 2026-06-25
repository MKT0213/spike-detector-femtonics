@echo off

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64

set "QT_ROOT=%USERPROFILE%\Qt\6.8.3\msvc2022_64"
set "VCPKG_ROOT=%USERPROFILE%\vcpkg"
set "CMAKE_PREFIX_PATH=%QT_ROOT%;%CMAKE_PREFIX_PATH%"
set "PATH=%QT_ROOT%\bin;%VCPKG_ROOT%;%VCPKG_ROOT%\installed\x64-windows\bin;%PATH%"

echo.
echo Native C++/Qt environment ready.
echo   QT_ROOT=%QT_ROOT%
echo   VCPKG_ROOT=%VCPKG_ROOT%
echo   CMake toolchain: %VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake
echo.
echo Example CMake configure:
echo   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="%QT_ROOT%" -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows
echo.
