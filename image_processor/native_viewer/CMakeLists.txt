cmake_minimum_required(VERSION 3.24)

project(femtonics_image_processor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)
find_package(TIFF REQUIRED)

add_executable(femtonics_image_processor
    src/FrameCache.cpp
    src/FrameCache.h
    src/MainWindow.cpp
    src/MainWindow.h
    src/NativeMetadata.cpp
    src/NativeMetadata.h
    src/TiffStack.cpp
    src/TiffStack.h
    src/TiffViewerWidget.cpp
    src/TiffViewerWidget.h
    src/main.cpp
)

target_link_libraries(femtonics_image_processor
    PRIVATE
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
        TIFF::TIFF
)

target_compile_definitions(femtonics_image_processor
    PRIVATE
        QT_NO_CAST_FROM_ASCII
        QT_NO_CAST_TO_ASCII
)

if(WIN32 AND TARGET Qt6::windeployqt)
    add_custom_command(TARGET femtonics_image_processor POST_BUILD
        COMMAND Qt6::windeployqt --no-translations --no-compiler-runtime "$<TARGET_FILE:femtonics_image_processor>"
        COMMENT "Deploying Qt runtime files"
    )
endif()

if(WIN32)
    add_custom_command(TARGET femtonics_image_processor POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E rm -f "$<TARGET_FILE_DIR:femtonics_image_processor>/spike_tiff_viewer.exe"
        COMMENT "Removing obsolete compatibility launcher name"
    )
endif()
