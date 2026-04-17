# Stream V2 consumer libraries and tools (included from repository root CMakeLists.txt).

option(DECTRIS_USE_BUNDLED_LIBTIFF "Build libtiff from third_party/libtiff (git submodule)" ON)

# Application sources live next to this directory under src/ (repository root).
set(DECTRIS_APP_SRC_DIR "${CMAKE_SOURCE_DIR}/src")

include(CheckCCompilerFlag)
find_package(Threads REQUIRED)

# OpenCV must find OpenCVConfig.cmake. On Windows CMake does not guess the install path: set
# OpenCV_DIR to the directory that contains that file (official pack: often .../build or
# .../build/x64/vc16 - match your MSVC version). Environment variable OpenCV_DIR is picked up here
# if the cache entry is empty (setx does not apply to the shell you run cmake from).
if(NOT OpenCV_DIR)
    if(DEFINED ENV{OpenCV_DIR} AND NOT "$ENV{OpenCV_DIR}" STREQUAL "")
        file(TO_CMAKE_PATH "$ENV{OpenCV_DIR}" _dectris_opencv_dir_env)
        set(OpenCV_DIR "${_dectris_opencv_dir_env}" CACHE PATH "Directory containing OpenCVConfig.cmake")
    endif()
endif()

find_package(OpenCV QUIET COMPONENTS core imgcodecs imgproc photo)

# Official Windows pack: OpenCVConfig.cmake often lives under build/x64/vcNN, not build/.
if(NOT OpenCV_FOUND AND OpenCV_DIR AND EXISTS "${OpenCV_DIR}" AND NOT EXISTS "${OpenCV_DIR}/OpenCVConfig.cmake")
    file(GLOB _dectris_opencv_cfgs "${OpenCV_DIR}/x64/vc*/OpenCVConfig.cmake")
    if(_dectris_opencv_cfgs)
        list(SORT _dectris_opencv_cfgs ORDER DESCENDING)
        list(GET _dectris_opencv_cfgs 0 _dectris_opencv_cfg)
        get_filename_component(_dectris_opencv_sub "${_dectris_opencv_cfg}" DIRECTORY)
        message(STATUS "OpenCV: using ${_dectris_opencv_sub} (OpenCVConfig.cmake under build/x64/vc*)")
        set(OpenCV_DIR "${_dectris_opencv_sub}" CACHE PATH "Directory containing OpenCVConfig.cmake" FORCE)
        find_package(OpenCV QUIET COMPONENTS core imgcodecs imgproc photo)
    endif()
endif()

if(NOT WIN32)
    option(BUILD_LIBZMQ "Build ZeroMQ library from source" OFF)
else()
    option(BUILD_LIBZMQ "Build ZeroMQ library from source" ON)
endif()

if(BUILD_LIBZMQ)
    include(FetchContent)

    # CMake 4.0+ removed compatibility with cmake_minimum_required < 3.5
    # libzmq v4.3.4 uses an older version, so we need this workaround
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0")
        set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    endif()

    # Set options before fetching
    set(BUILD_SHARED OFF CACHE INTERNAL "")
    set(BUILD_STATIC ON CACHE INTERNAL "")
    set(ENABLE_CURVE OFF CACHE BOOL "")
    set(ENABLE_DRAFTS OFF CACHE BOOL "")

    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.28")
        # CMake 3.28+ supports EXCLUDE_FROM_ALL in FetchContent_Declare
        FetchContent_Declare(
            libzmq
            GIT_REPOSITORY https://github.com/zeromq/libzmq.git
            GIT_TAG 4097855ddaaa65ed7b5e8cb86d143842a594eebd # v4.3.4
            EXCLUDE_FROM_ALL
            )
        FetchContent_MakeAvailable(libzmq)
    else()
        # Older CMake requires manual FetchContent_Populate + add_subdirectory
        FetchContent_Declare(
            libzmq
            GIT_REPOSITORY https://github.com/zeromq/libzmq.git
            GIT_TAG 4097855ddaaa65ed7b5e8cb86d143842a594eebd # v4.3.4
            )
        FetchContent_GetProperties(libzmq)
        if(NOT libzmq_POPULATED)
            FetchContent_Populate(libzmq)
            add_subdirectory("${libzmq_SOURCE_DIR}" "${libzmq_BINARY_DIR}" EXCLUDE_FROM_ALL)
        endif()
    endif()

    unset(ZEROMQ_LIBRARY CACHE)
    set(LIBZMQ_TARGET libzmq-static)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(PC_LIBZMQ QUIET libzmq)
    # Use NAMES zmq (not libzmq.so): CMake adds the lib prefix / correct suffix per platform.
    find_library(ZeroMQ_LIBRARY
        NAMES zmq libzmq
        PATHS ${PC_LIBZMQ_LIBDIR} ${PC_LIBZMQ_LIBRARY_DIRS}
    )
    if(NOT ZeroMQ_LIBRARY)
        message(FATAL_ERROR
            "ZeroMQ library not found. On Debian/Ubuntu install: sudo apt install libzmq3-dev "
            "(provides libzmq.so and libzmq.pc; package libzmq5 alone is runtime-only).")
    endif()
    set(ZMQ_INCLUDE_DIRS ${PC_LIBZMQ_INCLUDE_DIRS})
    if(NOT ZMQ_INCLUDE_DIRS)
        find_path(ZMQ_INCLUDE_DIR zmq.h)
        if(ZMQ_INCLUDE_DIR)
            set(ZMQ_INCLUDE_DIRS ${ZMQ_INCLUDE_DIR})
        endif()
    endif()
    if(NOT ZMQ_INCLUDE_DIRS)
        message(FATAL_ERROR
            "ZeroMQ headers (zmq.h) not found. On Debian/Ubuntu install: sudo apt install libzmq3-dev")
    endif()
    add_library(libzmq SHARED IMPORTED)
    set_property(TARGET libzmq PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${ZMQ_INCLUDE_DIRS})
    set_property(TARGET libzmq PROPERTY IMPORTED_LOCATION ${ZeroMQ_LIBRARY})
    set(LIBZMQ_TARGET libzmq)
endif()

add_subdirectory(third_party EXCLUDE_FROM_ALL)

set(TIFF_FOUND FALSE)
if(TARGET TIFF::TIFF)
    set(TIFF_FOUND TRUE)
elseif(DECTRIS_USE_BUNDLED_LIBTIFF)
    message(WARNING
        "Bundled libtiff requested but target TIFF::TIFF is missing. "
        "Initialize the submodule: git submodule update --init src/data_consumer/third_party/libtiff")
endif()
if(NOT TIFF_FOUND)
    find_package(TIFF)
    if(NOT TIFF_FOUND)
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(PC_LIBTIFF QUIET libtiff-4)
            if(NOT PC_LIBTIFF_FOUND)
                pkg_check_modules(PC_LIBTIFF QUIET libtiff)
            endif()
            if(PC_LIBTIFF_FOUND)
                set(TIFF_FOUND TRUE)
                set(TIFF_LIBRARIES ${PC_LIBTIFF_LIBRARIES})
                set(TIFF_INCLUDE_DIRS ${PC_LIBTIFF_INCLUDE_DIRS})
            endif()
        endif()
    endif()
    # FindTIFF / pkg-config often miss distro layouts; headers need libtiff-dev (not just libtiff6).
    if(NOT TIFF_FOUND)
        find_path(TIFF_MANUAL_INCLUDE_DIR NAMES tiff.h)
        find_library(TIFF_MANUAL_LIBRARY NAMES tiff libtiff libtiff5 tiffd libtiffd)
        if(TIFF_MANUAL_INCLUDE_DIR AND TIFF_MANUAL_LIBRARY)
            set(TIFF_FOUND TRUE)
            if(NOT TARGET TIFF::TIFF)
                add_library(TIFF::TIFF UNKNOWN IMPORTED)
                set_target_properties(TIFF::TIFF PROPERTIES
                    IMPORTED_LOCATION "${TIFF_MANUAL_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "${TIFF_MANUAL_INCLUDE_DIR}")
            endif()
        endif()
    endif()
endif()

set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
)

if(NOT "${CMAKE_C_COMPILER_ID}" MATCHES "MSVC")
    check_c_compiler_flag("-mf16c" COMPILER_SUPPORTS_MF16C)
    if(COMPILER_SUPPORTS_MF16C)
        add_compile_options(-mf16c)
    endif()
    add_compile_options(-Wall -Wextra)
endif()

add_library(stream2 STATIC
    stream2.c
    stream2.h
)
target_link_libraries(stream2 PRIVATE
    tinycbor
    )

# Shared helper library for examples
add_library(stream2_helpers STATIC
    stream2_common.c
    stream2_common.h
    stream2_stats.c
    stream2_stats.h
    stream2_decompress.c
    stream2_decompress.h
    stream2_image_buffer.c
    stream2_image_buffer.h
    stream2_buffer_decode_stack.c
    stream2_buffer_decode_stack.h
    tiff_writer.c
    tiff_writer.h
)
target_link_libraries(stream2_helpers PUBLIC
    ${LIBZMQ_TARGET}
    compression
    stream2
    tinycbor
    Threads::Threads
    )

add_executable(stream2_generic_receiver "${DECTRIS_APP_SRC_DIR}/stream2_generic_receiver.c")
target_link_libraries(stream2_generic_receiver stream2_helpers)

# Linux-only (pthread): skip on Windows — use DectrisStream2Demo_windows or WSL for similar workflows.
if(NOT WIN32)
    add_executable(DectrisStream2Receiver_linux "${DECTRIS_APP_SRC_DIR}/DectrisStream2Receiver_linux.c")
    target_link_libraries(DectrisStream2Receiver_linux stream2_helpers)
    add_executable(acquire_and_save_stream "${DECTRIS_APP_SRC_DIR}/acquire_and_save_stream.c")
    target_link_libraries(acquire_and_save_stream stream2_helpers)
endif()

macro(dectris_link_libtiff _name)
    target_compile_features(${_name} PRIVATE cxx_std_17)
    if(TARGET TIFF::TIFF)
        target_link_libraries(${_name} PRIVATE TIFF::TIFF)
    else()
        if(DEFINED TIFF_INCLUDE_DIRS AND TIFF_INCLUDE_DIRS)
            target_include_directories(${_name} PRIVATE ${TIFF_INCLUDE_DIRS})
        elseif(DEFINED TIFF_INCLUDE_DIR AND TIFF_INCLUDE_DIR)
            target_include_directories(${_name} PRIVATE ${TIFF_INCLUDE_DIR})
        endif()
        target_link_libraries(${_name} PRIVATE ${TIFF_LIBRARIES})
    endif()
endmacro()

if(TIFF_FOUND)
    set(DECTRIS_LIBTIFF_WARNING_FILTER tiff_warning_filter.cpp)
    add_executable(data_analysis_pipeline_example
        "${DECTRIS_APP_SRC_DIR}/data_analysis_pipeline_example.cpp"
        ${DECTRIS_LIBTIFF_WARNING_FILTER})
    dectris_link_libtiff(data_analysis_pipeline_example)
    add_executable(average_images
        "${DECTRIS_APP_SRC_DIR}/average_images.cpp"
        ${DECTRIS_LIBTIFF_WARNING_FILTER})
    dectris_link_libtiff(average_images)
endif()

if(OpenCV_FOUND)
    find_package(OpenMP QUIET COMPONENTS CXX)

    add_executable(normalize_images "${DECTRIS_APP_SRC_DIR}/normalize_images.cpp")
    target_compile_features(normalize_images PRIVATE cxx_std_17)
    if(TARGET opencv_core AND TARGET opencv_imgcodecs AND TARGET opencv_imgproc)
        target_link_libraries(normalize_images PRIVATE
            opencv_core opencv_imgcodecs opencv_imgproc)
    else()
        target_include_directories(normalize_images PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(normalize_images PRIVATE ${OpenCV_LIBS})
    endif()

    add_executable(inpaint_tiff
        "${DECTRIS_APP_SRC_DIR}/inpaint_images.cpp"
        "${DECTRIS_APP_SRC_DIR}/linear_inpaint.cpp")
    target_compile_features(inpaint_tiff PRIVATE cxx_std_17)
    if(TARGET opencv_core AND TARGET opencv_imgcodecs AND TARGET opencv_imgproc
            AND TARGET opencv_photo)
        target_link_libraries(inpaint_tiff PRIVATE
            opencv_core opencv_imgcodecs opencv_imgproc opencv_photo)
    else()
        target_include_directories(inpaint_tiff PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(inpaint_tiff PRIVATE ${OpenCV_LIBS})
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(inpaint_tiff PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # Optional LaMa CLI: LibTorch + OpenCV. Download the C++ LibTorch zip from
    # pytorch.org (Release, matching your CUDA/CPU), then configure e.g.:
    #   cmake -DTorch_DIR=C:/libtorch/share/cmake/Torch ...
    find_package(Torch QUIET)
    if(Torch_FOUND)
        add_library(lama_inpaint_lib OBJECT "${DECTRIS_APP_SRC_DIR}/lama_inpaint_minimal.cpp")
        target_compile_features(lama_inpaint_lib PRIVATE cxx_std_17)
        target_compile_definitions(lama_inpaint_lib PRIVATE WITH_LAMA_INPAINT=1)
        if(MSVC)
            target_compile_options(lama_inpaint_lib PRIVATE /wd4267)
        endif()
        if(TARGET Torch::Torch)
            target_link_libraries(lama_inpaint_lib PUBLIC Torch::Torch)
        else()
            target_link_libraries(lama_inpaint_lib PUBLIC ${TORCH_LIBRARIES})
        endif()
        if(TARGET opencv_core AND TARGET opencv_imgcodecs AND TARGET opencv_imgproc)
            target_link_libraries(lama_inpaint_lib PUBLIC
                opencv_core opencv_imgcodecs opencv_imgproc)
        else()
            target_include_directories(lama_inpaint_lib PUBLIC ${OpenCV_INCLUDE_DIRS})
            target_link_libraries(lama_inpaint_lib PUBLIC ${OpenCV_LIBS})
        endif()

        add_executable(lama_inpaint
            "${DECTRIS_APP_SRC_DIR}/lama_inpaint_minimal.cpp"
            "${DECTRIS_APP_SRC_DIR}/lama_inpaint.cpp")
        target_compile_features(lama_inpaint PRIVATE cxx_std_17)
        target_compile_definitions(lama_inpaint PRIVATE WITH_LAMA_INPAINT=1)
        # C4267 (size_t → int) is triggered inside LibTorch/STL headers on MSVC x64;
        # the link step and exe are still correct.
        if(MSVC)
            target_compile_options(lama_inpaint PRIVATE /wd4267)
        endif()
        if(TARGET Torch::Torch)
            target_link_libraries(lama_inpaint PRIVATE Torch::Torch)
        else()
            target_link_libraries(lama_inpaint PRIVATE ${TORCH_LIBRARIES})
        endif()
        if(TARGET opencv_core AND TARGET opencv_imgcodecs AND TARGET opencv_imgproc)
            target_link_libraries(lama_inpaint PRIVATE
                opencv_core opencv_imgcodecs opencv_imgproc)
        else()
            target_include_directories(lama_inpaint PRIVATE ${OpenCV_INCLUDE_DIRS})
            target_link_libraries(lama_inpaint PRIVATE ${OpenCV_LIBS})
        endif()

        target_sources(inpaint_tiff PRIVATE $<TARGET_OBJECTS:lama_inpaint_lib>)
        target_compile_definitions(inpaint_tiff PRIVATE WITH_LAMA_INPAINT=1)
        if(MSVC)
            target_compile_options(inpaint_tiff PRIVATE /wd4267)
        endif()
        if(TARGET Torch::Torch)
            target_link_libraries(inpaint_tiff PRIVATE Torch::Torch)
        else()
            target_link_libraries(inpaint_tiff PRIVATE ${TORCH_LIBRARIES})
        endif()

        if(WIN32)
            get_filename_component(_lama_torch_root "${Torch_DIR}" DIRECTORY)
            get_filename_component(_lama_torch_root "${_lama_torch_root}" DIRECTORY)
            get_filename_component(_lama_torch_root "${_lama_torch_root}" DIRECTORY)
            file(GLOB _lama_torch_dlls "${_lama_torch_root}/lib/*.dll")
            if(_lama_torch_dlls)
                add_custom_command(TARGET lama_inpaint POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${_lama_torch_dlls} $<TARGET_FILE_DIR:lama_inpaint>
                    COMMENT "Copying LibTorch DLLs next to lama_inpaint.exe")
                add_custom_command(TARGET inpaint_tiff POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${_lama_torch_dlls} $<TARGET_FILE_DIR:inpaint_tiff>
                    COMMENT "Copying LibTorch DLLs next to inpaint_tiff.exe")
            endif()
        endif()
    endif()
endif()

if(WIN32)
    add_executable(DectrisStream2Demo_windows "${DECTRIS_APP_SRC_DIR}/DectrisStream2Demo_windows.c")
    target_link_libraries(DectrisStream2Demo_windows
        stream2_helpers
        user32
        gdi32
        comctl32
        iphlpapi
        ws2_32
    )
endif()

if(NOT TIFF_FOUND)
    message(WARNING
        "LibTIFF not found: data_analysis_pipeline_example and average_images will not be built. "
        "Use the third_party/libtiff submodule (git submodule update --init) or "
        "on Debian/Ubuntu: sudo apt install libtiff-dev")
endif()

if(NOT OpenCV_FOUND)
    if(WIN32)
        message(WARNING
            "OpenCV (core, imgcodecs, imgproc, photo) not found: normalize_images and inpaint_tiff "
            "will not be built. Pass the folder that contains OpenCVConfig.cmake, e.g. "
            "cmake -DOpenCV_DIR=C:/path/to/opencv/build/x64/vc17 (vc* must match your VS). "
            "Or set OpenCV_DIR for this PowerShell session: $env:OpenCV_DIR='C:\\...'; cmake ... "
            "(setx does not affect the current window). With vcpkg, use that toolchain file and install opencv4. If OpenCV_DIR was wrong before, clear it: cmake -U OpenCV_DIR -B <builddir> -S .")
    else()
        message(WARNING
            "OpenCV (core, imgcodecs, imgproc, photo) not found: normalize_images and inpaint_tiff "
            "will not be built. On Debian/Ubuntu: sudo apt install libopencv-dev")
    endif()
endif()

# Add clean-build target (works on all platforms, especially useful for Windows/Visual Studio)
# Usage: cmake --build . --target clean-build
# Note: For Makefile generators, 'make clean' already exists and works. This target
# provides a way to clean everything including CMake cache files, binaries, and intermediate files.
add_custom_target(clean-build
    COMMAND ${CMAKE_COMMAND} -E echo "Cleaning build artifacts..."
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_SOURCE_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_SOURCE_DIR}/lib
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_BINARY_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_BINARY_DIR}/lib
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_BINARY_DIR}/CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E rm -f ${CMAKE_BINARY_DIR}/CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -E rm -f ${CMAKE_BINARY_DIR}/cmake_install.cmake
    COMMAND ${CMAKE_COMMAND} -E rm -f ${CMAKE_BINARY_DIR}/Makefile
    COMMAND ${CMAKE_COMMAND} -E rm -f ${CMAKE_BINARY_DIR}/compile_commands.json
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_BINARY_DIR}/_deps
    COMMENT "Cleaning all build artifacts, binaries, intermediate files, and CMake cache"
    VERBATIM
)

# Note: The standard 'clean' target is provided by CMake automatically.
# Use 'cmake --build . --target clean' for standard clean, or
# 'cmake --build . --target clean-build' for deep clean (removes everything).
