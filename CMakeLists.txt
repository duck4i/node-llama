cmake_minimum_required(VERSION 3.14)
project(npm-llama)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CMAKE-JS
cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0042 NEW)
add_definitions(-DNAPI_VERSION=7)

# Clean any existing CMake cache
if(EXISTS "${CMAKE_BINARY_DIR}/CMakeCache.txt")
    file(REMOVE "${CMAKE_BINARY_DIR}/CMakeCache.txt")
endif()

# Create and set permissions for runtime output directory before GGML tries to use it
if(APPLE)
    # First ensure the directory exists with proper permissions
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    execute_process(
        COMMAND chmod -R 777 "${CMAKE_BINARY_DIR}/bin"
        RESULT_VARIABLE CHMOD_RESULT
    )
    if(NOT CHMOD_RESULT EQUAL "0")
        message(FATAL_ERROR "Failed to set permissions on ${CMAKE_BINARY_DIR}/bin")
    endif()

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

# Configure Metal options
if(APPLE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "Embed metal library" FORCE)
    #set(GGML_METAL_USE_BF16 ON CACHE BOOL "Enable BF16 support" FORCE)
endif()

# Create build directories with proper permissions
execute_process(
    COMMAND mkdir -p "${CMAKE_BINARY_DIR}/Release"
    COMMAND chmod -R 755 "${CMAKE_BINARY_DIR}"
    COMMAND chmod -R 755 "${CMAKE_BINARY_DIR}/Release"
)

# Include directories for headers
include_directories(BEFORE ${CMAKE_JS_INC})
include_directories(
    ${CMAKE_SOURCE_DIR}/node_modules/nan
    ${CMAKE_SOURCE_DIR}/node_modules/node-addon-api
    ${CMAKE_SOURCE_DIR}/include
)

# Add subdirectory for ggml and llama
add_subdirectory(ggml)
add_subdirectory(llama/src)

# Source files and library target
file(GLOB SOURCE_FILES "src/*.cpp")
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${CMAKE_JS_SRC})

set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")
target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB} ggml llama)

if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
    execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF} /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
endif()

# Ensure output directories remain writable
add_custom_command(
    TARGET ${PROJECT_NAME}
    POST_BUILD
    COMMAND chmod -R 755 "${CMAKE_BINARY_DIR}/Release"
)
