cmake_minimum_required(VERSION 3.14)
project(npm-llama)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CMAKE-JS
cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0042 NEW)
add_definitions(-DNAPI_VERSION=7)

# Include directories for headers
include_directories(BEFORE ${CMAKE_JS_INC})
include_directories(
    ${CMAKE_SOURCE_DIR}/node_modules/nan
    ${CMAKE_SOURCE_DIR}/node_modules/node-addon-api
    ${CMAKE_SOURCE_DIR}/include
)

# Ensure build directories are writable
if(UNIX)
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/build")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/build/Release")
    execute_process(
        COMMAND chmod -R 755 "${CMAKE_BINARY_DIR}/build"
        COMMAND chmod -R 755 "${CMAKE_BINARY_DIR}/build/Release"
    )
endif()

# Set GGML options before adding subdirectory
set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "Embed metal library" FORCE)

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

# Ensure output directories are writable post-build
if(UNIX)
    add_custom_target(ensure_writable_directories ALL
        COMMAND chmod -R 755 "${CMAKE_BINARY_DIR}/build"
        COMMAND chmod -R 755 "${CMAKE_BINARY_DIR}/build/Release"
        COMMAND ${CMAKE_COMMAND} -E echo "Directories made writable"
    )
    add_dependencies(${PROJECT_NAME} ensure_writable_directories)
endif()
