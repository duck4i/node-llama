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

#   MacOS build metal shaders
set(DGGML_NO_METAL ON CACHE BOOL "No metal library" FORCE)

# Add subdirectory for ggml and llama
add_subdirectory(ggml)
add_subdirectory(llama/src)

# Source files and library target
file (GLOB SOURCE_FILES "src/*.cpp")
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${CMAKE_JS_SRC})

set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")
target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB} ggml llama)

if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
    # Generate node.lib
    execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF} /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
endif()
