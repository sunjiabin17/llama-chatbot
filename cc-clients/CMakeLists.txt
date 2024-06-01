project(cc-clients LANGUAGES C CXX)

cmake_minimum_required(VERSION 3.18)

set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(TRITON_COMMON_ENABLE_PROTOBUF ON)
set(TRITON_COMMON_ENABLE_GRPC ON)

#
# Dependencies
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)

FetchContent_Declare(
repo-core
GIT_REPOSITORY https://github.com/triton-inference-server/core.git
GIT_TAG ${TRITON_CORE_REPO_TAG}
GIT_SHALLOW ON
)

FetchContent_MakeAvailable(repo-common)
FetchContent_MakeAvailable(repo-core)

# print
message(DEBUG "AAA $<TARGET_OBJECTS:grpc-service-library>")

#
# libcurl
#
find_package(CURL REQUIRED)
message(STATUS "Using curl ${CURL_VERSION}")

#
# Protobuf
#
set(protobuf_MODULE_COMPATIBLE TRUE CACHE BOOL "protobuf_MODULE_COMPATIBLE" FORCE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")
include_directories(${Protobuf_INCLUDE_DIRS})

#
# GRPC
#
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")
include_directories($<TARGET_PROPERTY:gRPC::grpc,INTERFACE_INCLUDE_DIRECTORIES>)


add_subdirectory(library)

#
# llama sync
#
add_executable(
    llama_sync_client
    src/llama_sync.cc
    $<TARGET_OBJECTS:json-utils-library>
)
target_link_libraries(
    llama_sync_client
    PRIVATE
    grpcclient_static
    httpclient_static
)
install(
    TARGETS llama_sync_client
    RUNTIME DESTINATION bin
)

#
# llama async
#
add_executable(
    llama_async_client
    src/llama_async.cc
    $<TARGET_OBJECTS:json-utils-library>
)
target_link_libraries(
    llama_async_client
    PRIVATE
    grpcclient_static
    httpclient_static
)
install(
    TARGETS llama_async_client
    RUNTIME DESTINATION bin
)

#
# llama stream
#
add_executable(
    llama1_client
    src/llama_stream.cc
    $<TARGET_OBJECTS:json-utils-library>
)
target_link_libraries(
    llama1_client
    PRIVATE
    grpcclient_static
    httpclient_static
)
install(
    TARGETS llama1_client
    RUNTIME DESTINATION bin
)

add_executable(
    llama2_client
    src/llama_stream2.cc
    $<TARGET_OBJECTS:json-utils-library>
)
target_link_libraries(
    llama2_client
    PRIVATE
    grpcclient_static
    httpclient_static
)
install(
    TARGETS llama2_client
    RUNTIME DESTINATION bin
)
