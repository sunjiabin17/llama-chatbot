project(llama-chatbot LANGUAGES C CXX)

cmake_minimum_required(VERSION 3.18)

option(TRITON_ENABLE_CC_HTTP "Build C++ HTTP client libraries" ON)
option(TRITON_ENABLE_CC_GRPC "Build C++ GRPC client libraries" ON)

set(TRITON_THIRD_PARTY_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/third_party repo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#
include(FetchContent)

FetchContent_Declare(
  repo-third-party
  GIT_REPOSITORY https://github.com/triton-inference-server/third_party.git
  GIT_TAG ${TRITON_THIRD_PARTY_REPO_TAG}
  GIT_SHALLOW ON
)

set(TRITON_THIRD_PARTY_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/third-party)
FetchContent_MakeAvailable(repo-third-party)

set(LIB_DIR "lib")

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(TRITON_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/cc-clients/install)
else()
  set(TRITON_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()

include(ExternalProject)

ExternalProject_Add(cc-clients
    PREFIX cc-clients
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cc-clients
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/cc-clients
    CMAKE_CACHE_ARGS
        -DTRITON_THIRD_PARTY_INSTALL_PREFIX:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}
        -DTRITON_ENABLE_CC_HTTP:BOOL=${TRITON_ENABLE_CC_HTTP}
        -DTRITON_ENABLE_CC_GRPC:BOOL=${TRITON_ENABLE_CC_GRPC}
        -DCURL_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/curl/${LIB_DIR}/cmake/CURL
        -DProtobuf_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/protobuf/${LIB_DIR}/cmake/protobuf
        -DgRPC_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/grpc/lib/cmake/grpc
        -Dabsl_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/absl/${LIB_DIR}/cmake/absl
        -Dre2_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/re2/${LIB_DIR}/cmake/re2
        -Dc-ares_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/c-ares/${LIB_DIR}/cmake/c-ares

        -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_INSTALL_PREFIX}
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    DEPENDS
        re2 curl grpc protobuf
)

  