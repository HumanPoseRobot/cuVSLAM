# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

# Adapt the pinned dependency before project(cunls) selects its CUDA toolkit.
# An override source and a downloaded source must obey the same parent compiler.
if(NOT DEFINED CUNLS_SOURCE_DIR)
    message(FATAL_ERROR "CUNLS_SOURCE_DIR is required")
endif()
set(_cunls_cmake "${CUNLS_SOURCE_DIR}/CMakeLists.txt")
file(READ "${_cunls_cmake}" _original)
string(REPLACE "cmake_minimum_required(VERSION 3.24)" "cmake_minimum_required(VERSION 3.22)" _adapted "${_original}")
string(REGEX REPLACE "set\\(CMAKE_CUDA_COMPILER [^\n]*\\)\n" "# Inherit the parent CUDA compiler.\n" _adapted "${_adapted}")
string(REGEX REPLACE "\nset\\(CMAKE_CUDA_ARCHITECTURES ([^\n]*)\\)" "\nif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)\n  set(CMAKE_CUDA_ARCHITECTURES \\1)\nendif()" _adapted "${_adapted}")
if(NOT _adapted STREQUAL _original)
    file(WRITE "${_cunls_cmake}" "${_adapted}")
endif()
