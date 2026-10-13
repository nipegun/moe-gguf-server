# Project version, the single source for the binary, the libraries and the web interface.
# Raise the patch number (0.1.1, 0.1.2, ...) with every change to the project.
set(MGS_VERSION "0.1.12")
set(BUILD_COMPILER "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")

if(CMAKE_VS_PLATFORM_NAME)
  set(BUILD_TARGET ${CMAKE_VS_PLATFORM_NAME})
else()
  set(BUILD_TARGET "${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
endif()
