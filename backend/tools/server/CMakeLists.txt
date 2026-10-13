include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by mgs and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
  server-chat.cpp
  server-chat.h
  server-task.cpp
  server-task.h
  server-queue.cpp
  server-queue.h
  server-common.cpp
  server-common.h
  server-context.cpp
  server-context.h
  server-stream.cpp
  server-stream.h
  server-tools.cpp
  server-tools.h
  server-mcp.cpp
  server-mcp.h
  server-shell-mcp.cpp
  server-shell-mcp.h
  server-schema.cpp
  server-schema.h
  server-monitor.cpp
  server-monitor.h
  server-ui-settings.cpp
  server-ui-settings.h
)

if (BUILD_SHARED_LIBS)
  set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC mgs-common mtmd ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})

# mgs-server-impl: server logic, reusable by app

set(TARGET mgs-server-impl)

file(READ "${PROJECT_SOURCE_DIR}/vendor/swagger-ui/swagger-ui-bundle.js" vSwaggerScript)
file(READ "${PROJECT_SOURCE_DIR}/vendor/swagger-ui/swagger-ui.css" vSwaggerStyle)
file(READ "${PROJECT_SOURCE_DIR}/../frontend/api.html" vSwaggerPage)
configure_file(server-doc-assets.h.in server-doc-assets.h @ONLY)

add_library(${TARGET}
  server.cpp
  server-http.cpp
  server-http.h
  server-tls.cpp
  server-tls.h
  server-api-doc.cpp
  server-api-doc.h
  server-models.cpp
  server-models.h
)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ../mtmd ${PROJECT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC server-context mgs-ui cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

if(MGS_TOOLS_INSTALL)
  install(TARGETS ${TARGET} LIBRARY)
endif()

# mgs executable

set(TARGET mgs)

add_executable(${TARGET} main.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME mgs)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE mgs-server-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
