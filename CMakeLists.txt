cmake_minimum_required (VERSION 2.6)

# linux specific stuff

IF ( UNIX )
   set( CMAKE_CXX_FLAGS "-O3 -Wall -march=native ${CMAKE_CXX_FLAGS}" )
   set( CMAKE_PREFIX_PATH "/opt/opencv" )
   set( OPENMP_LINKER_FLAGS "-lgomp")
   set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)
   MESSAGE( "LINUX CONFIG" )
ENDIF ( UNIX )

# windows (inc. 64-bit specific stuff)

IF ( WIN32 )
   set( CMAKE_PREFIX_PATH "C:/OpenCV/build" )
   set( OpenCV_DIR "C:/OpenCV/build" )
   MESSAGE( "WINDOWS CONFIG" )
ENDIF ( WIN32 )

find_package( OpenCV REQUIRED )

project(omni)
add_executable(main main.cpp)
target_link_libraries( main ${OpenCV_LIBS} )

project(depth)
add_executable(depthmap depth_filter.cpp)
target_link_libraries( depthmap ${OpenCV_LIBS} )



