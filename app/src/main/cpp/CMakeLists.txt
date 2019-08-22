# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

# configure import libs
set(distribution_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../distribution)

add_library(lib_tensorflowLite SHARED IMPORTED)
set_target_properties(lib_tensorflowLite PROPERTIES IMPORTED_LOCATION
	${distribution_DIR}/tensorflowLite/lib/${ANDROID_ABI}/libtensorflowLite.so)

# build application's shared lib
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
             native-lib SHARED
             DLDTDelegate.cpp
             UseDelegate.cpp
	         native-lib.cpp )

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
	          log )

set(pathToTensorflowLite /home/anishak/tensor/tensorflow)
set(pathToFlatBuff /home/anishak/tensor/tensorflow/third_party/flatbuffers/flatbuffers/include)

target_include_directories(native-lib PRIVATE
                           ${pathToTensorflowLite}
    	                   ${pathToFlatBuff})

target_link_libraries(native-lib
                      ${log-lib}
                      android
		              lib_tensorflowLite)