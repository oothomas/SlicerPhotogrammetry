cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)
project(Photogrammetry)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/SlicerMorph/SlicerPhotogrammetry")
set(EXTENSION_CONTRIBUTORS "Oshane Thomas (SCRI), Murat Maga (SCRI)")
set(EXTENSION_DESCRIPTION "An extension to preprocess (masking) large collection photographs which then can be processed to construct 3D models with texture.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/SlicerMorph/SlicerPhotogrammetry/master/SlicerPhotogrammetry.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/SlicerMorph/SlicerPhotogrammetry/master/screenshot.png https://raw.githubusercontent.com/SlicerMorph/SlicerPhotogrammetry/master/screenshot_2.png")
set(EXTENSION_DEPENDS "SlicerMorph PyTorch")

#-----------------------------------------------------------------------------
if(APPLE OR WIN32)
  message(FATAL_ERROR "SlicerPhotogrammetry is only supported on Linux. Skipping build on this platform.")
endif()

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(Photogrammetry)
add_subdirectory(ClusterPhotos)
add_subdirectory(SamuraiVideoMasking)

## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
