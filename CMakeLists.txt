cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SlicerPhotogrammetry)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/SlicerMorph/SlicerPhotogrammetry")
set(EXTENSION_CONTRIBUTORS "Oshane Thomas (SCRI), Murat Maga (SCRI)")
set(EXTENSION_DESCRIPTION "Photogrammetry module allows preprocessing (e.g., background removal) and submission of photographs to Open Drone Map program for 3D model creation.")
set(EXTENSION_ICONURL "https://github.com/SlicerMorph/SlicerPhotogrammetry/blob/master/SlicerPhotogrammetry/Resources/Icons/Photogrammetry.png")
set(EXTENSION_SCREENSHOTURLS "https://github.com/SlicerMorph/SlicerPhotogrammetry/blob/master/SlicerPhotogrammetry/Resources/Icons/Photogrammetry.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SlicerPhotogrammetry)

## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
