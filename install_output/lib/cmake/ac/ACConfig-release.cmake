#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "AC::Core" for configuration "Release"
set_property(TARGET AC::Core APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(AC::Core PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libac.so"
  IMPORTED_SONAME_RELEASE "libac.so"
  )

list(APPEND _cmake_import_check_targets AC::Core )
list(APPEND _cmake_import_check_files_for_AC::Core "${_IMPORT_PREFIX}/lib/libac.so" )

# Import target "AC::Binding::C" for configuration "Release"
set_property(TARGET AC::Binding::C APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(AC::Binding::C PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "AC::Core"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libac_c.so"
  IMPORTED_SONAME_RELEASE "libac_c.so"
  )

list(APPEND _cmake_import_check_targets AC::Binding::C )
list(APPEND _cmake_import_check_files_for_AC::Binding::C "${_IMPORT_PREFIX}/lib/libac_c.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
