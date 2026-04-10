# FindAnime4KCPP.cmake
# This module defines the following variables:
#   ANIME4KCPP_FOUND - True if Anime4KCPP is found
#   ANIME4KCPP_INCLUDE_DIR - The include directory
#   ANIME4KCPP_LIBRARY - The library to link against
#   ANIME4KCPP::Core - Imported target for linking

find_package(Anime4KCPP CONFIG QUIET)

if(NOT Anime4KCPP_FOUND)
    find_path(ANIME4KCPP_INCLUDE_DIR
        NAMES AC/Core.hpp AC/Core.h
        HINTS
            ${ANIME4KCPP_HOME}/include
            ${CMAKE_PREFIX_PATH}
            /opt/anime4k/include
            /usr/include
            /usr/local/include
        PATH_SUFFIXES anime4k
    )

    find_library(ANIME4KCPP_LIBRARY
        NAMES ac
        HINTS
            ${ANIME4KCPP_HOME}/lib
            ${CMAKE_PREFIX_PATH}
            /opt/anime4k/lib
            /usr/lib
            /usr/local/lib
        PATH_SUFFIXES lib lib64
    )

    find_library(ANIME4KCPP_LIBRARY_C
        NAMES ac_c
        HINTS
            ${ANIME4KCPP_HOME}/lib
            ${CMAKE_PREFIX_PATH}
            /opt/anime4k/lib
            /usr/lib
            /usr/local/lib
        PATH_SUFFIXES lib lib64
    )

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(Anime4KCPP
        REQUIRED_VARS ANIME4KCPP_INCLUDE_DIR ANIME4KCPP_LIBRARY
        VERSION_VAR ANIME4KCPP_VERSION
    )

    if(ANIME4KCPP_FOUND)
        if(NOT TARGET Anime4KCPP::Core)
            add_library(Anime4KCPP::Core SHARED IMPORTED)
            set_target_properties(Anime4KCPP::Core PROPERTIES
                IMPORTED_LOCATION "${ANIME4KCPP_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${ANIME4KCPP_INCLUDE_DIR}"
            )

            # For Windows, also need the import library
            if(WIN32 AND EXISTS "${CMAKE_FIND_PACKAGE_NAME_CORE_LIBRARY}")
                set_target_properties(Anime4KCPP::Core PROPERTIES
                    IMPORTED_IMPLIB "${CMAKE_FIND_PACKAGE_NAME_CORE_LIBRARY}"
                )
            endif()
        endif()

        if(ANIME4KCPP_LIBRARY_C AND NOT TARGET Anime4KCPP::C)
            add_library(Anime4KCPP::C SHARED IMPORTED)
            set_target_properties(Anime4KCPP::C PROPERTIES
                IMPORTED_LOCATION "${ANIME4KCPP_LIBRARY_C}"
                INTERFACE_INCLUDE_DIRECTORIES "${ANIME4KCPP_INCLUDE_DIR}"
            )
        endif()

        mark_as_advanced(ANIME4KCPP_INCLUDE_DIR ANIME4KCPP_LIBRARY ANIME4KCPP_LIBRARY_C)
    endif()
endif()
