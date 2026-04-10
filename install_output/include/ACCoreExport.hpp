
#ifndef AC_CORE_EXPORT_H
#define AC_CORE_EXPORT_H

#ifdef AC_CORE_STATIC_DEFINE
#  define AC_CORE_EXPORT
#  define AC_CORE_NO_EXPORT
#else
#  ifndef AC_CORE_EXPORT
#    ifdef ac_EXPORTS
        /* We are building this library */
#      define AC_CORE_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define AC_CORE_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef AC_CORE_NO_EXPORT
#    define AC_CORE_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef AC_CORE_DEPRECATED
#  define AC_CORE_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef AC_CORE_DEPRECATED_EXPORT
#  define AC_CORE_DEPRECATED_EXPORT AC_CORE_EXPORT AC_CORE_DEPRECATED
#endif

#ifndef AC_CORE_DEPRECATED_NO_EXPORT
#  define AC_CORE_DEPRECATED_NO_EXPORT AC_CORE_NO_EXPORT AC_CORE_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef AC_CORE_NO_DEPRECATED
#    define AC_CORE_NO_DEPRECATED
#  endif
#endif

#endif /* AC_CORE_EXPORT_H */
