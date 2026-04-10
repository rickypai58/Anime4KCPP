
#ifndef AC_C_EXPORT_H
#define AC_C_EXPORT_H

#ifdef AC_C_STATIC_DEFINE
#  define AC_C_EXPORT
#  define AC_C_NO_EXPORT
#else
#  ifndef AC_C_EXPORT
#    ifdef ac_c_EXPORTS
        /* We are building this library */
#      define AC_C_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define AC_C_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef AC_C_NO_EXPORT
#    define AC_C_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef AC_C_DEPRECATED
#  define AC_C_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef AC_C_DEPRECATED_EXPORT
#  define AC_C_DEPRECATED_EXPORT AC_C_EXPORT AC_C_DEPRECATED
#endif

#ifndef AC_C_DEPRECATED_NO_EXPORT
#  define AC_C_DEPRECATED_NO_EXPORT AC_C_NO_EXPORT AC_C_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef AC_C_NO_DEPRECATED
#    define AC_C_NO_DEPRECATED
#  endif
#endif

#endif /* AC_C_EXPORT_H */
