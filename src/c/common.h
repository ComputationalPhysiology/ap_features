#ifndef _AP_FEATURES_COMMON_H
#define _AP_FEATURES_COMMON_H

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
// OS is Windows
#define CDECL __cdecl
#define _OS_WINDOWS
#else
#define CDECL
#endif

#endif // defined(_AP_FEATURES_COMMON_H)
