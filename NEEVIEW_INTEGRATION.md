# Anime4KCPP DLL Integration Guide for NeeView

## Overview
This guide explains how to integrate the compiled Anime4KCPP shared library (DLL/.so) with NeeView for image upscaling functionality.

## Build Output Structure

After compilation, you'll have:

```
install_shared/
├── include/
│   └── AC/
│       ├── Core.hpp
│       ├── Specs.hpp
│       ├── Core/
│       │   ├── Image.hpp
│       │   ├── Model.hpp
│       │   ├── Processor.hpp
│       │   ├── SIMD.hpp
│       │   └── Util.hpp
│       └── (other headers)
├── lib/
│   ├── libac.so (Linux)
│   ├── ac.dll (Windows)
│   └── ac.lib (Windows import library)
└── bin/
    └── (Windows DLL if built on Windows)
```

## C Binding Alternative (Recommended for Cross-Language Integration)

If you need a C interface instead of C++, use the C binding:

```
install_shared/
├── include/
│   └── AC/
│       ├── Core.h (C header)
│       ├── Error.h
│       └── Core/ (C++ headers for reference)
├── lib/
│   ├── libac_c.so (Linux)
│   └── ac_c.dll (Windows)
```

## Integration Steps

### 1. Copy Library Files

```bash
# Copy to NeeView's dependency folder
cp install_shared/lib/libac.so* /path/to/NeeView/dependencies/anime4k/lib/
```

### 2. Copy Header Files

```bash
# Copy C++ headers
cp -r install_shared/include/AC /path/to/NeeView/include/anime4k/

# Or for C binding
cp install_shared/include/AC/Core.h /path/to/NeeView/include/anime4k/
cp install_shared/include/AC/Error.h /path/to/NeeView/include/anime4k/
```

### 3. Update NeeView's CMakeLists.txt

Add to your NeeView project's CMakeLists.txt:

```cmake
# Find Anime4KCPP
find_library(ANIME4K_LIB ac PATHS /path/to/install_shared/lib)
add_library(Anime4K SHARED IMPORTED)
set_target_properties(Anime4K PROPERTIES
    IMPORTED_LOCATION "${ANIME4K_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "/path/to/install_shared/include"
)

# For C binding
# find_library(ANIME4K_C_LIB ac_c PATHS /path/to/install_shared/lib)
# ...

# Link to your target
target_link_libraries(your_neeview_target PRIVATE Anime4K)
```

### 4. Build and Test

```bash
cd /path/to/NeeView
mkdir build && cd build
cmake .. -DCMAKE_LIBRARY_PATH=/path/to/install_shared/lib
cmake --build .
```

## System Requirements for Runtime

### Linux (.so)
- libstdc++ (GCC runtime)
- OpenGL libraries (if rendering is used)
- Qt6 libraries (if GUI features are linked)

### Windows (.dll)
- Visual C++ Redistributable (if built with MSVC)
- OpenGL libraries
- Qt6 libraries

## Configuration Options Used

The provided build configurations compile with:
- ✅ Eigen3 support (for CPU optimization)
- ✅ AVX2 instructions (for modern CPUs)
- ✅ Fast math optimizations enabled
- ❌ OpenCL disabled (optional, requires SDK)
- ❌ CUDA disabled (optional, requires NVIDIA toolkit)

To enable these optional features, modify the CMakePresets or build script.

## Advanced Integration: Using pkg-config

Create a `.pc` file for easier integration:

```
# Create build/anime4k.pc
prefix=/path/to/install_shared
exec_prefix=${prefix}
libdir=${exec_prefix}/lib
includedir=${prefix}/include

Name: Anime4KCPP
Description: High performance anime upscaler
Version: 3.1.0
Cflags: -I${includedir}
Libs: -L${libdir} -lac
```

Then use in CMake:
```cmake
pkg_check_modules(ANIME4K anime4k)
target_link_libraries(your_target PRIVATE ${ANIME4K_LIBRARIES})
target_include_directories(your_target PRIVATE ${ANIME4K_INCLUDE_DIRS})
```

## C/C++ Usage Example

### C++ Example
```cpp
#include <AC/Core.hpp>

// Initialize processor
auto processor = std::make_unique<AC::Processor>(AC::ProcessorType::CPU);

// Create image
AC::Image image(width, height, AC::PixelType::RGB32F);
// ... load data ...

// Process
processor->upscale(image, scale_factor, model_type);
```

### C Example (if using C Binding)
```c
#include <AC/Core.h>

// Use C API
// ...
```

## Troubleshooting

### DLL/SO Not Found
- Ensure library is in system path or LD_LIBRARY_PATH
- Check file permissions: `chmod +x libac.so`

### Link Errors
- Verify header paths match include directory
- Check library architecture (32-bit vs 64-bit)
- Ensure dependency libraries are accessible

### Runtime Crashes
- Verify Anime4KCPP build matches target platform
- Check for missing runtime dependencies
- Enable debug symbols: rebuild without `-DCMAKE_BUILD_TYPE=Release`

## Building on Different Platforms

### Linux (GCC/Clang)
```bash
# Install dependencies
sudo apt install cmake build-essential libstdc++-dev

# Build
./build.sh
# or
cmake --preset shared-lib-release
cmake --build --preset shared-lib-release
```

### Windows (MSVC)
```powershell
# Install dependencies (Visual Studio, CMake)

# Build
cmake --preset shared-lib-release -G "Visual Studio 17 2022"
cmake --build --preset shared-lib-release --config Release
```

### Windows (MinGW)
```bash
cmake --preset shared-lib-release -G "MinGW Makefiles"
cmake --build --preset shared-lib-release
```

## Next Steps

1. Build using `./build.sh` or CMake presets
2. Copy output files to NeeView project
3. Update NeeView's build configuration
4. Test integration with sample images
5. Handle runtime dependencies (Qt, OpenGL, etc.)
