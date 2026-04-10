# Anime4KCPP + NeeView Integration Package

## Quick Start for NeeView Integration

This package provides everything needed to integrate Anime4KCPP with NeeView.

### Files Provided

```
Anime4KCPP/
├── build.sh                      # Linux/macOS build script
├── build.ps1                     # Windows PowerShell build script  
├── CMakePresets.json             # CMake preset configurations
├── NEEVIEW_INTEGRATION.md        # Detailed integration guide
├── ExampleNeeViewCMake.txt       # Example CMakeLists.txt snippet
├── cmake/
│   └── FindAnime4KCPP.cmake      # CMake find module for NeeView
├── install_shared/               # Build output (after compilation)
│   ├── include/AC/               # Header files
│   ├── lib/                      # Compiled library files (.so/.dll)
│   └── bin/                      # Windows DLL (if applicable)
└── [source files...]
```

## Step-by-Step Integration

### Step 1: Build Anime4KCPP

**On Linux/macOS:**
```bash
cd Anime4KCPP
chmod +x build.sh
./build.sh
```

**On Windows (PowerShell):**
```powershell
cd Anime4KCPP
.\build.ps1
# or for full features with OpenCL:
.\build.ps1 -Preset shared-lib-full
```

### Step 2: Verify Build Output

After successful build, verify the output:
```bash
ls install_shared/lib/        # Should contain libac.so, ac.dll, or ac.lib
ls install_shared/include/AC/ # Should contain headers
```

### Step 3: Set Up NeeView Integration

Choose one of these approaches:

#### Option A: System-wide Installation
```bash
# Copy to system paths
sudo cp -r install_shared/include/AC /usr/local/include/
sudo cp install_shared/lib/* /usr/local/lib/
sudo ldconfig  # On Linux, update library cache
```

Then in NeeView's CMakeLists.txt:
```cmake
find_package(Anime4KCPP REQUIRED)
target_link_libraries(neeview PRIVATE Anime4KCPP::Core)
```

#### Option B: Local Project Integration
```bash
# In NeeView project directory
cp -r ../Anime4KCPP/install_shared/include .
cp -r ../Anime4KCPP/install_shared/lib .
```

Then in NeeView's CMakeLists.txt:
```cmake
# Add this path to CMake module search
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../Anime4KCPP/install_shared")
find_package(Anime4KCPP REQUIRED)
target_link_libraries(neeview PRIVATE Anime4KCPP::Core)
```

#### Option C: Manual CMake Configuration
See [ExampleNeeViewCMake.txt](ExampleNeeViewCMake.txt) for detailed example.

### Step 4: Use in NeeView Code

```cpp
#include <AC/Core.hpp>

namespace AC {
    // Anime4KCPP namespace provides:
    // - Processor: Handles image upscaling
    // - Image: Image data structure
    // - Model: AI model types
    // - ProcessorType: CPU, GPU, etc.
}

// Example usage:
auto processor = std::make_unique<AC::Processor>(AC::ProcessorType::CPU);
AC::Image image(width, height, AC::PixelType::RGB32F);
// ... load image data ...
processor->upscale(image, 2, AC::ModelType::ACNetHD);
```

## Build Configuration Details

Default build includes:
- ✅ Core upscaling engine
- ✅ Eigen3 optimization
- ✅ AVX2 instruction support
- ✅ Fast math enabled
- ✅ C binding library (libac_c)
- ❌ OpenCL (disabled - add `-DAC_CORE_WITH_OPENCL=ON` if needed)
- ❌ CUDA (disabled - add `-DAC_CORE_WITH_CUDA=ON` if NVIDIA SDK available)

### Customizing Build

Edit CMakePresets.json or modify the build script to enable:

```cmake
# In CMakePresets.json or command line:
-DAC_CORE_WITH_OPENCL=ON   # Enable OpenCL GPU support
-DAC_CORE_WITH_CUDA=ON     # Enable CUDA GPU support  
-DAC_BUILD_VIDEO=ON        # Enable video processing
```

## Platform-Specific Notes

### Linux
- Output: `libac.so`, `libac_c.so`
- Runtime dependency: libstdc++
- Optional: OpenGL, Qt6 libraries for GUI features

### Windows (MSVC)
- Output: `ac.dll`, `ac.lib` (import library)
- Runtime dependency: Visual C++ Redistributable 2022+
- Use: Link against `ac.lib`, distribute `ac.dll`

### Windows (MinGW)
- Output: `libac.dll`, `libac.dll.a`
- Use as shared library (compatible with any compiler)

### macOS
- Output: `libac.dylib`
- Similar to Linux, may need code signing

## Troubleshooting

### Build Fails: OpenCL Not Found
**Solution:** Either install OpenCL SDK or add `-DAC_CORE_WITH_OPENCL=OFF` to build command

### Build Fails: CUDA Not Found
**Solution:** Either install CUDA Toolkit or add `-DAC_CORE_WITH_CUDA=OFF` to build command

### Runtime Error: Library Not Found
**Solution:**
- Linux: Add `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:path/to/install_shared/lib`
- Windows: Ensure DLL is in same directory as executable or in PATH
- Check File → Preferences → CMake in NeeView for path settings

### Symbol Not Found / Link Error
**Solution:**
- Verify library file exists in install_shared/lib/
- Ensure CMakeLists.txt correctly links against Anime4KCPP::Core
- Check for architecture mismatch (32-bit vs 64-bit)

## Deployment

Once integrated, ensure end users have:

### Linux
```bash
# All libraries in application directory or system library path
libac.so
libac.so.3  (if versioned)
```

### Windows
```
ac.dll          # In application directory or PATH
msvcp140.dll    # Visual C++ Runtime (often pre-installed)
vcruntime140.dll
```

### macOS
```  
libac.dylib
# May need code signing for distribution
codesign -s - libac.dylib
```

## Advanced: Environment Variables

For development and debugging:

```bash
# Linux: Add to library search path
export LD_LIBRARY_PATH="${PWD}/install_shared/lib:$LD_LIBRARY_PATH"

# Windows: Not needed if DLL in same directory
set PATH=%CD%\install_shared\bin;%PATH%

# macOS: Similar to Linux
export DYLD_LIBRARY_PATH="${PWD}/install_shared/lib:$DYLD_LIBRARY_PATH"
```

## Support and Further Integration

- **Anime4KCPP Repository:** https://github.com/TianZerL/Anime4KCPP
- **NeeView Repository:** Check NeeView's documentation for plugin architecture
- **C/C++ API Reference:** See `include/AC/Core.hpp` for full API documentation
- **Build Documentation:** See main [README.md](readme.md)

## License

Anime4KCPP is dual-licensed under:
- GPLv3 (see LICENSE-GPLv3)
- MIT (see LICENSE-MIT)

Choose the appropriate license for your NeeView integration.
