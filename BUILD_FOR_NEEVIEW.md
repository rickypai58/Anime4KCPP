# Anime4KCPP → NeeView 集成方案

**編譯日期:** 2026-04-10  
**Anime4KCPP 版本:** 3.1.0  
**輸出格式:** DLL/SO + Header Files

---

## 📦 已準備文件清單

| 文件 | 說明 |
|------|------|
| `build.sh` | Linux/macOS 自動編譯腳本 |
| `build.ps1` | Windows PowerShell 編譯腳本 |
| `CMakePresets.json` | CMake 預設配置（支援多種配置） |
| `cmake/FindAnime4KCPP.cmake` | CMake 搜尋模組 (供 NeeView 使用) |
| `NEEVIEW_PACKAGE.md` | 完整集成指南（必讀） |
| `NEEVIEW_INTEGRATION.md` | 詳細技術集成文件 |
| `ExampleNeeViewCMake.txt` | CMakeLists.txt 整合示例 |

---

## 🚀 快速開始（三步驟）

### 1️⃣ 編譯 Anime4KCPP (構建 DLL/SO)

**Linux/macOS:**
```bash
cd /workspaces/Anime4KCPP
chmod +x build.sh
./build.sh
```

**Windows (PowerShell):**
```powershell
cd \workspaces\Anime4KCPP
.\build.ps1
```

### 2️⃣ 驗證編譯輸出

```bash
# 檢查動態庫
ls install_shared/lib/          # 應包含 libac.so / ac.dll

# 檢查頭文件  
ls install_shared/include/AC/   # 應包含 Core.hpp, Specs.hpp...
```

### 3️⃣ 複製到 NeeView

```bash
# 假設 NeeView 在相鄰目錄
cp -r install_shared/lib/*        ../NeeView/dependencies/lib/
cp -r install_shared/include/AC   ../NeeView/dependencies/include/
```

---

## 🔧 編譯配置詳情

### 已啟用功能（預設版本）
- ✅ **Core 引擎** - 主要的圖像放大功能
- ✅ **Eigen3 優化** - CPU 加速計算
- ✅ **AVX2 支援** - 向量指令集優化
- ✅ **Fast Math** - 數學計算優化
- ✅ **C 語言綁定** (libac_c) - 跨語言支援
- ❌ OpenCL (可選，需要 SDK)
- ❌ CUDA (可選，需要 NVIDIA 工具包)

### 修改配置
編輯 `CMakePresets.json` 中的 `cacheVariables` 或修改 build 腳本：

```cmake
# 啟用 OpenCL GPU 支援
-DAC_CORE_WITH_OPENCL=ON

# 啟用 CUDA GPU 支援  
-DAC_CORE_WITH_CUDA=ON

# 啟用視頻處理
-DAC_BUILD_VIDEO=ON
```

---

## 📂 輸出檔案結構

編譯後生成：

```
install_shared/
├── include/AC/
│   ├── Core.hpp              # 主頭文件
│   ├── Core/
│   │   ├── Image.hpp         # 圖像類
│   │   ├── Processor.hpp     # 處理器類
│   │   ├── Model.hpp         # 模型定義
│   │   ├── SIMD.hpp          # SIMD 指令
│   │   └── ...
│   ├── Specs.hpp             # 規格定義
│   └── ...
├── lib/
│   ├── libac.so (Linux)
│   ├── ac.dll (Windows)
│   ├── ac.lib (Windows import lib)
│   ├── libac_c.so (Linux C binding)
│   └── ac_c.dll (Windows C binding)
└── bin/ (Windows)
    └── ac.dll (重新定位副本)
```

---

## 💻 NeeView 集成方式

### 方式 A：使用 CMake find_package（推薦）

在 NeeView 的 `CMakeLists.txt` 中：

```cmake
# 指定 Anime4KCPP 安裝路徑
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../Anime4KCPP/install_shared")

# 或使用提供的 FindAnime4KCPP 模組
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../Anime4KCPP/cmake")

# 搜尋並鏈接
find_package(Anime4KCPP REQUIRED)
target_link_libraries(neeview PRIVATE Anime4KCPP::Core)
```

### 方式 B：手動配置

```cmake
# 設定路徑
set(ANIME4KCPP_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Anime4KCPP/install_shared/include")
set(ANIME4KCPP_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Anime4KCPP/install_shared/lib")

# 鏈接
target_include_directories(neeview PRIVATE ${ANIME4KCPP_INCLUDE_DIR})
target_link_directories(neeview PRIVATE ${ANIME4KCPP_LIB_DIR})
target_link_libraries(neeview PRIVATE ac)
```

詳見 `ExampleNeeViewCMake.txt` 檔案。

---

## 🔨 平台特定注意事項

### Windows (MSVC)
- **DLL 輸出:** `install_shared/bin/ac.dll` 或 `install_shared/lib/ac.dll`
- **Import Library:** `install_shared/lib/ac.lib` (CMake 會自動處理)
- **運行時:** 需要 Visual C++ Redistributable 2022+
- **編譯器:** Visual Studio 17 2022 或更新版本

### Linux
- **SO 輸出:** `install_shared/lib/libac.so` (或 `libac.so.3.1.0`)
- **運行時:** 需要 libstdc++ (通常已安裝)
- **編譯器:** GCC 11+ 或 Clang 13+
- **環境變數:** `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:path/to/lib`

### macOS  
- **Dylib 輸出:** `install_shared/lib/libac.dylib`
- **簽署:** `codesign -s - libac.dylib` (發佈用)
- **編譯器:** Apple Clang (Xcode Command Line Tools)

---

## 🎯 使用示例 (C++ 代碼)

```cpp
#include <AC/Core.hpp>
#include <memory>

// 初始化處理器
auto processor = std::make_unique<AC::Processor>(
    AC::ProcessorType::CPU  // 或使用 GPU 如果已編譯
);

// 建立圖像
AC::Image image(width, height, AC::PixelType::RGB32F);

// 加載圖像數據
// image.data() = ...

// 執行放大 (2x 放大, ACNet HD 模型)
processor->upscale(image, 2, AC::ModelType::ACNetHD);

// 取得結果
// image.data() 現在包含放大後的數據
```

---

## 🐛 常見問題解決

| 問題 | 解決方案 |
|------|---------|
| **"libac.so not found"** | 設定 `LD_LIBRARY_PATH` 或新增至 `/usr/lib` |
| **"openCL not found" 編譯失敗** | 新增 `-DAC_CORE_WITH_OPENCL=OFF` 參數 |
| **"CUDA not found" 編譯失敗** | 新增 `-DAC_CORE_WITH_CUDA=OFF` 參數 |
| **CMake 找不到 Anime4KCPP** | 設定 `CMAKE_PREFIX_PATH` 或複製 `FindAnime4KCPP.cmake` |
| **Windows DLL 執行時錯誤** | 檢查 VC++ Redistributable 安裝，或複製 DLL 到執行目錄 |

---

## 📄 相關文件說明

| 文件 | 用途 |
|------|------|
| `NEEVIEW_PACKAGE.md` | **主要整合指南** - 詳細步驟和部署說明 |
| `NEEVIEW_INTEGRATION.md` | 技術細節 - API 使用、C/C++ 綁定等 |
| `ExampleNeeViewCMake.txt` | CMake 範例 - 複製粘貼可用的配置片段 |
| `cmake/FindAnime4KCPP.cmake` | CMake 模組 - 提供 `find_package()` 支援 |
| `CMakePresets.json` | 編譯預設 - 多種配置選項 |

---

## 🎬 完整工作流 (參考)

```bash
# 1. 編譯構建
cd Anime4KCPP
./build.sh

# 2. 驗證輸出
file install_shared/lib/libac.so*
ls -lh install_shared/include/AC/

# 3. 複製文件到 NeeView
cp -r install_shared/include/AC /path/to/NeeView/dependencies/
cp install_shared/lib/libac.so* /path/to/NeeView/dependencies/lib/

# 4. 更新 NeeView CMakeLists.txt
# (參考 ExampleNeeViewCMake.txt)

# 5. 編譯 NeeView
cd /path/to/NeeView
mkdir build && cd build
cmake .. -DAC_PATH=/path/to/Anime4KCPP/install_shared
cmake --build . -j$(nproc)

# 6. 測試
./neeview
```

---

## ⚖️ 授權

Anime4KCPP 採用雙重授權：
- **GPLv3** (見 `LICENSE-GPLv3`)
- **MIT** (見 `LICENSE-MIT`)

在與 NeeView 整合時，請遵守相應的授權條款。

---

## 📞 進階支援

- **Anime4KCPP 官方倉庫:** https://github.com/TianZerL/Anime4KCPP
- **原始 README:** 見 `readme.md`
- **C++ API 文檔:** 見 `include/AC/Core.hpp`
- **CMake 文檔:** 見根目錄 `CMakeLists.txt`

---

**✨ Anime4KCPP DLL/SO 編譯包準備完成！**

請參考 `NEEVIEW_PACKAGE.md` 執行詳細的集成步驟。
