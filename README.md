# PicAdj

Windows 桌面小工具：从手机拍摄的文档照片中自动检测四角，透视矫正为正视图（**CPU**，可选 **ONNX Runtime** 分割模型兜底）。

## 功能概要

### 列表与图片

- **导入图片…**：可多选 JPG / PNG / BMP / TIFF / WebP 等；左侧列表显示文件名。
- **列表选择**：**单击**一项切换主预览；支持 **Ctrl** 点选多张、**Shift** 扩展范围（与资源管理器类似）。多选时，若**当前主预览**仍在选中范围内，主图**不跳转**；否则主预览切到选中项中**序号最小**的那一张。
- **每张图的四个角点单独记忆**：在左侧原图上拖动黄点微调；切换到其他图再切回，角点不丢失。
- **移除所选**：从列表去掉**当前列表选中项**（可多张）；若列表里没有任何选中行，则移除**当前主预览**对应的那一张。**仅从内存列表删除，不删磁盘文件**。列表有焦点时可按 **Delete**，与按钮行为一致（会弹出确认框，并列出待移除文件名）。
- **清空列表**：清空全部（需确认）。
- **自动检测**：导入后按**队列逐张**检测。若在轮到某张之前已拖动过角点，会**跳过**该张自动检测，以免覆盖手动结果。
- **重新检测**：只对**当前主预览**那张重新跑检测（会取消「已手动调整」标记并按新角点更新预览）。

### 矫正预览（右侧面板）

- **选中列表中的某张**后，会按**当前四角**自动做透视矫正并显示在右侧（无需先点「应用矫正」）。
- **后台检测完成**时，若正好在看那张图，预览会随新角点刷新。
- **拖动角点**：拖动过程中只更新左侧框线（减少闪烁），**不会在拖动中重复做透视**；**松开鼠标**后再计算矫正并刷新右侧。
- **应用矫正（当前）**：与自动预览效果相同，仍保留按钮以便显式刷新或在习惯上点一次确认。

### 导出

- **全部应用并保存…**：对列表中**每一张**用各自四角矫正，写入所选文件夹。若仍有图未完成自动检测，会先列出文件名并确认是否仍按当前角点导出。文件名默认为 **`IMG_YYYYMMDD_三位编号`**（如 `IMG_20260406_001.jpg`）；工具栏可选批量格式 **JPEG** 或 **PNG**。
- **保存当前为…**：单张另存为；默认文件名规则与批量一致。

### 工具栏选项

- **CLAHE 增强**：只作用于**矫正之后**的结果（右侧预览、保存、批量导出），不参与四角检测。适合阴影、半边偏暗等场景。
- **低置信度时尝试 ONNX**：若存在 `models\doc_segment.onnx`（或环境变量 `PICADJ_ONNX_MODEL`），可在 OpenCV 结果不稳时融合分割模型。

## CLAHE 增强（技术说明）

作用于透视**完成之后**的 BGR 图像，**不参与**自动找角。

- **实现**：BGR → LAB，仅对 **L** 通道做 **CLAHE**（`clipLimit=2.0`，`8×8` tile），再还原 BGR。代码见 `src/rectify.py` 中 `optional_clahe_bgr`。
- **说明**：检测管线里的 `adaptive_clahe_canny` 仅用于**内部边缘**，与界面「CLAHE 增强」无关。

## 目录结构

| 路径 | 说明 |
|------|------|
| `run_picadj.py` | 入口（开发与 PyInstaller 共用） |
| `src/app_main.py` | Tkinter UI、列表、预览与导出 |
| `src/detect_document.py` | OpenCV 文档四边形检测 |
| `src/rectify.py` | 透视矫正、CLAHE、落盘 |
| `src/onnx_backend.py` | 可选 ONNX 与 OpenCV 融合 |
| `picadj.spec` | PyInstaller 配置 |
| `scripts/build_exe.bat` | 安装依赖并生成 exe |

## 环境要求

- **系统**：Windows 10 / 11（x64）
- **Python**：3.10+（仅开发与打包时需要）

## 运行（开发）

在**项目根目录** `PicAdj` 下：

```powershell
py -3 -m pip install -r requirements.txt
py -3 run_picadj.py
```

也可：`py -3 -m src`

## 打包为 exe

在**项目根目录**执行批处理（会调用 `pip` 与 PyInstaller）：

```bat
scripts\build_exe.bat
```

若在 **PowerShell** 中调用，可使用：

```powershell
cmd /c "scripts\build_exe.bat"
```

生成 **`dist\PicAdj.exe`**（单文件、无控制台窗口）。

- 首次打包需联网拉依赖。
- 部分环境可能对 PyInstaller 产物误报，可按需加信任。
- **ONNX 模型不会被打进 exe**：请将模型放到 **`PicAdj.exe` 同目录下的** `models\doc_segment.onnx`，或通过 **`PICADJ_ONNX_MODEL`** 指向文件。

## 可选 ONNX 模型

将符合 `src/onnx_backend.py` 约定的分割 ONNX 放到 `models\doc_segment.onnx`（与 exe 同级的 `models\` 文件夹），或设置环境变量 **`PICADJ_ONNX_MODEL`**。未放置时仅使用 OpenCV 检测。

## 依赖

见 **`requirements.txt`**（`opencv-python`、`numpy`、`Pillow`、`onnxruntime`、`pyinstaller` 等）。
