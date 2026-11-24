# 月薪喵桌宠（手势版）

一只会吃饭、会生气、会睡觉的桌面猫，用摄像头识别手势控制它。Windows 10/11，x64。

## 直接运行（推荐，不需要装 Python）

从 Release 下载 `月薪喵桌宠-便携版.zip`，约 208 MB，解压到任意目录，中文路径也可以。双击 `启动.bat`。

要手势，双击 `启动-手势.bat`。不确定环境能不能跑，双击 `环境自检.bat`，它会打印实际使用的 Python 路径和依赖版本。

包里带完整解释器和依赖，`runtime\` 里有 Python 3.11 加 mediapipe、OpenCV、PyQt5，和系统上装没装 Python 无关，删掉 `runtime\` 就跑不起来。

镜头权限在第一次运行时会由 Windows 弹窗询问，允许即可。摄像头同一时间只能给一个程序用。

### 这个包处理掉的两个坑

mediapipe 的 C++ 层打不开含中文的路径，报 `Unable to open file at ...` 并且路径变成乱码。程序启动时检查模型路径，不是纯 ASCII 就复制一份到 `C:\Users\Public\YueXinMiao\` 再读原文。

mediapipe 和 Qt 在同一个进程里初始化，顺序不对就会 DLL 初始化失败，PyQt5 直接 abort，退出码 0xC0000409，没有任何 traceback。识别跑在独立子进程，主程序只读状态文件，两者不共享进程。

## 从源码运行

需要 Python 3.11，64 位。其他版本 mediapipe 0.10.21 未必有对应轮子。

```
pip install -r requirements.txt
python pet.py
python pet.py --gesture
```

`--gesture` 会在启动 1.5 秒后自动开识别。

## 手势

规则模式，默认。

| 手势 | 动作 |
| --- | --- |
| 握拳 | 喂食，好感度加 5 |
| 张开手掌 | 欺负我，好感度减 10 |
| 拇指食指捏合 | 拖动猫，捏着移动它跟着指尖走 |

MLP 模式，右键菜单「切换 CV 模式」后生效。

| 手势 | 动作 |
| --- | --- |
| 单手比心 | 陪我玩 |
| 双手比心 | 去睡觉 |
| OK | 抚摸，好感度加 5 |
| 倒赞 | 惹我生气，好感度减 10 |

V 字手在两种模式下都当切换信号，不触发动作。这是原设计，不是 bug。

## 判定参数

`pinch` 拇指食指距离小于 0.05 算捏合；指尖到手腕的距离超过指节到手腕的 1.15 倍算这根手指伸直；伸直手指数不超过 1 判握拳，不少于 3 判张开；同一个手势连续 6 帧才确认，触发后冷却 90 帧。摄像头 640×480、CAP_DSHOW、画面水平翻转。

识别跑在独立进程（`gesture_service.py`），主程序每 30 毫秒读一次结果文件。这样 mediapipe 和 Qt 不在同一个进程里，不会因为 DLL 初始化冲突把整个程序带崩。

## 常见问题

**开了手势几秒后程序消失，没有报错。** PyQt5 在槽函数里抛出未捕获异常会直接 abort，没有 traceback。项目里已经用 `app_dir()` 和提前创建 QAction 避开已知的两处，如果还能复现，先看 `pet_debug.log` 和 `gesture_state.json.debug.log` 的最后几行。

**手势没反应。** 先看 `gesture_state.json.debug.log` 有没有 `hands=1` 或 `hands=2`。全是 `hands=0` 说明摄像头没抓到人，检查镜头遮挡、距离、光线。有 `hands` 但 `g=none`，多半是被 V 字判定拦住了，把手型摆正一点。

**捏着拖动一顿一顿。** 摄像头同时只能给一个程序用，别让别的软件占着镜头。帧率低于 25fps 时把其他占 CPU 的程序关掉。

**换机器后摄像头打不开。** 检查设备管理器里摄像头状态，以及 Windows 设置里「相机」的全局开关。

## 目录

```
pet.py                 主程序，界面、动作、好感度、气泡
gesture_service.py     识别子进程，判定逻辑
hand_landmarker.task   MediaPipe 手部关键点模型
mlp_weights.npz        MLP 权重，比心/OK/倒赞
gif/                   各状态动画
Screenshot_...gif      默认待机动画
requirements.txt       锁定版本
```

## 打包

PyInstaller 这套依赖上用不了。`--collect-all mediapipe` 和精简参数都卡在 `Looking for dynamic libraries` 不动，进程占着内存但 CPU 停住，十分钟没有新日志。便携包是手工搭的：

1. 下载 `python-3.11.6-embed-amd64.zip`，解压到 `runtime\`
2. 改 `runtime\python311._pth`，四行：`python311.zip`、`.`、`Lib\site-packages`、`import site`
3. 从 `C:\Windows\System32` 复制 `msvcp140.dll`、`msvcp140_1.dll`、`msvcp140_2.dll`、`concrt140.dll`、`vcomp140.dll` 到 `runtime\`。embeddable 只带 vcruntime140 系列，缺这几个 PyQt5 和 mediapipe 起不来
4. 把 venv 的 `site-packages` 整体复制进 `runtime\Lib\site-packages`。matplotlib、PIL、fontTools、kiwisolver、cycler、pyparsing、dateutil、contourpy 都不能删，mediapipe 的 `python\solutions\drawing_utils.py` 会 import matplotlib
5. `app\` 放 `pet.py`、`gesture_service.py`、模型、动画；根目录放三个 bat，都 `chcp 65001`、CRLF，`cd` 到 `app` 再调 `..\runtime\pythonw.exe pet.py`

成品 505 MB，压完 208 MB。真机验证：换目录启动、全新解压启动都正常，32fps，子进程在跑。

## 许可

MIT
