# 月薪喵桌宠（手势版）

一只会吃饭、会生气、会睡觉的桌面猫，用摄像头识别手势控制它。Windows 10/11，x64。

## 直接运行（推荐，不需要装 Python）

从 Release 下载 `月薪喵桌宠-手势完整版.zip`，解压到任意目录，双击 `月薪喵.exe`。

只想要手势，双击 `start_gesture.bat`，或者右键宠物选「开启 CV 手势控制」。

镜头权限在第一次运行时会由 Windows 弹窗询问，允许即可。

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

```
pip install pyinstaller
python build.py
```

产物在 `dist/月薪喵/`，把整个目录打 zip 分发。

## 许可

MIT
