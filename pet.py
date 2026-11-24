import sys
import os
import random
import json
import subprocess
from datetime import datetime, timedelta

# Qt 的插件路径是从 Qt5Core.dll 位置推出来的，路径里有中文会被转成问号，
# 平台插件就找不到、窗口起不来。非 ASCII 时把插件目录复制到公共目录再指过去。
import shutil
import PyQt5

_QT_KEYS = ("platforms/qwindows.dll", "imageformats/qgif.dll", "styles/qwindowsvistastyle.dll")


def _qt_plugins():
    root = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
    try:
        root.encode("ascii")
        return root
    except UnicodeEncodeError:
        pass
    base = os.environ.get("PUBLIC") or os.path.join(os.environ.get("SystemDrive", "C:"), "ProgramData")
    dst = os.path.join(base, "YueXinMiao", "qtplugins")
    if all(os.path.exists(os.path.join(dst, k)) for k in _QT_KEYS):
        return dst
    try:
        if os.path.isdir(dst):
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(root, dst)
        return dst
    except Exception:
        return root


qt_plugin_root = _qt_plugins()
os.environ["QT_PLUGIN_PATH"] = qt_plugin_root
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(qt_plugin_root, "platforms")

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QMenu, QAction, QInputDialog
from PyQt5.QtCore import Qt, QTimer, QPoint, QPropertyAnimation, QEasingCurve, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QMovie, QCursor, QFont, QColor, QPainter, QBrush, QPen, QPainterPath


def app_dir():
    """程序所在目录：源码运行时是脚本目录，打包后是 exe 目录"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def res_dir():
    """资源目录：打包后模型和 gif 在 _internal 下"""
    base = app_dir()
    if getattr(sys, "frozen", False):
        inner = os.path.join(base, "_internal")
        if os.path.exists(os.path.join(inner, "hand_landmarker.task")):
            return inner
    return base


def _classify_hand(lm):
    """21个关键点 -> 规则手势: fist/open/pinch/none"""
    import math
    def d(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)
    wrist = lm[0]
    if d(lm[4], lm[8]) < 0.05:
        return "pinch"
    tips, pips = [8, 12, 16, 20], [6, 10, 14, 18]
    extended = 0
    for t, p in zip(tips, pips):
        if d(lm[t], wrist) > d(lm[p], wrist) * 1.15:
            extended += 1
    if extended <= 1:
        return "fist"
    if extended >= 3:
        return "open"
    return "none"


def _is_v_sign(lm):
    """V字手（食指中指伸直，其他弯）= 切换模式信号"""
    import math
    def d(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)
    wrist = lm[0]
    # 食指(8)和中指(12)必须伸
    idx_ext = d(lm[8], wrist) > d(lm[6], wrist) * 1.15
    mid_ext = d(lm[12], wrist) > d(lm[10], wrist) * 1.15
    # 无名指(16)和小指(20)必须弯
    ring_bend = d(lm[16], wrist) < d(lm[14], wrist) * 1.15
    pinky_bend = d(lm[20], wrist) < d(lm[18], wrist) * 1.15
    return idx_ext and mid_ext and ring_bend and pinky_bend


def _build_vec(hands_list):
    """拼126维: 左手前63, 右手后63, 没手填0"""
    import numpy as np
    vec = np.zeros(126, dtype="float32")
    for hi, lm in enumerate(hands_list[:2]):
        if lm is None:
            continue
        off = hi * 63
        for i, p in enumerate(lm):
            vec[off + i*3 + 0] = p.x
            vec[off + i*3 + 1] = p.y
            vec[off + i*3 + 2] = p.z
    return vec.reshape(1, 126)


# A2 onnx 类别: 0=单手比心 1=双手比心 2=OK 3=倒赞
_ONNX_MAP = {0: "heart1", 1: "heart2", 2: "ok", 3: "thumbsdown"}


class GestureProc(object):
    """手势识别子进程：主程序只负责启动、读结果、发模式"""

    def __init__(self, mode="rule"):
        self.base_dir = app_dir()
        self.res_dir = res_dir()
        self.script = os.path.join(self.res_dir, "gesture_service.py")
        self.state_path = os.path.join(self.base_dir, "gesture_state.json")
        self.mode_path = self.state_path + ".mode"
        self.stop_flag = self.state_path + ".stop"
        self.mode = mode
        self.proc = None
        self.last_seq = 0
        self.restarts = 0

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def set_mode(self, mode):
        self.mode = mode
        try:
            with open(self.mode_path, "w", encoding="utf-8") as f:
                f.write(mode)
        except OSError:
            pass

    def clear_flag(self):
        try:
            if os.path.isfile(self.stop_flag):
                os.remove(self.stop_flag)
        except OSError:
            pass

    def raise_flag(self):
        try:
            with open(self.stop_flag, "w", encoding="utf-8") as f:
                f.write("1")
        except OSError:
            pass

    def start(self):
        for p in (self.state_path, self.state_path + ".tmp"):
            try:
                os.remove(p)
            except OSError:
                pass
        self.set_mode(self.mode)
        frozen = getattr(sys, "frozen", False)
        model = os.path.join(self.res_dir, "hand_landmarker.task")
        mlp = os.path.join(self.res_dir, "mlp_weights.npz")
        if frozen:
            # 打包成 exe 后 sys.executable 就是主程序本身，用 --service 自调用
            cmd = [sys.executable, "--service",
                   "--out", self.state_path, "--model", model,
                   "--mlp", mlp, "--mode", self.mode]
            cwd = self.base_dir
        else:
            exe = sys.executable
            if os.name == "nt" and exe.lower().endswith("python.exe"):
                alt = exe[:-10] + "pythonw.exe"
                if os.path.exists(alt):
                    exe = alt
            cmd = [exe, self.script, "--out", self.state_path, "--model", model,
                   "--mlp", mlp, "--mode", self.mode]
            cwd = self.base_dir
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        self.proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )

    def kill_hard(self):
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()
        self.proc = None

    def read(self):
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None


class GestureWorker(QThread):
    """后台摄像头手势识别线程"""
    gesture = pyqtSignal(str)          # 离散动作: fist/open/pinch(捏下瞬间)
    pinch_move = pyqtSignal(float, float)  # 捏合时持续跟随: 食尖归一化坐标(x,y)

    def __init__(self, model_path, mlp=None):
        super().__init__()
        self.model_path = model_path
        self.mlp = mlp
        self._running = False
        self.mode = "rule"  # "rule"=拳掌捏，"mlp"=比心/OK/倒赞

    def run(self):
        import cv2
        import numpy as np
        import mediapipe as mp
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode
        MImage = mp.Image
        MFormat = mp.ImageFormat

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            num_hands=2,
            running_mode=RunningMode.VIDEO,
        )
        landmarker = HandLandmarker.create_from_options(options)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self._running = True
        last, streak, cooldown, ts = "none", 0, 0, 0
        try:
            while self._running:
                try:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = MImage(image_format=MFormat.SRGB, data=rgb)
                    ts += 33
                    res = landmarker.detect_for_video(mp_img, ts)
                    hands = res.hand_landmarks if res.hand_landmarks else []
                    g = "none"
                    lm = None

                    if self.mode == "rule":
                        # ===== 规则模式：只认拳/掌/捏合，其他全忽略 =====
                        if len(hands) == 1:
                            lm = hands[0]
                            # V字手优先检测（切换信号），不跑规则
                            if _is_v_sign(lm):
                                g = "none"
                            else:
                                g = _classify_hand(lm)
                        # 双手在规则模式下忽略
                    else:
                        # ===== MLP模式：只认4个训练手势，其他全忽略 =====
                        if len(hands) >= 2:
                            g = "heart2"  # 双手 = 双手比心
                            lm = hands[0]
                        elif len(hands) == 1:
                            lm = hands[0]
                            # V字手优先检测（切换信号），不跑MLP
                            if _is_v_sign(lm):
                                g = "none"
                            elif self.mlp is not None:
                                x = _build_vec([lm, None])
                                W1,b1,W2,b2,W3,b3 = self.mlp
                                h = np.maximum(0, x @ W1.T + b1)
                                h = np.maximum(0, h @ W2.T + b2)
                                logits = h @ W3.T + b3
                                e = np.exp(logits - logits.max())
                                probs = e / e.sum()
                                cid = int(probs.argmax())
                                conf = float(probs.max())
                                if conf >= 0.8:
                                    g = _ONNX_MAP.get(cid, "none")

                    if g == "pinch" and lm is not None:
                        self.pinch_move.emit(lm[8].x, lm[8].y)

                    if g == last:
                        streak += 1
                    else:
                        last, streak = g, 0
                    if streak == 6 and g != "none" and g != "pinch" and cooldown <= 0:
                        self.gesture.emit(g)
                        cooldown = 90
                    if cooldown > 0:
                        cooldown -= 1
                except Exception as frame_err:
                    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pet_debug.log"), "a", encoding="utf-8") as lf:
                        lf.write(f"\n[frame err] {frame_err}\n")
        finally:
            cap.release()
            landmarker.close()


class SpeechBubble(QWidget):
    """消息气泡（独立顶层窗口，避免被父窗口裁剪）"""
    def __init__(self, parent=None):
        super().__init__(None)
        self.pet_window = parent
        self.setWindowFlags(
            Qt.Tool |
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WA_TranslucentBackground)

        # 内部文字标签
        self.label = QLabel(self)
        self.label.setStyleSheet("""
            QLabel {
                background-color: transparent;
                border: none;
                padding: 0px;
                color: #000000;
                font-size: 18px;
                font-weight: bold;
                font-family: "Microsoft YaHei", "SimHei", sans-serif;
            }
        """)
        self.label.setWordWrap(True)
        self.label.setMaximumWidth(280)

        self.hide()

    def paintEvent(self, event):
        """手动绘制白色圆角背景 + 粉色边框"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(1, 1, -1, -1)
        radius = 15

        path = QPainterPath()
        path.addRoundedRect(float(rect.x()), float(rect.y()), float(rect.width()), float(rect.height()), radius, radius)

        # 白色填充背景
        painter.fillPath(path, QBrush(QColor(255, 255, 255)))

        # 粉色边框
        pen = QPen(QColor(255, 182, 193))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawPath(path)

        painter.end()

    def update_position(self):
        """根据宠物窗口位置更新气泡位置，显示在宠物上方居中"""
        if self.pet_window is None:
            return
        pet_geo = self.pet_window.geometry()
        # 先根据文字调整内部标签大小
        self.label.adjustSize()
        # 气泡窗口大小 = 标签大小 + 留白边距
        margin_x = 24
        margin_y = 16
        self.resize(self.label.width() + margin_x * 2, self.label.height() + margin_y * 2)
        self.label.move(margin_x, margin_y)
        bubble_w = self.width()
        bubble_h = self.height()
        # 水平居中，垂直在宠物上方
        x = pet_geo.x() + (pet_geo.width() - bubble_w) // 2
        y = pet_geo.y() - bubble_h - 8
        self.move(x, y)

    def show_message(self, text, duration=2000):
        self.label.setText(text)
        self.update_position()
        self.show()
        self.raise_()
        if self._timer is not None:
            self._timer.stop()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)
        self._timer.start(duration)

    _timer = None


class DesktopPet(QWidget):
    def __init__(self):
        super().__init__()
        
        # 窗口设置：无边框、置顶、透明背景、工具窗口（不显示在任务栏）
        self.setWindowFlags(
            Qt.FramelessWindowHint | 
            Qt.WindowStaysOnTopHint | 
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # 启用拖放接收
        self.setAcceptDrops(True)
        
        # 配置文件路径
        self.config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pet_config.json")
        
        # 窗口尺寸
        self.window_width = 200
        self.window_height = 220
        
        # 初始化位置（优先从配置文件读取，否则屏幕右下角）
        screen = QApplication.primaryScreen().geometry()
        self.config = self.load_config()
        saved_pos = self.config.get('x'), self.config.get('y')
        if saved_pos[0] is not None and saved_pos[1] is not None:
            x, y = saved_pos
        else:
            x = screen.width() - self.window_width - 50
            y = screen.height() - self.window_height - 100
        self.setGeometry(x, y, self.window_width, self.window_height)
        
        # 布局
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)
        
        # GIF 动画标签
        self.gif_label = QLabel(self)
        self.gif_label.setAlignment(Qt.AlignCenter)
        self.gif_label.setStyleSheet("background: transparent;")
        
        # 资源路径兼容开发模式和 PyInstaller 打包模式
        def resource_path(relative_path):
            if getattr(sys, 'frozen', False):
                base_path = os.path.dirname(sys.executable)
                candidates = [base_path, os.path.join(base_path, "_internal")]
                for candidate in candidates:
                    full = os.path.join(candidate, relative_path)
                    if os.path.exists(full):
                        return full
                return os.path.join(base_path, relative_path)
            return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)
        
        # 加载默认 GIF
        self.default_gif = resource_path("Screenshot_2026_0610_231825.gif")
        self.movie = QMovie(self.default_gif)
        self.movie.setScaledSize(QSize(180, 180))
        self.gif_label.setMovie(self.movie)
        self.movie.start()
        
        layout.addWidget(self.gif_label)
        
        # 消息气泡（独立顶层窗口）
        self.bubble = SpeechBubble(self)

        # CV 手势控制后台线程（右键菜单开关）
        self.gesture_worker = None
        self.gesture_proc = None
        self._cv_mode = False
        self._mlp_mode = False  # False=规则模式(拳掌捏)，True=MLP模式(比心/OK/倒赞)
        # 菜单项提前建好，保证不打开菜单也能读写（否则 toggle_gesture 会抛 AttributeError 直接崩）
        self.action_cv = QAction("开启 CV 手势控制 🖐", self)
        self.action_cv.triggered.connect(self.toggle_gesture)
        self.action_cv_mode = QAction("CV模式：规则（拳掌捏）", self)
        self.action_cv_mode.triggered.connect(self.toggle_cv_mode)
        self.gesture_timer = QTimer(self)
        self.gesture_timer.setInterval(30)
        self.gesture_timer.timeout.connect(self.on_gesture_poll)

        # 交互GIF映射
        self.gif_map = {
            "angry": resource_path("gif/mmexport1781187121041.gif"),
            "comfort": resource_path("gif/mmexport1781187182407.gif"),
            "enjoy": resource_path("gif/mmexport1781187232159.gif"),
            "cry": resource_path("gif/mmexport1781187246966.gif"),
            "play": resource_path("gif/mmexport1781187265905.gif"),
            "float": resource_path("gif/mmexport1781187281168.gif"),
        }
        
        # 调试日志：记录资源路径信息
        try:
            with open(resource_path("pet_debug.log"), "w", encoding="utf-8") as f:
                f.write(f"frozen={getattr(sys, 'frozen', False)}\n")
                f.write(f"executable={sys.executable}\n")
                f.write(f"default_gif={self.default_gif}\n")
                f.write(f"default_gif_exists={os.path.exists(self.default_gif)}\n")
                for k, v in self.gif_map.items():
                    f.write(f"{k}={v} exists={os.path.exists(v)}\n")
        except Exception:
            pass
        
        # GIF恢复定时器
        self.restore_timer = QTimer(self)
        self.restore_timer.setSingleShot(True)
        self.restore_timer.timeout.connect(self.restore_default_gif)
        
        # 拖拽相关
        self.dragging = False
        self.drag_position = QPoint()
        self.drag_start_time = None
        self.drag_positions = []  # 记录拖拽位置历史 [(x, y, timestamp), ...]
        self.is_throwing = False
        self.throw_timer = QTimer(self)
        self.throw_timer.timeout.connect(self.do_throw_physics)
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.bounce_count = 0

        # 剪贴板感知
        self.clipboard = QApplication.clipboard()
        self.clipboard.dataChanged.connect(self.on_clipboard_changed)
        self.last_clipboard_text = ""
        self.clipboard_messages = [
            "你复制了什么呀？给我看看！",
            "又在抄什么呢~",
            "这个看起来很重要哦！",
            "剪贴板里有秘密~"
        ]

        # 待办提醒
        self.reminders = []

        # 宠物状态
        self.hunger = 80
        self.mood = 80
        self.energy = 80
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(30000)
        self._low_stat_warned = {"hunger": False, "mood": False, "energy": False}

        # 自动散步
        self.walk_timer = QTimer(self)
        self.walk_timer.timeout.connect(self.auto_walk)
        self.walk_timer.start(random.randint(20000, 40000))
        self.walk_animation = None

        # 状态
        self.messages = [
            "呜呜...不要欺负我...",
            "抱抱我好不好？",
            "我会一直陪着你的！",
            "点击我，我会开心的~",
            "今天也要加油哦！",
            "我可爱吗？",
            "摸摸头~",
            "好想和你一起玩！",
            "不要走太远哦...",
            "你是我最好的朋友！",
            "你在忙什么呀？",
            "我饿了，给我吃点好吃的吧！",
            "好无聊呀，陪陪我嘛~",
            "你最近是不是很累？要注意休息哦",
            "我在这里等你好久了呢",
            "要不要跟我讲个秘密？",
            "嘿嘿，抓到你了！",
            "我就静静地看着你工作~",
            "你笑起来的样子一定很好看！",
            "不管发生什么，我都在你身边",
            "能遇见你真是太好了",
            "我要黏着你，不许跑！",
            "你今天真好看，比昨天还好看",
            "工作累了就看看我，心情会变好的",
            "我会变魔术哦，你看——我的心变大了！",
            "你知道吗？你是我最喜欢的人",
            "啦啦啦~我是快乐的小猫咪",
            "要是有小鱼干就好了...",
            "你在想什么呢？是在想我吗？",
            "给我挠挠下巴好不好嘛~",
            "我数到三，你要对我笑哦！一、二、三！",
            "你是我心里最特别的人，没有之一"
        ]
        
        self.click_count = 0
        
        # 好感度系统（0-100）
        self.affection = 50
        self.messages_low = [
            "你是谁呀...我有点怕生...",
            "不太熟的话，我还是保持距离吧...",
            "你要是能多陪陪我，也许我会开心一点",
            "我是不是做错什么了...",
            "你能多来看看我吗？",
        ]
        self.messages_mid = [
            "呜呜...不要欺负我...",
            "抱抱我好不好？",
            "我会一直陪着你的！",
            "点击我，我会开心的~",
            "今天也要加油哦！",
            "我可爱吗？",
            "摸摸头~",
            "好想和你一起玩！",
            "不要走太远哦...",
            "你是我最好的朋友！",
            "你在忙什么呀？",
            "我饿了，给我吃点好吃的吧！",
            "好无聊呀，陪陪我嘛~",
            "你最近是不是很累？要注意休息哦",
            "我在这里等你好久了呢",
            "要不要跟我讲个秘密？",
            "嘿嘿，抓到你了！",
            "我就静静地看着你工作~",
            "你笑起来的样子一定很好看！",
            "不管发生什么，我都在你身边",
            "能遇见你真是太好了",
            "我要黏着你，不许跑！",
            "你今天真好看，比昨天还好看",
            "工作累了就看看我，心情会变好的",
            "我会变魔术哦，你看——我的心变大了！",
            "你知道吗？你是我最喜欢的人",
            "啦啦啦~我是快乐的小猫咪",
            "要是有小鱼干就好了...",
            "你在想什么呢？是在想我吗？",
            "给我挠挠下巴好不好嘛~",
            "我数到三，你要对我笑哦！一、二、三！",
            "你是我心里最特别的人，没有之一"
        ]
        self.messages_high = [
            "我最喜欢你啦！",
            "嘿嘿，只要你陪着我，我就超开心！",
            "你已经离不开我了吧？",
            "我们要一直一直在一起哦！",
            "你对我最好了，我也最喜欢你！",
            "抱抱！充电完成！",
            "你就是我的全世界~",
            "不管去哪里都要带上我哦！",
            "我会一直守护你的！",
            "有你在，我每天都是晴天！",
        ]
        
        # 时间感知话术
        self.time_greetings = {
            "morning": ["早上好呀！今天也要元气满满！", "早起的猫咪有鱼吃~", "早安！新的一天开始啦！"],
            "noon": ["中午好！要不要休息一下？", "午饭时间到啦，你吃了吗？", "太阳好大呀，好困..."],
            "evening": ["晚上好！今天过得怎么样？", "夕阳西下，我想陪着你~", "天黑啦，要早点休息哦"],
            "night": ["夜深了...还不睡吗？", "晚安~我在梦里等你", "熬夜对身体不好哦..."],
        }
        
        # 空闲动画定时器（随机眨眼/晃动效果）
        self.idle_timer = QTimer(self)
        self.idle_timer.timeout.connect(self.idle_animation)
        self.idle_timer.start(5000)  # 每5秒检查一次
        
        # 自动说话定时器
        self.talk_timer = QTimer(self)
        self.talk_timer.timeout.connect(self.random_talk)
        self.talk_timer.start(15000)  # 每15秒随机说句话
        
        # 初始化动画
        self.scale_animation = None
        
        # 启动时根据时间说问候语
        QTimer.singleShot(500, self.greet_by_time)
    
    def switch_gif(self, gif_key, message, duration=5000):
        """切换到交互GIF，duration毫秒后恢复默认GIF（含3秒停留）"""
        path = self.gif_map.get(gif_key)
        if not path or not os.path.exists(path):
            return
        
        # 停止恢复定时器，防止重叠
        self.restore_timer.stop()
        
        # 切换GIF
        self.movie.stop()
        self.movie = QMovie(path)
        self.movie.setScaledSize(QSize(180, 180))
        self.gif_label.setMovie(self.movie)
        self.movie.start()
        
        # 显示气泡
        if message:
            self.bubble.show_message(message, duration)
        
        # 定时恢复默认GIF
        self.restore_timer.start(duration)
    
    def restore_default_gif(self):
        """恢复默认GIF"""
        self.movie.stop()
        self.movie = QMovie(self.default_gif)
        self.movie.setScaledSize(QSize(180, 180))
        self.gif_label.setMovie(self.movie)
        self.movie.start()
        
    def idle_animation(self):
        """空闲时的微小动画"""
        # 简单的缩放呼吸效果
        if not self.scale_animation:
            self.scale_animation = QPropertyAnimation(self.gif_label, b"minimumSize")
            self.scale_animation.setDuration(800)
            self.scale_animation.setEasingCurve(QEasingCurve.InOutQuad)
        
        current_size = self.gif_label.size()
        target_size = QSize(175, 175) if current_size.width() > 175 else QSize(180, 180)
        
        self.scale_animation.setStartValue(current_size)
        self.scale_animation.setEndValue(target_size)
        self.scale_animation.start()
        
    def get_message_by_affection(self):
        """根据好感度返回对应话术"""
        if self.affection >= 70:
            return random.choice(self.messages_high)
        elif self.affection >= 30:
            return random.choice(self.messages_mid)
        else:
            return random.choice(self.messages_low)
    
    def get_current_time_period(self):
        """获取当前时段：morning/noon/evening/night"""
        hour = datetime.now().hour
        if 5 <= hour < 11:
            return "morning"
        elif 11 <= hour < 17:
            return "noon"
        elif 17 <= hour < 22:
            return "evening"
        else:
            return "night"
    
    def greet_by_time(self):
        """启动时根据时间问候"""
        period = self.get_current_time_period()
        # 深夜自动切换睡觉GIF
        if period == "night":
            self.switch_gif("float", random.choice(self.time_greetings[period]), 5000)
        else:
            self.bubble.show_message(random.choice(self.time_greetings[period]), 3000)
    
    def random_talk(self):
        """随机说话（结合时间和好感度）"""
        if random.random() < 0.3:  # 30% 概率说话
            # 20%概率说时间相关的话，80%概率说好感度相关的话
            if random.random() < 0.2:
                period = self.get_current_time_period()
                msg = random.choice(self.time_greetings[period])
            else:
                msg = self.get_message_by_affection()
            self.bubble.show_message(msg, 3000)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.is_throwing:
                event.ignore()
                return
            self.dragging = True
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            self.drag_start_time = datetime.now()
            self.drag_positions = []
            self.click_count += 1
            
            # 点击增加好感度
            self.affection = min(100, self.affection + 2)
            # 点击增加心情
            self.mood = min(100, self.mood + 3)
            
            # 点击反馈动画
            self.bubble.show_message(self.get_message_by_affection(), 2500)
            
            # 简单的点击缩放反馈
            self.movie.setScaledSize(QSize(170, 170))
            QTimer.singleShot(150, lambda: self.movie.setScaledSize(QSize(180, 180)))
            
            # 每日签到检查
            self.check_daily_signin()
            
            event.accept()
        elif event.button() == Qt.RightButton:
            self.show_context_menu(event.globalPos())
    
    def mouseMoveEvent(self, event):
        if self.dragging and event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_position)
            now = datetime.now()
            self.drag_positions.append((event.globalPos().x(), event.globalPos().y(), now))
            if len(self.drag_positions) > 5:
                self.drag_positions.pop(0)
            event.accept()

    def moveEvent(self, event):
        """宠物移动时同步更新气泡位置 + 屏幕边缘碰撞检测"""
        super().moveEvent(event)
        if hasattr(self, 'bubble') and self.bubble.isVisible():
            self.bubble.update_position()
        
        # 边缘碰撞检测（投掷模式下由投掷物理处理）
        if self.is_throwing:
            return
        
        screen = QApplication.primaryScreen().geometry()
        margin = 5
        x, y = self.x(), self.y()
        bounced = False
        
        if x <= margin:
            x = margin + 10
            bounced = True
        elif x + self.window_width >= screen.width() - margin:
            x = screen.width() - self.window_width - margin - 10
            bounced = True
        
        if y <= margin:
            y = margin + 10
            bounced = True
        elif y + self.window_height >= screen.height() - margin:
            y = screen.height() - self.window_height - margin - 10
            bounced = True
        
        if bounced:
            self.move(x, y)
            self.switch_gif("angry", "哎呀！撞疼我了！😣", 3000)

    def mouseReleaseEvent(self, event):
        self.dragging = False
        if self.is_throwing:
            return
        if len(self.drag_positions) >= 2:
            last = self.drag_positions[-1]
            prev = self.drag_positions[-2]
            dx = last[0] - prev[0]
            dy = last[1] - prev[1]
            dt_ms = (last[2] - prev[2]).total_seconds() * 1000
            if dt_ms > 0:
                vel = ((dx / dt_ms * 50) ** 2 + (dy / dt_ms * 50) ** 2) ** 0.5
                if vel > 15:
                    self.start_throw(dx / dt_ms * 50, dy / dt_ms * 50)
        
    def mouseDoubleClickEvent(self, event):
        """双击互动"""
        if event.button() == Qt.LeftButton:
            self.bubble.show_message("双击好开心！✨", 2000)
            # 弹跳动画效果
            orig_y = self.y()
            
            anim_up = QPropertyAnimation(self, b"geometry")
            anim_up.setDuration(200)
            anim_up.setStartValue(self.geometry())
            anim_up.setEndValue(self.geometry().adjusted(0, -30, 0, -30))
            anim_up.setEasingCurve(QEasingCurve.OutQuad)
            
            anim_down = QPropertyAnimation(self, b"geometry")
            anim_down.setDuration(200)
            anim_down.setStartValue(self.geometry().adjusted(0, -30, 0, -30))
            anim_down.setEndValue(self.geometry())
            anim_down.setEasingCurve(QEasingCurve.InQuad)
            
            anim_up.finished.connect(anim_down.start)
            anim_up.start()

    def dragEnterEvent(self, event):
        """拖拽进入时，如果是文件则接受"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """放下文件时，删除文件并播放吃掉的动画"""
        urls = event.mimeData().urls()
        if not urls:
            return
        file_path = urls[0].toLocalFile()
        if not os.path.isfile(file_path):
            self.bubble.show_message("这个我吃不掉呀...", 2000)
            return
        try:
            os.remove(file_path)
            self.switch_gif("enjoy", "嘿嘿，文件被我吃掉啦！真香~", 7000)
        except Exception as e:
            self.bubble.show_message("哎呀，吃撑了没删掉...", 2000)
    
    def show_context_menu(self, pos):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: rgba(255, 255, 255, 240);
                border: 1px solid #FFB6C1;
                border-radius: 8px;
                padding: 5px;
            }
            QMenu::item {
                padding: 8px 25px;
                border-radius: 5px;
                color: #333;
                font-family: "Microsoft YaHei";
            }
            QMenu::item:selected {
                background-color: #FFB6C1;
                color: white;
            }
        """)
        
        action_feed = QAction("喂食 🍖", self)
        action_feed.triggered.connect(self.feed)
        menu.addAction(action_feed)
        
        action_pet = QAction("蛐蛐 🖐", self)
        action_pet.triggered.connect(self.pet)
        menu.addAction(action_pet)
        
        action_angry = QAction("惹我生气 😠", self)
        action_angry.triggered.connect(self.angry)
        menu.addAction(action_angry)
        
        action_cry = QAction("溜了 😢", self)
        action_cry.triggered.connect(self.cry)
        menu.addAction(action_cry)
        
        action_play = QAction("陪我玩 📱", self)
        action_play.triggered.connect(self.play_phone)
        menu.addAction(action_play)
        
        action_float = QAction("去睡觉 💤", self)
        action_float.triggered.connect(self.go_sleep)
        menu.addAction(action_float)

        self.action_cv = QAction("开启 CV 手势控制 🖐", self)
        self.action_cv.triggered.connect(self.toggle_gesture)
        menu.addAction(self.action_cv)

        self.action_cv_mode = QAction("CV模式：规则（拳掌捏）", self)
        self.action_cv_mode.triggered.connect(self.toggle_cv_mode)
        menu.addAction(self.action_cv_mode)

        menu.addSeparator()
        
        action_reminder = QAction("添加提醒 ⏰", self)
        action_reminder.triggered.connect(self.add_reminder)
        menu.addAction(action_reminder)
        
        action_show_reminders = QAction("我的提醒 📋", self)
        action_show_reminders.triggered.connect(self.show_reminders)
        menu.addAction(action_show_reminders)
        
        action_stats = QAction("查看状态 📊", self)
        action_stats.triggered.connect(self.show_stats)
        menu.addAction(action_stats)
        
        menu.addSeparator()
        
        action_quit = QAction("退出 😢", self)
        action_quit.triggered.connect(QApplication.quit)
        menu.addAction(action_quit)
        
        menu.exec_(pos)
    
    def feed(self):
        """喂食 - 享受动画，增加好感度"""
        self.affection = min(100, self.affection + 5)
        self.hunger = min(100, self.hunger + 30)
        self.energy = min(100, self.energy + 10)
        self.switch_gif("enjoy", "好吃好吃！谢谢你！🍖", 6000)
    
    def pet(self):
        """抚摸 - 安慰动画，增加好感度"""
        self.affection = min(100, self.affection + 5)
        self.mood = min(100, self.mood + 20)
        self.switch_gif("comfort", "嘿嘿~", 6000)
    
    def angry(self):
        """生气 - 发怒动画，减少好感度"""
        self.affection = max(0, self.affection - 10)
        self.switch_gif("angry", "哼！不理你了！😠", 5000)
    
    def cry(self):
        """被欺负 - 大哭动画，减少好感度"""
        self.affection = max(0, self.affection - 10)
        self.switch_gif("cry", "溜啦溜啦~", 7000)
    
    def play_phone(self):
        """玩手机 - 开心动画"""
        self.mood = min(100, self.mood + 15)
        self.energy = max(0, self.energy - 5)
        self.switch_gif("play", "一起来玩手机呀~ 📱", 6000)
    
    def go_sleep(self):
        """睡觉 - 飘走动画"""
        self.energy = min(100, self.energy + 40)
        self.switch_gif("float", "我去休息啦~ 💤", 7000)

    def toggle_gesture(self):
        """右键菜单：开启/关闭摄像头手势识别"""
        if self.gesture_proc is None:
            self.gesture_proc = GestureProc("mlp" if self._mlp_mode else "rule")
            self.gesture_proc.clear_flag()
            self.gesture_proc.start()
            self.gesture_timer.start()
            self._pinch_pos = None
            self._cv_mode = True
            self.walk_timer.stop()  # 开CV期间暂停自动散步，交给手控制
            self.action_cv.setText("关闭 CV 手势控制 🖐")
            self.bubble.show_message("手势识别开启：✊喂食 🖐溜了 👌蛐蛐 💛陪玩 💛💛睡觉 👎生气 🤏拖动", 4000)
        else:
            self.stop_gesture()
            self.walk_timer.start(15000)  # 15秒后恢复自动散步
            self.action_cv.setText("开启 CV 手势控制 🖐")
            self.bubble.show_message("手势识别已关闭", 2000)

    def stop_gesture(self):
        """关闭手势子进程"""
        self.gesture_timer.stop()
        if self.gesture_proc is not None:
            self.gesture_proc.raise_flag()
            self.gesture_proc.kill_hard()
            self.gesture_proc.clear_flag()
            self.gesture_proc = None
        self._cv_mode = False

    def on_gesture_poll(self):
        """轮询子进程写出的状态文件"""
        p = self.gesture_proc
        if p is None:
            return
        if not p.alive():
            if p.restarts < 3:
                p.restarts += 1
                p.clear_flag()
                p.start()
                return
            self.stop_gesture()
            self.action_cv.setText("开启 CV 手势控制 🖐")
            return
        data = p.read()
        if not data:
            return
        seq = data.get("seq", 0)
        if seq == p.last_seq:
            return
        p.last_seq = seq
        event = data.get("event")
        if event == "pinch":
            self.on_pinch_move(float(data.get("x", 0.5)), float(data.get("y", 0.5)))
            return
        if data.get("fire"):
            self.on_gesture(event)

    def toggle_cv_mode(self):
        """切换CV识别模式：规则(拳掌捏) <-> MLP(比心/OK/倒赞)"""
        self._mlp_mode = not self._mlp_mode
        if self._mlp_mode:
            self.action_cv_mode.setText("CV模式：MLP（比心/OK/倒赞）")
            self.bubble.show_message("切换到MLP模式：💛比心 👌OK 👎倒赞 💛💛双手", 3000)
            if self.gesture_proc:
                self.gesture_proc.set_mode("mlp")
        else:
            self.action_cv_mode.setText("CV模式：规则（拳掌捏）")
            self.bubble.show_message("切换到规则模式：✊拳 🖐掌 🤏捏", 3000)
            if self.gesture_proc:
                self.gesture_proc.set_mode("rule")

    def on_gesture(self, name):
        """手势线程回调：映射到6个菜单动作"""
        mapping = {
            "fist":       self.feed,        # 握拳 -> 喂食
            "open":       self.cry,         # 张开掌 -> 欺负我
            "ok":         self.pet,          # OK -> 抚摸
            "heart1":     self.play_phone,   # 单手比心 -> 陪我玩
            "heart2":     self.go_sleep,     # 双手比心 -> 去睡觉
            "thumbsdown": self.angry,        # 倒赞 -> 惹我生气
        }
        if name in mapping:
            mapping[name]()

    def on_pinch_move(self, nx, ny):
        """捏合跟随：猫中心移到指尖对应屏幕位置（EMA 平滑，跟手不抖）"""
        screen = QApplication.primaryScreen().geometry()
        tx, ty = nx * screen.width(), ny * screen.height()
        if self._pinch_pos is None:
            self._pinch_pos = (tx, ty)
        else:
            sx, sy = self._pinch_pos
            self._pinch_pos = (sx * 0.15 + tx * 0.85, sy * 0.15 + ty * 0.85)
        px, py = self._pinch_pos
        g = self.geometry()
        nx, ny = int(px - g.width() / 2), int(py - g.height() / 2)
        if abs(nx - g.x()) < 2 and abs(ny - g.y()) < 2:
            return
        self.move(nx, ny)

    def load_config(self):
        """从配置文件读取配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}
    
    def save_config(self):
        """保存当前配置到文件"""
        try:
            data = self.config.copy()
            data['x'] = self.x()
            data['y'] = self.y()
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def start_throw(self, vel_x, vel_y):
        """开始投掷物理模拟"""
        self.is_throwing = True
        self.pos_x = float(self.x())
        self.pos_y = float(self.y())
        self.vel_x = vel_x
        self.vel_y = vel_y
        self.bounce_count = 0
        self.throw_timer.start(16)

    def do_throw_physics(self):
        """投掷物理每帧更新"""
        if not self.is_throwing:
            return
        self.pos_x += self.vel_x
        self.pos_y += self.vel_y
        self.vel_y += 0.5

        screen = QApplication.primaryScreen().geometry()
        margin = 5
        bounced = False

        if self.pos_x <= margin:
            self.pos_x = margin
            self.vel_x = -self.vel_x * 0.7
            bounced = True
        elif self.pos_x + self.window_width >= screen.width() - margin:
            self.pos_x = screen.width() - self.window_width - margin
            self.vel_x = -self.vel_x * 0.7
            bounced = True

        if self.pos_y <= margin:
            self.pos_y = margin
            self.vel_y = -self.vel_y * 0.7
            bounced = True
        elif self.pos_y + self.window_height >= screen.height() - margin:
            self.pos_y = screen.height() - self.window_height - margin
            self.vel_y = -self.vel_y * 0.7
            bounced = True

        self.move(int(self.pos_x), int(self.pos_y))

        if bounced:
            self.bounce_count += 1
            self.switch_gif("angry", "哎呀！撞疼我了！😵", 3000)
            if self.bounce_count >= 3:
                self.is_throwing = False
                self.throw_timer.stop()
                return

        speed = (self.vel_x ** 2 + self.vel_y ** 2) ** 0.5
        if speed < 1:
            self.is_throwing = False
            self.throw_timer.stop()

    def on_clipboard_changed(self):
        """剪贴板内容变化回调"""
        text = self.clipboard.text()
        if text and text != self.last_clipboard_text:
            self.last_clipboard_text = text
            self.bubble.show_message(random.choice(self.clipboard_messages), 2500)

    def add_reminder(self):
        """添加提醒"""
        content, ok = QInputDialog.getText(self, "添加提醒", "提醒内容:")
        if not ok or not content.strip():
            return
        minutes, ok = QInputDialog.getInt(self, "添加提醒", "几分钟后提醒:", 5, 1, 120, 1)
        if not ok:
            return
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: self.trigger_reminder(timer, content.strip()))
        timer.start(minutes * 60 * 1000)
        self.reminders.append(timer)
        self.bubble.show_message(f"好的，{minutes}分钟后提醒你：{content.strip()}", 2000)

    def trigger_reminder(self, timer, content):
        """提醒触发"""
        if timer in self.reminders:
            self.reminders.remove(timer)
        self.bubble.show_message(f"你让我提醒你：{content}！", 5000)

    def show_reminders(self):
        """显示当前提醒数量"""
        count = len(self.reminders)
        if count == 0:
            self.bubble.show_message("当前没有提醒哦~", 2000)
        else:
            self.bubble.show_message(f"当前还有 {count} 个提醒待完成~", 2000)

    def check_daily_signin(self):
        """每日签到检查"""
        today = datetime.now().strftime("%Y-%m-%d")
        last_signin = self.config.get("last_signin_date", "")
        consecutive = self.config.get("consecutive_days", 0)

        if today == last_signin:
            return

        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        if last_signin == yesterday:
            consecutive += 1
        else:
            consecutive = 1

        self.config["last_signin_date"] = today
        self.config["consecutive_days"] = consecutive

        if consecutive == 1:
            msg = "今天也来找我啦！开心！"
        elif consecutive == 7:
            msg = "你连续7天来看我了！好开心！🎉"
        elif consecutive == 30:
            msg = "连续30天！我们是最好的朋友了！🏆"
        else:
            msg = f"连续{consecutive}天签到！加油！"
        self.bubble.show_message(msg, 3000)

        if consecutive >= 7:
            bonus = "你是我见过最坚持的人！"
            if bonus not in self.messages_high:
                self.messages_high.append(bonus)

    def update_stats(self):
        """定时衰减宠物状态"""
        self.hunger = max(0, self.hunger - 2)
        self.mood = max(0, self.mood - 1)
        self.energy = max(0, self.energy - 1)

        if self.hunger < 20 and not self._low_stat_warned["hunger"]:
            self.bubble.show_message("我好饿呀...给我吃点东西吧", 3000)
            self._low_stat_warned["hunger"] = True
        elif self.hunger >= 30:
            self._low_stat_warned["hunger"] = False

        if self.mood < 20 and not self._low_stat_warned["mood"]:
            self.bubble.show_message("我有点不开心...陪陪我好不好", 3000)
            self._low_stat_warned["mood"] = True
        elif self.mood >= 30:
            self._low_stat_warned["mood"] = False

        if self.energy < 20 and not self._low_stat_warned["energy"]:
            self.bubble.show_message("我好累呀...让我睡一会儿吧", 3000)
            self._low_stat_warned["energy"] = True
        elif self.energy >= 30:
            self._low_stat_warned["energy"] = False

    def show_stats(self):
        """显示宠物状态"""
        msg = f"饥饿:{self.hunger} 心情:{self.mood} 精力:{self.energy}\n好感度:{self.affection}"
        self.bubble.show_message(msg, 4000)

    def auto_walk(self):
        """自动散步"""
        if self._cv_mode:
            # CV 模式下由手势控制位置，暂停自动散步
            self.walk_timer.start(random.randint(20000, 40000))
            return
        if self.dragging or self.is_throwing:
            self.walk_timer.start(random.randint(20000, 40000))
            return
        screen = QApplication.primaryScreen().geometry()
        margin = 50
        target_x = random.randint(margin, screen.width() - self.window_width - margin)
        target_y = random.randint(margin, screen.height() - self.window_height - margin)
        duration = random.randint(6000, 10000)
        self.walk_animation = QPropertyAnimation(self, b"pos")
        self.walk_animation.setDuration(duration)
        self.walk_animation.setStartValue(self.pos())
        self.walk_animation.setEndValue(QPoint(target_x, target_y))
        self.walk_animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.walk_animation.finished.connect(lambda: self.walk_timer.start(random.randint(20000, 40000)))
        self.walk_animation.start()

    def enterEvent(self, event):
        """鼠标悬停"""
        self.setCursor(QCursor(Qt.OpenHandCursor))
        
    def leaveEvent(self, event):
        """鼠标离开"""
        self.setCursor(QCursor(Qt.ArrowCursor))


def main():
    # 打包成 exe 后，识别子进程用同一个 exe 加 --service 自调用
    if "--service" in sys.argv:
        argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a != "--service"]
        sys.argv = argv
        import gesture_service
        gesture_service.main()
        return

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    
    pet = DesktopPet()
    pet.show()
    
    # 应用退出时保存配置并结束手势子进程
    app.aboutToQuit.connect(pet.save_config)
    app.aboutToQuit.connect(pet.stop_gesture)
    
    if "--gesture" in sys.argv:
        QTimer.singleShot(1500, pet.toggle_gesture)
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
