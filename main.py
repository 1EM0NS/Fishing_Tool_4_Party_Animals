"""
fishing_tool.py

依赖:
    pip install opencv-python mss Pillow numpy pyautogui keyboard

功能:
 - 实时显示 region 区域 (progress area) 与 HSV 掩膜
 - 检测圆角三角指针 (使用颜色 + minEnclosingTriangle)
 - 显示 angle / speed，调节策略参数（速度阈值、短按/长按时长等）
 - 键盘:
     1: 启动/停止自动钓鱼循环 (Start/Stop automation)
     2: 切换“有鱼检测/手动标记有鱼”模式 (当按下2时，立刻认为有鱼并触发收杆)
     3: 立即停止自动化 (紧急停止)
 - UI 上也有按钮控制与参数输入

使用说明:
 - 修改 region 变量为你的进度条区域 (你已提供):
     region = {"top": 1010, "left": 1370, "width": 400, "height": 200}
 - 首次运行可用 "Set Bite Region" 手动框选 感叹号/头像区域（可选）。
 - 调整阈值后点击 "Start Detect" 查看效������
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import mss
import time
import threading
import pyautogui
import keyboard  # 新增库

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

# ---------- 初始参数 (你可以在 UI 中调整) ----------
# 指针检测���域
pointer_region = {"top": 1010, "left": 1370, "width": 400, "height": 200}
bite_region = {"top": 300, "left": 1400, "width": 300, "height": 300}  # 感叹号区域
# HSV 颜色阈值（根据你给的 HSV: H≈18 S≈214 V≈255）
LOWER_ORANGE = np.array([8, 150, 200])
UPPER_ORANGE = np.array([28, 255, 255])

# 自动钓鱼策略默认参数
params = {
    "cast_hold_time": 3.0,       # 抛竿按住时长 (秒)
    "post_cast_wait": 0.5,       # 抛竿后等待的微小间隔
    "bite_threshold": 0.78,      # 上钩模板匹配阈值
    "max_bite_wait": 30,         # 等待上钩的最大时间 (秒)
    "pointer_loss_time": 0.2,    # 未检测到指针的时间 (秒)
    "release_angle": 60,         # 角度小于多少就松开
    "release_speed": 10,         # 速度超过多少就松开
    "reel_end_wait": 1.8,        # 收杆循环结束等待时间 (秒)
    "short_press_time": 0.2,     # 收杆结束后的短按时间 (秒)
    "next_cast_sleep": 2.0       # 开始下一轮抛竿的睡眠时间 (秒)
}

# 其它
FRAME_INTERVAL_MS = 50  # UI 刷新间隔 (ms)

# ---------- 全局状态 ----------
running_detection = False      # 是否在检��（UI 刷新）
automation_running = False     # 自动钓鱼循环是否在运行
bite_mode_manual = False       # 手动触发“有鱼” (按2会切换 / 触发)
stop_requested = False         # 请求停止自动化线���
last_action_text = "Idle"

sct = mss.mss()

# ---------- 图像处理与���针检测 ----------
def rotate_image(img, angle):
    """旋转图像，不裁剪内容，返回旋转后的图像"""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(rot_mat[0, 0])
    sin = np.abs(rot_mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(img, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)

def detect_pointer_angle_and_annotate(bgr_img, ui_handle=None):
    img = np.ascontiguousarray(bgr_img, dtype=np.uint8)

    # 模板匹配模式（支持旋转+颜色mask）
    if ui_handle and ui_handle.pointer_template is not None:
        # 对进度条区域做HSV颜色mask
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_img = cv2.inRange(hsv_img, LOWER_ORANGE, UPPER_ORANGE)
        img_masked = cv2.bitwise_and(img, img, mask=mask_img)

        best_val = -1
        best_loc = None
        best_angle = 0
        best_tpl = None
        best_tpl_shape = None

        # 旋转模板并匹配（彩色+mask）
        for ang in range(-40, 41, 10):
            tpl_rot = rotate_image(ui_handle.pointer_template, ang)
            # 对模板做HSV颜色mask
            hsv_tpl = cv2.cvtColor(tpl_rot, cv2.COLOR_BGR2HSV)
            mask_tpl = cv2.inRange(hsv_tpl, LOWER_ORANGE, UPPER_ORANGE)
            tpl_masked = cv2.bitwise_and(tpl_rot, tpl_rot, mask=mask_tpl)
            # 彩色匹配
            if tpl_masked.shape[2] != img_masked.shape[2]:
                tpl_masked = cv2.cvtColor(tpl_masked, cv2.COLOR_GRAY2BGR)
            res = cv2.matchTemplate(img_masked, tpl_masked, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_angle = ang
                best_tpl = tpl_masked
                best_tpl_shape = tpl_masked.shape

        if best_val < 0.6:  # 阈值可调
            return None, img, mask_img

        h, w = best_tpl_shape[:2]
        top_left = best_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cx = top_left[0] + w // 2
        cy = top_left[1] + h // 2

        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
        cv2.circle(img, (cx, cy), 4, (255, 0, 0), -1)
        cv2.putText(img, f"angle:{best_angle}", (top_left[0], top_left[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        img_h, img_w = img.shape[:2]
        center = (img_w//2, img_h)
        dx, dy = cx - center[0], center[1] - cy
        angle = np.degrees(np.arctan2(dy, dx))

        return angle, img, mask_img

    return None, img, None


# ---------- Region 选择工具 (OpenCV 窗口, 拖动框选) ----------
def select_region_via_drag():
    """
    ��出一个截图窗口，允����������标拖���选择矩形区域������回 region dict 或 None�������
    """
    monitor = sct.monitors[1]
    img = np.array(sct.grab(monitor))[:, :, :3]
    clone = img.copy()
    window_name = "拖动选择区域 - 按回车确认，ESC取消"  # 原英文改为中文

    rect = {"x1":0,"y1":0,"x2":0,"y2":0,"drawing":False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            rect["drawing"] = True
            rect["x1"], rect["y1"] = x, y
            rect["x2"], rect["y2"] = x, y
        elif event == cv2.EVENT_MOUSEMOVE and rect["drawing"]:
            rect["x2"], rect["y2"] = x, y
            temp = clone.copy()
            cv2.rectangle(temp, (rect["x1"], rect["y1"]), (x, y), (0,255,0), 2)
            cv2.imshow(window_name, temp)
        elif event == cv2.EVENT_LBUTTONUP:
            rect["drawing"] = False
            rect["x2"], rect["y2"] = x, y
            temp = clone.copy()
            cv2.rectangle(temp, (rect["x1"], rect["y1"]), (x, y), (0,255,0), 2)
            cv2.imshow(window_name, temp)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    cv2.imshow(window_name, clone)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter confirm
            break
        if key == 27:  # ESC cancel
            cv2.destroyWindow(window_name)
            return None
    cv2.destroyWindow(window_name)
    x1, y1 = rect["x1"], rect["y1"]
    x2, y2 = rect["x2"], rect["y2"]
    left, top = min(x1,x2), min(y1,y2)
    width, height = abs(x2-x1), abs(y2-y1)
    if width == 0 or height == 0:
        return None
    return {"top": top, "left": left, "width": width, "height": height}

# ---------- 自动钓���逻辑 (线程执行) ----------
def automation_loop(ui_handle):
    """
    自动钓鱼主循环：
    - 抛竿（按住 cast_hold_time）
    - 自动通过 bite_region 和模板匹配检测是否咬钩
    - 收杆：实时速度监控
    """
    global automation_running, stop_requested, last_action_text
    sct_thread = mss.mss()

    automation_running = True
    stop_requested = False
    last_action_text = "Automation started"

    try:
        while not stop_requested:
            # 1) 抛竿
            last_action_text = "Casting (hold mouse)"
            ui_handle.set_last_action(last_action_text)
            pyautogui.mouseDown()
            time.sleep(params["cast_hold_time"])
            pyautogui.mouseUp()
            time.sleep(params["post_cast_wait"])

            # 2) 等待有鱼
            last_action_text = "Waiting for bite..."
            ui_handle.set_last_action(last_action_text)
            start_wait = time.time()
            bite_detected = False

            while not stop_requested and not bite_detected:
                # 自动检测咬钩
                img = np.array(sct_thread.grab(bite_region))[:, :, :3]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                bite_template = cv2.imread("bite_template.png", cv2.IMREAD_GRAYSCALE)
                if bite_template is None:
                    raise FileNotFoundError("未找到咬钩模板文件 bite_template.png")

                # 动态生成模板缩放范围
                region_h, region_w = bite_region["height"], bite_region["width"]
                tpl_h, tpl_w = bite_template.shape[:2]
                max_scale = min(region_h / tpl_h, region_w / tpl_w)  # 缩放到检测区域的 100%
                min_scale = max_scale * 0.25  # 当前大小的 25%
                scaled_templates = [
                    cv2.resize(bite_template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                    for scale in np.linspace(min_scale, max_scale, 10)  # 等距取 10 份
                ]

                for tpl in scaled_templates:
                    res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
                    if np.max(res) >= params["bite_threshold"]:  # 阈值可调
                        bite_detected = True
                        break
                if bite_detected:
                    break

                # 如果等待超过 params["max_bite_wait"] 秒，重新抛竿
                if time.time() - start_wait > params["max_bite_wait"]:
                    last_action_text = "No bite detected (timeout) - restarting"
                    ui_handle.set_last_action(last_action_text)
                    time.sleep(1.0)
                    break

                time.sleep(0.12)

            if stop_requested:
                break
            if not bite_detected:
                continue

            # 3) 收杆阶段：实时速度监控循环
            last_action_text = "Reeling: real-time speed monitor"
            ui_handle.set_last_action(last_action_text)

            press_start = None
            last_angle = None
            last_time = None
            no_pointer_time = None
            pointer_lost = False

            while not stop_requested:
                print("收杆：按下")
                pyautogui.mouseDown()
                press_start = time.time()
                released = False

                while not stop_requested:
                    time.sleep(0.1)
                    frame = np.array(sct_thread.grab(pointer_region))[:, :, :3]
                    angle, _, _ = detect_pointer_angle_and_annotate(frame, ui_handle=ui_handle)
                    now = time.time()

                    if angle is None:
                        if no_pointer_time is None:
                            no_pointer_time = now
                        elif now - no_pointer_time > params["pointer_loss_time"]:
                            print(f"{params['pointer_loss_time']}秒内未检测到指针，收杆循环结束")
                            pyautogui.mouseUp()
                            pointer_lost = True
                            break
                        continue
                    else:
                        no_pointer_time = None

                    if angle < params["release_angle"] and not released:
                        print(f"当前角度小于{params['release_angle']}度，松开2秒")
                        pyautogui.mouseUp()
                        released = True
                        time.sleep(2)
                        break

                    if last_angle is not None and last_time is not None:
                        dt = now - last_time if now - last_time > 1e-6 else 1e-6
                        speed = abs(angle - last_angle) / dt
                        print(f"当前速度: {speed:.2f} 度/秒")
                        if speed >= params["release_speed"] and not released:
                            print(f"速度超过{params['release_speed']}，松开0.1秒")
                            pyautogui.mouseUp()
                            released = True
                            time.sleep(0.1)
                            break
                    last_angle = angle
                    last_time = now

                if pointer_lost:
                    # 收杆循环结束后，等待 params["reel_end_wait"] 秒，短按左键 params["short_press_time"] 秒，再立即进入下一轮���竿
                    time.sleep(params["reel_end_wait"])
                    print(f"收杆结束后短按左键{params['short_press_time']}秒")
                    pyautogui.mouseDown()
                    time.sleep(params["short_press_time"])
                    pyautogui.mouseUp()
                    time.sleep(params["next_cast_sleep"])
                    # 立即进入下一轮抛竿（跳出收杆循环，回到主循环）
                    break

                if no_pointer_time is not None and (now - no_pointer_time > 1.0):
                    break

            # 小暂停后��入下一轮
            time.sleep(1.0)

    except Exception as e:
        ui_handle.set_last_action("Automation error: " + str(e))
        print(str(e))
    finally:
        automation_running = False
        stop_requested = False
        ui_handle.set_last_action("Automation stopped")

# ---------- UI 类 ----------
class FishingUI:
    def __init__(self, root):
        global LOWER_ORANGE, UPPER_ORANGE

        self.root = root
        self.root.title("钓鱼助手 (Tkinter)")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # 顶部信息栏
        self.info_label = tk.Label(root, text="角度: -, 速度: -, 状态: 空闲", font=("Consolas", 12))
        self.info_label.pack(padx=6, pady=4)

        # 检测区域显示（上下排）
        self.img_label = tk.Label(root, text="指针检测区域", font=("Consolas", 10), bg="gray")
        self.img_label.pack(padx=6, pady=4)
        self.mask_label = tk.Label(root, text="咬钩检测区域", font=("Consolas", 10), bg="gray")
        self.mask_label.pack(padx=6, pady=4)

        # 参数分组显示
        param_frame = tk.Frame(root)
        param_frame.pack(fill="x", padx=6, pady=4)

        self.entries = {}
        def add_param(group, name, label_text, row):
            lbl = tk.Label(group, text=label_text)
            lbl.grid(row=row, column=0, sticky="w")
            ent = tk.Entry(group, width=6)
            ent.insert(0, str(params[name]))
            ent.grid(row=row, column=1, sticky="w")
            self.entries[name] = ent

        # 抛竿前参数
        cast_frame = tk.LabelFrame(param_frame, text="抛竿前", padx=5, pady=5)
        cast_frame.grid(row=0, column=0, padx=5, pady=5, sticky="n")
        add_param(cast_frame, "cast_hold_time", "抛竿时长(s)", 0)
        add_param(cast_frame, "post_cast_wait", "抛竿后等待(s)", 1)

        # 等上钩时参数
        bite_frame = tk.LabelFrame(param_frame, text="等上钩时", padx=5, pady=5)
        bite_frame.grid(row=0, column=1, padx=5, pady=5, sticky="n")
        add_param(bite_frame, "bite_threshold", "上钩阈值", 0)
        add_param(bite_frame, "max_bite_wait", "上钩等待(s)", 1)

        # 收杆时参数
        reel_frame = tk.LabelFrame(param_frame, text="收杆时", padx=5, pady=5)
        reel_frame.grid(row=0, column=2, padx=5, pady=5, sticky="n")
        add_param(reel_frame, "pointer_loss_time", "指针丢失(s)", 0)
        add_param(reel_frame, "release_angle", "松开角度", 1)
        add_param(reel_frame, "release_speed", "松开速度", 2)

        # 收杆后参数
        post_reel_frame = tk.LabelFrame(param_frame, text="收杆后", padx=5, pady=5)
        post_reel_frame.grid(row=0, column=3, padx=5, pady=5, sticky="n")
        add_param(post_reel_frame, "reel_end_wait", "收杆等待(s)", 0)
        add_param(post_reel_frame, "short_press_time", "短按时间(s)", 1)
        add_param(post_reel_frame, "next_cast_sleep", "下一轮等待(s)", 2)

        # 控制按钮
        btn_frame = tk.Frame(root)
        btn_frame.pack(fill="x", padx=6, pady=4)

        self.btn_detect = tk.Button(btn_frame, text="开始检测", command=self.toggle_detect)
        self.btn_detect.pack(side="left", padx=3)

        self.btn_start_auto = tk.Button(btn_frame, text="启动自动钓鱼 (或按键1)", command=self.toggle_automation)
        self.btn_start_auto.pack(side="left", padx=3)

        self.btn_stop_all = tk.Button(btn_frame, text="停止 (或按3)", fg="red", command=self.stop_all)
        self.btn_stop_all.pack(side="right", padx=3)

        # 设置区域按钮
        region_frame = tk.Frame(root)
        region_frame.pack(fill="x", padx=6, pady=4)

        self.btn_set_pointer_region = tk.Button(region_frame, text="设置指针区域", command=self.set_pointer_region)
        self.btn_set_pointer_region.pack(side="left", padx=3)

        self.btn_set_bite_region = tk.Button(region_frame, text="设置咬钩区域", command=self.set_bite_region)
        self.btn_set_bite_region.pack(side="left", padx=3)

        # 状态显示
        self.last_action_var = tk.StringVar(value="空闲")
        tk.Label(root, textvariable=self.last_action_var, font=("Consolas", 11)).pack(padx=6, pady=4)

        # internal:
        self.prev_angle = None
        self.prev_time = None
        self.current_speed = None

        # 移除 tkinter 键盘绑定
        # root.bind('<Key>', self.on_key)

        # 全局按键监听
        keyboard.add_hotkey('1', self.toggle_automation)
        keyboard.add_hotkey('3', self.stop_all)

        # detection timer
        self._after_id = None
        self._detecting = False

        # 初始化时直接加载指针模板
        try:
            tpl = cv2.imread("pointer_template.png", cv2.IMREAD_COLOR)
            if tpl is None:
                raise FileNotFoundError("请将指针模板命名为 pointer_template.png 并放到当前脚本目录")
            self.pointer_template = tpl
            self.pointer_h, self.pointer_w = tpl.shape[:2]
            print("指针模板已成功加载")
        except Exception as e:
            messagebox.showerror("错误", f"加载指针模板失败: {e}")
            self.pointer_template = None

    # ---------- UI helper methods ----------
    def set_last_action(self, text):
        self.last_action_var.set(text)

    def toggle_detect(self):
        if self._detecting:
            self._detecting = False
            if self._after_id is not None:
                self.root.after_cancel(self._after_id)
            self.btn_detect.config(text="开始检测")  # 改为中文
            self.set_last_action("检测已停止")  # 改为中文
        else:
            self.sync_params_from_entries()
            self._detecting = True
            self.btn_detect.config(text="停止检测")  # 改为中文
            self.set_last_action("检测进行中")  # 改为中文
            self._loop_detect()

    def sync_params_from_entries(self):
        # ��� UI entries 写回 params
        for name, ent in self.entries.items():
            try:
                val = float(ent.get()) if '.' in ent.get() else int(ent.get())
                params[name] = val
            except Exception:
                pass

    def toggle_automation(self):
        global automation_running, stop_requested
        if automation_running:
            stop_requested = True
            self.set_last_action("正在停止自动钓鱼...")  # 改为中文
        else:
            self.sync_params_from_entries()
            stop_requested = False
            t = threading.Thread(target=automation_loop, args=(self,), daemon=True)
            t.start()
            self.set_last_action("自动钓鱼线程已启动 (按3可停止)")  # 改为中文

    def consume_manual_bite_flag(self):
        # called by automation thread: consume a manual bite flag if set
        global bite_mode_manual
        if bite_mode_manual:
            bite_mode_manual = False
            return True
        return False

    def stop_all(self):
        global stop_requested
        stop_requested = True
        if self._detecting:
            self.toggle_detect()
        self.set_last_action("已请求停止")  # 改为中文

    def on_close(self):
        self.stop_all()
        time.sleep(0.2)
        keyboard.unhook_all_hotkeys()  # 关闭全局热键监听
        self.root.destroy()

    def set_pointer_region(self):
        """设置指针检测区域"""
        global pointer_region
        new_region = select_region_via_drag()
        if new_region:
            pointer_region = new_region
            messagebox.showinfo("成功", f"指针区域已更新为: {pointer_region}")
        else:
            messagebox.showwarning("���消", "未更新指针区域")

    def set_bite_region(self):
        """设置咬���检测区域"""
        global bite_region
        new_region = select_region_via_drag()
        if new_region:
            bite_region = new_region
            messagebox.showinfo("成��", f"咬钩区域已更新为: {bite_region}")
        else:
            messagebox.showwarning("取消", "未更新咬钩区域")

    # ---------- detection loop (UI) ----------
    def _loop_detect(self):
        # 1) grab frame, detect pointer, update angle/speed display and images
        frame = np.array(sct.grab(pointer_region))[:, :, :3]
        angle, annotated, mask = detect_pointer_angle_and_annotate(frame, ui_handle=self)

        # ���钩检测可视化
        bite_frame = np.array(sct.grab(bite_region))[:, :, :3]
        bite_frame = cv2.cvtColor(bite_frame, cv2.COLOR_BGRA2BGR)  # 确保为 BGR 格式
        gray_bite = cv2.cvtColor(bite_frame, cv2.COLOR_BGR2GRAY)
        bite_template = cv2.imread("bite_template.png", cv2.IMREAD_GRAYSCALE)
        if bite_template is not None:
            # ���模板进行多级缩放处理
            scaled_templates = [
                cv2.resize(bite_template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                for scale in np.arange(0.5, 3.5, 0.5)  # 从 50% 到 300%，每 50% 一个模板
            ]
            bite_detected = False
            for tpl in scaled_templates:
                res = cv2.matchTemplate(gray_bite, tpl, cv2.TM_CCOEFF_NORMED)
                if np.max(res) >= 0.78:  # 阈值可调
                    bite_detected = True
                    break
            if bite_detected:
                print("咬钩检测到！")
                cv2.putText(bite_frame, "Bite Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        now = time.time()
        speed_txt = "-"
        if angle is not None and self.prev_angle is not None and self.prev_time is not None:
            dt = now - self.prev_time if now - self.prev_time > 1e-6 else 1e-6
            sp = (self.prev_angle - angle) / dt
            self.current_speed = sp
            speed_txt = f"{sp:.2f} 度/秒"  # 改为中文
        elif self.current_speed is not None:
            speed_txt = f"{self.current_speed:.2f} 度/秒"  # 改为中文

        angle_txt = f"{angle:.2f}" if angle is not None else "-"

        status = "自动钓鱼运行中" if automation_running else "空闲"  # 改为中文
        self.info_label.config(text=f"角度: {angle_txt} , 速度: {speed_txt} , 状态: {status}")  # 改为中文

        # show annotated frame
        try:
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img = img.resize((int(pointer_region["width"]*1.2), int(pointer_region["height"]*1.2)))
            imgtk = ImageTk.PhotoImage(image=img)
            self.img_label.imgtk = imgtk
            self.img_label.config(image=imgtk)
        except Exception:
            pass

        # show mask
        try:
            if mask is not None:
                mvis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                mvis = Image.fromarray(mvis)
                mvis = mvis.resize((int(pointer_region["width"]*1.2), int(pointer_region["height"]*1.2)))
                mtk = ImageTk.PhotoImage(image=mvis)
                self.mask_label.imgtk = mtk
                self.mask_label.config(image=mtk)
        except Exception:
            pass

        # 显示咬钩检测区域
        try:
            rgb_bite = cv2.cvtColor(bite_frame, cv2.COLOR_BGR2RGB)
            img_bite = Image.fromarray(rgb_bite)
            img_bite = img_bite.resize((int(bite_region["width"] * 1.2), int(bite_region["height"] * 1.2)))
            imgtk_bite = ImageTk.PhotoImage(image=img_bite)
            self.mask_label.imgtk = imgtk_bite
            self.mask_label.config(image=imgtk_bite)
        except Exception:
            pass

        # keep prev values
        self.prev_angle = angle
        self.prev_time = now

        # schedule next
        if self._detecting:
            self._after_id = self.root.after(FRAME_INTERVAL_MS, self._loop_detect)
        else:
            self._after_id = None

# ---------- main ----------
def main():
    root = tk.Tk()
    app = FishingUI(root)
    # show quick hint
    message = ("提示：\n"
               " - 按��� '1' 启动/停止自动钓鱼 (也可点击启动自动钓鱼)\n"
               " - 按键 '2' 手动触发“有鱼”事件（会被自动钓鱼线程消费）\n"
               " - 按键 '3' 立即停止自动钓鱼\n"
               " - 如果需要自动检���感叹号（咬钩），请用“设置咬钩区域”选区并将模板命名为 bite_template.png\n"
               " - 调节参数后建议先“开始检测”观察识别效果，再启用自动钓鱼\n")  # 全部改为中文
    print(message)
    root.mainloop()
    keyboard.unhook_all_hotkeys()  # 程序退出时清理

if __name__ == "__main__":
    main()
