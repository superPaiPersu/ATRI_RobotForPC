import math
import json
import multiprocessing
import os
import queue
import random
import signal
import sys
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np

import threading
import time
from datetime import datetime
import speech_recognition as sr

import easygui
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QPushButton, QLabel, QSlider, QHBoxLayout, QMenuBar, QMenu, QComboBox)
from PyQt6.QtCore import Qt, QTimer, QSize, QRect, pyqtSignal, QPointF, QRectF
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QIcon, QAction, QFont, QPainter, QColor, QPainterPath, QPen, QBrush, QFontDatabase, \
    QLinearGradient, QImage, QCursor, QPalette, QMovie, QActionGroup
from PyQt6.QtCore import QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtWidgets import QApplication, QPushButton

import live2d.v3 as live2d
from OpenGL.GLUT import *
from threading import Thread
from voicerecver import *
from voice_producer import *
import audio_prepare
import pygame
import ncm
import AI_tools


def get_arg(name, defalut=""):
    """返回桌宠资源根路径：优先使用 --pet-dir 参数，否则用脚本所在目录"""
    for i, arg in enumerate(sys.argv):
        if arg == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return defalut


basepath = get_arg("--pet-dir")
pet_id = get_arg("--pet-id")
if not basepath:
    raise RuntimeError("--pet-dir is required")
basepath = os.path.abspath(basepath) + os.sep

WIDTH = 400
HEIGHT = 800
POSX = 1300
POSY = 550
RATE = 60
SCALE_FACTOR = 1
MUSIC_EVENT = pygame.USEREVENT + 1

os.environ['SDL_VIDEO_WINDOW_POS'] = "{},{}".format(POSX, POSY)

pygame.init()
live2d.init()


def make_chat_message(role, text):
    return {"role": role, "parts": [{"text": text}]}


def _history_event_to_messages(event):
    messages = []
    if not isinstance(event, dict):
        return messages

    event_type = event.get("type") or ""
    started_at = event.get("started_at")
    ended_at = event.get("ended_at")
    marker = event.get("marker") or event.get("source") or ""

    if event_type.endswith("session_start") and started_at:
        messages.append({
            "role": "developer",
            "content": "【历史会话信息】一段历史对话开始于 {}，记录标记：{}。".format(started_at, marker),
        })

    for message in event.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role") or "user"
        if role == "model":
            role = "assistant"
        content = message.get("content")
        if content is None:
            content = message.get("cn") or message.get("text")
        content = str(content or "").strip()
        if content:
            messages.append({"role": role, "content": content})

    if event_type.endswith("session_end") and ended_at:
        messages.append({
            "role": "developer",
            "content": "【历史会话信息】上一段历史对话结束于 {}。".format(ended_at),
        })

    return messages


def parse_dialogue_history_text(text):
    messages = []
    if not text or not text.strip():
        return messages

    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        event = json.loads(line)
        messages.extend(_history_event_to_messages(event))

    return messages


def parse_model_reply_text(text, voice_refs, expressions):
    payload = json.loads(str(text or "").strip())
    if not isinstance(payload, dict):
        raise ValueError("model reply must be a JSON object")

    action = str(payload.get("action") or "reply").strip()
    if action not in ("reply", "wait", "history_require"):
        raise ValueError("unsupported model reply action: {}".format(action))

    if action != "reply":
        content = str(payload.get("content") or action).strip()
        return {
            "action": action,
            "content": content,
        }

    cn_text = str(payload.get("cn") or payload.get("content") or "").strip()
    jp_text = str(payload.get("jp") or "").strip()
    if not cn_text or not jp_text:
        raise ValueError("reply action requires cn and jp")

    mood = str(payload.get("mood") or "").strip()
    expression = str(payload.get("expression") or "").strip()
    heart_beat_seconds = str(payload.get("heart_beat_seconds") or "").strip()
    special = str(payload.get("special") or "").strip()
    if mood not in voice_refs:
        mood = voice_refs[0] if voice_refs else "normal"
    if expression not in expressions:
        expression = "normal" if "normal" in expressions else (expressions[0] if expressions else "")

    return {
        "action": "reply",
        "content": cn_text,
        "cn": cn_text,
        "jp": jp_text,
        "mood": mood,
        "expression": expression,
        "heart_beat_seconds": heart_beat_seconds,
        "special": special
    }


def history_messages_to_jsonl(messages, started_at, ended_at, marker):
    rows = [
        {
            "type": "dialogue_session_start",
            "started_at": started_at,
            "marker": marker,
            "messages": [],
        }
    ]

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = str(message.get("content") or "").strip()
        if not role or not content:
            continue

        item = {
            "role": role,
            "content": content,
        }
        for key in ("action", "mood", "expression"):
            value = message.get(key)
            if value:
                item[key] = value
        rows.append({
            "type": "dialogue_message",
            "messages": [item],
        })

    rows.append({
        "type": "dialogue_session_end",
        "ended_at": ended_at,
        "messages": [],
    })
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


class GLWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.is_loading_end = False
        self.setGeometry(POSX, POSY, WIDTH, HEIGHT)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("ATRI")
        self.loop_timer = QTimer()
        self.loop_timer.timeout.connect(self.update)
        self.loop_timer.start(int(1 / 60 * 1000))

    def handle_shutdown_signal(self, signum=None, frame=None):
        self.save_log()
        sys.exit(0)

    def threads_exception_handler(self, args):
        print("捕获到未处理的异常，程序即将崩溃，执行紧急清理...")
        print(args)
        # 在这里可以进行资源释放、记录错误日志等操作
        # 调用原始的 excepthook 以确保正常的错误信息仍然会打印出来
        self.save_log()
        sys.exit()

    def main_exception_handler(self, exc_type, exc_val, exc_tb):
        print("捕获到未处理的异常，程序即将崩溃，执行紧急清理...")
        print(exc_type, exc_val, exc_tb)
        # 在这里可以进行资源释放、记录错误日志等操作
        # 调用原始的 excepthook 以确保正常的错误信息仍然会打印出来
        self.save_log()
        sys.exit()

    def ITimer(self, seconds, fun):
        while True:
            time.sleep(seconds)
            fun()

    def initializeGL(self) -> None:

        live2d.glInit()

        self.expression_lt = []
        self.param_dt = {}
        self.load()

        self.upload_file_path = ""
        self.shake_range = 0
        # self.mouth_open_dict = {"shy": 0.3, "happy": 0.8, "得意": 0.8, "sad": 0.4, "normal": 0.6, "confuse": 0.6,
        #                         "气急败坏": 0.8, "赌气": 0.8, "委屈想哭": 0.3, "彻底坏掉": 0.3}  # 不同心情嘴张多大
        # self.body_shake_range_dict={"shy":2,"happy":8,"得意":8}
        self.hair_flying = True
        self.in_shake_body = False
        self.is_on_chat = False
        self.is_voice_flag = False
        self.in_initiative = False
        self.mood_active = False
        self.is_speaking = False
        self.is_user_turn = False
        self.is_saving_log = False
        self.is_evil_mode = False
        self.mood_status = ""
        self.now_expression = ""
        self.now_text = ""
        self.latest_input = ""
        self.all_prefab_audio = []
        self.all_music = {}
        self.now_music = []
        self.music_on = True
        self.on_change_music = False
        self.is_ncm_loaded = False
        self.is_screen_monitor = False
        self.in_like = False
        self.in_daily = True
        self.on_auto_play = True
        self.now_playing_sound = None
        self.in_moving = False
        self.last_cloth = None

        self.cloth_lt = [18, 9, 19, 20]

        self.init_music()
        self.config_llm()
        self.set_menu()
        self.update_ui()
        self.music_play_over_check()
        self.awake_threads()
        self.motions_lt = self.model.GetMotionGroups()
        self.set_model_deepseek()

        self.is_loading_end = True
        sys.excepthook = self.main_exception_handler
        threading.excepthook = self.threads_exception_handler
        signal.signal(signal.SIGINT, self.handle_shutdown_signal)
        signal.signal(signal.SIGTERM, self.handle_shutdown_signal)

        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, self.handle_shutdown_signal)

    def load_init_music(self):
        lt = []
        for i in os.listdir(basepath + "init_musics"):
            name = i.split(".")[0]
            lt.append([basepath + "init_musics/" + i, name, 0])
        self.all_music["init"] = lt

    def init_music(self):
        self.prefab_audio_index = 0

        for i in os.listdir(basepath + "config/prefab_audio"):
            pt = basepath + "config/prefab_audio/" + i
            self.all_prefab_audio.append([i.split(".")[0], pygame.mixer.Sound(pt), pt])

        pygame.mixer.music.set_volume(0.2)

        self.load_init_music()

        path = "../cookie.json"
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    cookie = json.load(f).get("cookie")
                ncm.init(cookie)
                self.load_ncm_music()
            else:
                self.load_init_music()
        except Exception as e:

            print("加载网易云发生了异常错误或未连接网易云，故加载默认音乐")
            self.load_init_music()

    def set_chat_mode(self):
        self.is_on_chat = not self.is_on_chat

    def set_voice_flag(self):
        self.is_voice_flag = not self.is_voice_flag

    def set_text(self, text):
        self.now_text = text

    def change_expression_motion(self):
        value = self.expression_effect_dict[self.now_expression]["shake"]
        if value != 0:
            self.in_shake_body = True
            self.shake_range = value

    def load_model_custom_data(self):
        with open(basepath + "config/custom.json", "r", encoding="utf-8") as f:
            self.user_custom_dict = json.load(f)

    def load_model_expression_data(self):
        self.expression_effect_dict = {}
        with open(basepath + "config/expression_effect.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            self.expression_effect_dict: dict = data["expressions"]
            self.effect_label_dict = data["paramID"]
        for i in self.expression_effect_dict.keys():
            self.expression_lt.append(i)

    def load_voice_categories(self):
        lt = os.listdir(basepath + "config/voice_refs/")
        self.voice_refs_lt = []
        for i in lt:
            self.voice_refs_lt.append(i.split(".")[0])
        if len(self.voice_refs_lt) == 0:
            print("无参考音频")
            sys.exit()
        self.ref_voice_format = lt[0].split(".")[1]

    def draw_text(self):
        painter = QPainter(self)
        # painter.begin(self)
        font = QFont('Arial', int(16 * SCALE_FACTOR))
        if self.now_text != "":
            text_rect = QRect(0, int(50 * SCALE_FACTOR), int(150 * SCALE_FACTOR), int(350 * SCALE_FACTOR))
            painter.setFont(font)
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWrapAnywhere, self.now_text)
        if self.music_on:
            text_rect = QRect(int(300 * SCALE_FACTOR), int(50 * SCALE_FACTOR), int(100 * SCALE_FACTOR),
                              int(350 * SCALE_FACTOR))
            painter.setFont(font)
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWrapAnywhere, self.now_music[1])  #
        painter.end()

    def save_log(self, is_summary=True):

        if not self.chat_begin_time or self.is_saving_log:
            return

        save_start = self.last_saved_dialogue_history_end
        save_end = len(self.all_dialogue_history)
        new_messages = self.all_dialogue_history[save_start:save_end]
        if not new_messages:
            return

        print("准备保存历史")
        self.is_saving_log = True
        try:
            ended_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            txt = history_messages_to_jsonl(
                new_messages,
                self.chat_begin_time,
                ended_at,
                "唤醒model--PC端",
            )
            with open(basepath + "config/dialogue_history.txt", "a", encoding="utf-8") as f:
                f.write(txt)
            result = self.AI_tool.save_dialogue_history(txt)
            if result != "success":
                print("保存聊天历史失败:", result)
                return
            self.last_saved_dialogue_history_end = save_end
            self.current_session_history_start = save_end
            if is_summary:
                try:
                    summary_result = self.save_summary_update(new_messages)
                    if summary_result != "success":
                        print("保存总结历史失败:", summary_result)
                except Exception as e:
                    print("保存总结历史异常:", e)
                sys.exit()
            self.chat_begin_time = None
        finally:
            self.is_saving_log = False
            # with open(basepath + "config/chat_history.txt", "a", encoding="utf-8") as f:
            #     f.write(txt)

    def save_summary_update(self, messages=None):
        if messages is None:
            prompt = self.create_prompt(False)
        else:
            prompt = self.preset_prompt + self.summary_prompt + messages + self.rule_prompt
        prompt = prompt + [{"role": "user", "content": "[系统消息]:这是系统提示，对话即将结束。"}]
        return self.AI_tool.update_summary_history(prompt, self.chat_begin_time)

    def load_summary_history(self):
        ans = self.AI_tool.get_summary_history()
        self.summary_prompt = []
        if ans[0] == "success":
            self.summary_prompt.extend(parse_dialogue_history_text(ans[1]))
            self.summary_prompt.append(
                {"role": "developer", "content": "接下来是我们近期的聊天记录，有助于你了解最近发生了什么。"})
        else:
            print("bakaYou!")
            sys.exit()

    def load_dialogue_history(self):
        ans = self.AI_tool.get_dialogue_history()
        self.all_dialogue_history = []
        if ans[0] == "success":
            self.all_dialogue_history.extend(parse_dialogue_history_text(ans[1]))
            self.current_session_history_start = len(self.all_dialogue_history)
            self.last_saved_dialogue_history_end = self.current_session_history_start
            print("加载聊天历史，共{}条".format(len(self.all_dialogue_history)))
        else:
            print("baka!")
            sys.exit()

    def get_preset_prompt(self):
        with open(basepath + "config/rules.txt", "r", encoding="utf-8") as f:
            rules = f.read()

            self.rule_prompt = [{"role": "developer", "content": "下面为规则重申"}]
            self.rule_prompt.append({"role": "developer", "content": rules})
            self.rule_prompt.append(
                {"role": "developer", "content": "这是可选心情列表————" + str(self.voice_refs_lt)})
            self.rule_prompt.append(
                {"role": "developer", "content": "这是可选表现列表————" + str(self.expression_lt)})
        self.preset_prompt = []

        self.preset_prompt += self.rule_prompt
        with open(basepath + "config/personal_setting.txt", "r", encoding="utf-8") as f:
            self.preset_prompt.append({"role": "developer", "content": f.read()})

        self.preset_prompt.append(
            {"role": "developer", "content": "接下来是我们对话历史的精炼版，你可以从中参考获得更详细的人设定位。"})

    def create_prompt(self, is_all_history):

        if is_all_history:
            return self.preset_prompt + self.summary_prompt + self.all_dialogue_history + [
                {"role": "user", "content": "[系统消息]:你已经请求并且获得了全部对话历史"}] + self.rule_prompt
        # 把前50条内容也一起发过去,当作近期记忆
        current_memory_num = 50
        start_index = max(0, self.current_session_history_start - current_memory_num)

        return self.preset_prompt + self.summary_prompt + self.all_dialogue_history[start_index:] + self.rule_prompt

        # return self.all_dialogue_history

    def config_llm(self):

        self.voice_msg_queue = queue.Queue()
        self.chat_begin_time = None
        self.get_preset_prompt()
        self.load_summary_history()
        self.load_dialogue_history()

    def play_sound(self, sound, filepath="voice.wav"):

        self.now_playing_sound = AudioData(sound, audio_prepare.analyze_wav(filepath), sound.play())
        self.is_speaking = True
        return sound.get_length()

    def random_music(self):
        choice_lt = self.all_music["init"]
        if self.is_ncm_loaded:

            if self.in_daily:
                choice_lt = self.all_music["日推歌单"]
            if self.in_like:
                choice_lt = self.all_music["红心歌单"]

        legal_lt = [i for i in choice_lt if i[2] == 0]
        if len(legal_lt) > 0:
            new = random.choice(legal_lt)
            new[2] = 1
            self.now_music = new
            if self.is_ncm_loaded:
                self.change_music(self.now_music[0])
            else:
                pygame.mixer.music.load(new[0])
                pygame.mixer.music.play()
        else:
            for i in choice_lt:
                i[2] = 0
            self.random_music()

    def music_ctrl(self):
        while True:
            if self.on_change_music:
                self.on_change_music = False
                self.random_music()

            self.music_play_over_check()
            time.sleep(1 / 60)

    def load_ncm_music(self):
        self.is_ncm_loaded = True
        self.all_music["红心歌单"] = ncm.get_like_music_lt()
        self.all_music["happy"] = ncm.get_music_lt("萝卜子高兴")
        self.all_music["normal"] = ncm.get_music_lt("萝卜子日常")
        self.all_music["sad"] = ncm.get_music_lt("萝卜子伤心")
        self.all_music["angry"] = ncm.get_music_lt("萝卜子生气")
        self.all_music["shy"] = ncm.get_music_lt("萝卜子害羞")
        self.all_music["confuse"] = ncm.get_music_lt("日推歌单")
        self.all_music["日推歌单"] = ncm.get_music_lt("日推歌单")

    #
    def resizeEvent(self, event):
        if self.is_loading_end:
            self.update_ui()
        super().resizeEvent(event)

    def add_menu(self, menu, name, solution, checkable=False, toggle=False):
        button = QAction(name, self)
        button.setCheckable(checkable)
        if checkable:
            button.setChecked(toggle)

        button.triggered.connect(solution)

        menu.addAction(button)
        return button

    def change_music(self, id):

        pygame.mixer.music.unload()
        fn = ncm.get_music(id)

        pygame.mixer.music.load(fn)
        pygame.mixer.music.play()

    def check_music(self):
        self.music_on = not self.music_on
        if self.music_on == False:
            pygame.mixer.music.pause()
        else:

            pygame.mixer.music.unpause()

    def change_music_clicked(self):
        self.on_change_music = True

        #

    def get_random_sound(self):
        if self.is_speaking == False:
            self.set_text(self.all_prefab_audio[self.prefab_audio_index][0])
            self.play_sound(self.all_prefab_audio[self.prefab_audio_index][1],
                            self.all_prefab_audio[self.prefab_audio_index][2])

            self.prefab_audio_index += 1
            if self.prefab_audio_index == len(self.all_prefab_audio):
                self.prefab_audio_index = 0

    def auto_play(self):
        self.on_auto_play = not self.on_auto_play

    def random_expression(self):

        self.model.SetRandomExpression()

    def on_scale_slider(self):

        self.model.Resize(int(WIDTH * SCALE_FACTOR), int(HEIGHT * SCALE_FACTOR))
        self.resize(QSize(int(WIDTH * SCALE_FACTOR), int(HEIGHT * SCALE_FACTOR)))

    def update_ui(self):

        self.move_button.resize(QSize(int(30 * SCALE_FACTOR), int(30 * SCALE_FACTOR)))
        self.move_button.setIconSize(QSize(int(30 * SCALE_FACTOR), int(30 * SCALE_FACTOR)))
        self.move_button.move(int(50 * SCALE_FACTOR), 0)

        self.loading_gif_movie.setScaledSize(QSize(int(60 * SCALE_FACTOR), int(60 * SCALE_FACTOR)))
        self.loading_gif.resize(QSize(int(60 * SCALE_FACTOR), int(60 * SCALE_FACTOR)))
        self.loading_gif.move(int(280 * SCALE_FACTOR), int(20 * SCALE_FACTOR))

        self.menu_button.resize(QSize(int(60 * SCALE_FACTOR), int(35 * SCALE_FACTOR)))
        self.menu_button.setIconSize(QSize(int(30 * SCALE_FACTOR), int(30 * SCALE_FACTOR)))
        self.menu_button.move(int(220 * SCALE_FACTOR), 0)

    def send_file(self):
        try:
            if self.is_on_chat:
                file_path = easygui.fileopenbox(msg="选择文件")
                if file_path != "" and file_path:
                    self.upload_file_path = file_path
        except:
            print("上传文件失败！")

    def allow_move_window(self):
        self.in_moving = True
        self.window_origin_pos = self.pos()
        self.mouse_origin_pos = QCursor.pos()

    def save_log_async(self, flag1):
        Thread(target=self.save_log, args=(flag1,)).start()

    def toggle_initiative(self):
        self.last_chat_time = time.time()
        self.in_initiative = not self.in_initiative
        self.is_user_turn = False

    def evil(self):
        self.is_evil_mode = not self.is_evil_mode
        if self.is_evil_mode:
            self.AI_tool.model_name = "gpt"
        else:
            self.AI_tool.model_name = "deepseek"

    def set_model_deepseek(self):
        self.AI_tool.api_url = "https://ai-pixel.online/v1"
        self.AI_tool.api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        self.AI_tool.model_name = "deepseek"

    def set_model_gemini(self):
        self.AI_tool.api_url = "https://api.yomiapi.com/v1"
        self.AI_tool.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.AI_tool.model_name = "gemini"

    def set_model_gpt(self):
        self.AI_tool.api_url = "https://speed.ai-pixel.online/v1"
        self.AI_tool.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.AI_tool.model_name = "gpt"

    def set_menu(self):
        self.menu = CustomMenu()
        self.add_menu(self.menu, "开启对话", self.set_chat_mode, True)
        self.add_menu(self.menu, "主动模式", self.toggle_initiative, True)
        self.add_menu(self.menu, "语音输入模式", self.set_voice_flag, True)
        self.add_menu(self.menu, "上传文件", self.send_file, False)
        self.add_menu(self.menu, "开启音乐", self.check_music, True, True)
        if self.is_ncm_loaded:
            self.like_button = self.add_menu(self.menu, "红心漫游", self.like_music, True)
            self.recommend_button = self.add_menu(self.menu, "日推抽奖", self.daily_recommend, True, True)
        self.add_menu(self.menu, "自动播放下一曲", self.auto_play, True, True)
        self.add_menu(self.menu, "随机切换音乐", self.change_music_clicked, False)
        self.add_menu(self.menu, "随机变换", self.random_expression, False)
        self.add_menu(self.menu, "保存对话历史", self.save_log_async, False)

        model_menu = CustomMenu2("模型选择", self.menu)
        self.menu.addMenu(model_menu)

        group = QActionGroup(self)
        group.addAction(self.add_menu(model_menu, "gpt", self.set_model_gpt, True, False))
        group.addAction(self.add_menu(model_menu, "gemini", self.set_model_gemini, True, False))
        group.addAction(self.add_menu(model_menu, "deepseek", self.set_model_deepseek, True, True))
        group.setExclusive(True)

        self.menu_button = QPushButton(self)
        self.menu_button.setMenu(self.menu)

        self.menu_button.setIcon(QIcon(basepath + "img/setting.png"))

        self.move_button = QPushButton(self)
        self.move_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;  /* 透明背景 */
                border: none;                   /* 无边框 */

            }
        """)
        self.move_button.pressed.connect(self.allow_move_window)
        self.move_button.setIcon(QIcon(basepath + "img/pull.png"))

        self.loading_gif = QLabel(self)
        self.loading_gif_movie = QMovie(basepath + "img/loading.gif")
        self.loading_gif_movie.start()
        self.loading_gif.setMovie(self.loading_gif_movie)
        self.loading_gif.hide()

        self.menu.setStyleSheet("""

                        /* 菜单项 */
                        QMenu {
                            icon-size:1000px;
                        }
                        QMenu::item {
                            background-origin: content;
                            border-image:url(:/play2.png)0 0 0 0;
                            border-width:0px 0px 0px 0px;

                            background-repeat: no-repeat;
                            text-decoration: none;

                            font-size: 16px;
                            font-family: 微软雅黑,宋体, Arial, Helvetica, Verdana, sans-serif;
                            font-weight: bold;

                            border-radius: 3px;

                        }

                        /* 鼠标悬停时的菜单项 */
                        QMenu::item:selected {
                            background-color: #0d6efd;
                            color: black;
                        }
                        /*开启*/
                        QMenu::item:checked {
                            background-color: green;
                            color: black;
                        }
                        /* 禁用的菜单项 */
                        QMenu::item:disabled {
                            color: #adb5bd;
                            background-color: transparent;
                        }

                        /* 分隔线 */
                        QMenu::separator {
                            height: 1px;
                            background-color: #dee2e6;
                            margin: 4px 0;
                        }
                    """)

    def voice_recv(self, msg_queue, sample_rate=16000, silence_duration=1.2, max_record_seconds=12.0):

        import http.client

        stt_host = "127.0.0.1"
        stt_port = 18765

        def transcribe_audio(audio):
            data = audio.astype(np.float32, copy=False).tobytes()
            conn = http.client.HTTPConnection(stt_host, stt_port, timeout=45)
            try:
                conn.request(
                    "POST",
                    "/transcribe",
                    body=data,
                    headers={
                        "Content-Length": str(len(data)),
                        "Content-Type": "application/octet-stream",
                        "X-Sample-Rate": str(sample_rate),
                    },
                )
                response = conn.getresponse()
                payload = json.loads(response.read().decode("utf-8"))
            finally:
                conn.close()
            if "error" in payload:
                raise RuntimeError(payload["error"])
            return payload.get("text", "").strip()

        frames_per_check = int(sample_rate * 0.1)  # 每100ms检查一次

        start_threshold = 0.03  # 开始说话阈值，避免静音片段反复送入模型
        silence_threshold = 0.02  # 结束静音阈值，按当前训练音频响度调低
        speech_start_frames = 2
        pre_buffer_frames = 3
        print("语音阈值:", start_threshold, silence_threshold)
        # no_speech_threshold = 0.6
        while True:

            time.sleep(1 / 60)

            if self.is_voice_flag:
                buffer = []
                pre_buffer = []
                silent_frames = 0
                speech_frames = 0
                speech_started = False
                record_started_at = None
                with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
                    while self.is_voice_flag:
                        frame, _ = stream.read(frames_per_check)
                        frame_mean = float(np.abs(frame).mean())

                        if not speech_started:
                            pre_buffer.append(frame.copy())
                            if len(pre_buffer) > pre_buffer_frames:
                                pre_buffer.pop(0)

                            if frame_mean >= start_threshold:
                                speech_frames += 1
                            else:
                                speech_frames = 0

                            if speech_frames >= speech_start_frames:
                                speech_started = True
                                record_started_at = time.time()
                                print("开始说话...")
                                buffer.extend(pre_buffer)
                                pre_buffer = []
                                silent_frames = 0
                            continue

                        buffer.append(frame.copy())

                        # 检测是否静音
                        if frame_mean < silence_threshold:
                            silent_frames += 1
                        else:
                            silent_frames = 0

                        # 静音超过 silence_duration 秒，认为说完了
                        if silent_frames > (silence_duration / 0.1):
                            break
                        if record_started_at and time.time() - record_started_at > max_record_seconds:
                            print("录音达到最大时长，强制结束")
                            break
                if not buffer:
                    continue
                t = time.time()
                audio = np.concatenate(buffer).flatten()
                print("录音结束，音频长度{0:.2f}秒".format(len(audio) / sample_rate))
                if len(audio) < sample_rate * 0.3:  # 少于0.3秒丢弃
                    continue
                print("发送音频到STT HTTP服务")
                try:
                    ans = transcribe_audio(audio)
                except Exception as e:
                    print("STT HTTP识别失败:", e)
                    ans = "......"
                if ans == " " or ans == "":
                    ans = "......"
                print("语音识别耗时{0}秒，结果为{1}".format(time.time() - t, ans))
                msg_queue.put(ans)

    #
    # def voice_recv(self, msg_queue: queue.Queue):
    #
    #     r = sr.Recognizer()
    #     mic = sr.Microphone(sample_rate=16000)
    #
    #
    #     with open("test.pcm","ab", buffering=0) as f:
    #         def write_in_pcm(indata, frames, time, status):
    #             pcm = (indata * 32767).astype(np.int16).tobytes()
    #             f.write(pcm)
    #
    #
    #
    #     while True:
    #         time.sleep(1 / 60)
    #         if self.is_voice_flag:
    #             with mic as source:
    #                 r.adjust_for_ambient_noise(source, 1)
    #                 print("录音中...")
    #                 try:
    #                     audio = r.listen(source, timeout=20).get_raw_data()
    #
    #                 except sr.WaitTimeoutError:
    #                     return "......"
    #                 print("录音结束")
    #                 t = time.time()
    #                 result = self.AI_tool.translate(audio)
    #                 if result == " " or result == "":
    #                     result = "......"
    #                 print("语音识别耗时{0}秒，结果为{1}".format(time.time() - t, result))
    #                 msg_queue.put(result)
    #

    def music_play_over_check(self):
        if pygame.mixer.music.get_busy() == False and self.music_on:
            if self.on_auto_play:
                print("结束音乐！")
                self.random_music()
            else:
                pygame.mixer.music.play()

    def awake_threads(self):
        Thread(target=self.voice_recv, args=(self.voice_msg_queue,), daemon=True).start()
        Thread(target=self.chat, daemon=True).start()
        Thread(target=self.speaking, daemon=True).start()
        Thread(target=self.music_ctrl, daemon=True).start()
        Thread(target=self.shake_body, daemon=True).start()

    def speaking(self):
        mouth_param_label = self.effect_label_dict["mouthParam"]
        while True:
            if self.is_speaking:

                if self.now_playing_sound.check_over():

                    self.is_speaking = False
                    self.model.SetParameterValue(mouth_param_label, 0)
                else:
                    if self.now_expression == "":
                        self.model.SetParameterValue(mouth_param_label, 0.5
                                                     * self.now_playing_sound.get_realtime_react())
                    else:
                        self.model.SetParameterValue(mouth_param_label, self.expression_effect_dict[
                            self.now_expression]["mouth_open"] * self.now_playing_sound.get_realtime_react())
            else:
                time.sleep(1)

    def show_expression(self, expression):
        self.model.ResetExpression()
        self.model.SetExpression(expression)

    def chat_solution(self, is_all_history=False):

        self.loading_gif.show()
        if self.upload_file_path != "":

            ans = self.AI_tool.upload_file(self.create_prompt(is_all_history), self.upload_file_path)
            self.upload_file_path = ""
        else:
            ans = self.AI_tool.get_answer(self.create_prompt(is_all_history))

        if isinstance(ans, dict):
            ans_text = ans["content"]
        else:
            ans_text = str(ans)

        if self.chat_begin_time == None:
            self.chat_begin_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        reply = parse_model_reply_text(ans_text, self.voice_refs_lt, self.expression_lt)
        print(reply)
        if reply["action"] == "history_require":
            if is_all_history == False:
                print("进入")
                return self.chat_solution(True)
            else:
                self.loading_gif.hide()
                print("错误，连续请求历史")
                return False

        ans_message = {
            "role": "assistant",
            "content": reply["content"],
            "action": reply["action"],

        }
        if reply["action"] == "reply":
            ans_message["mood"] = reply["mood"]
            ans_message["expression"] = reply["expression"]

        self.all_dialogue_history.append(ans_message)

        if reply["action"] == "wait":
            print("等待")
            self.loading_gif.hide()
            return False
        if reply["special"] == "save":
            self.save_log_async(False)
        if reply["special"] == "conversation-over":
            self.save_log_async(True)
        if reply["heart_beat_seconds"] == "":
            reply["heart_beat_seconds"] = 30
        return (reply["jp"], reply["cn"], reply["mood"], reply["expression"], int(reply["heart_beat_seconds"]))

    def set_user_input_prompt(self, text):
        if text != "heart_beat":
            self.is_user_turn = False
        self.all_dialogue_history.append(
            {"role": "user", "content": datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " | " + text})

    def get_user_input(self):
        if self.upload_file_path != "":
            return True
        if self.is_voice_flag:
            is_empty = True
            while not self.voice_msg_queue.empty():
                is_empty = False
                self.set_user_input_prompt(self.voice_msg_queue.get())
            if is_empty:
                return False
        else:
            text = self.latest_input
            self.latest_input = ""
            if text == "":
                if self.in_initiative and self.is_user_turn and time.time() - self.last_chat_time > self.heartbeat_time:
                    text = "heart_beat"
                    self.last_chat_time = time.time()
                    print("心跳")
                else:
                    return False
            self.set_user_input_prompt(text)
        return True

    def get_model_ans(self):
        self.loading_gif.show()

        ans = self.chat_solution()
        if ans:
            ja_text = ans[0]
            cn_text = ans[1]
            mood = ans[2]
            expression = ans[3]
            heart_beat_seconds = ans[4]
            addr = "voice.wav"
            get_radio(ja_text, addr, mood + "." + self.ref_voice_format, self.AI_tool, pet_id)
            return [pygame.mixer.Sound(addr), cn_text, mood, expression, heart_beat_seconds]
        return False

    def chat_perform(self, arr):
        self.last_chat_time = time.time()
        self.is_user_turn = True
        self.now_text = arr[1]
        self.mood_status = arr[2]
        self.now_expression = arr[3]
        self.heartbeat_time = max(self.min_heart_beat_time, arr[4])
        length = self.play_sound(arr[0])
        if length > 3:
            self.change_expression_motion()
            Thread(target=self.wait_for_reset, args=(length,)).start()

        self.show_expression(self.now_expression)

    def chat(self):
        self.last_chat_time = time.time()
        self.min_heart_beat_time = 5
        self.heartbeat_time = self.min_heart_beat_time
        self.next_ans = None
        while True:
            time.sleep(1 / 60)
            if self.is_on_chat:

                if self.next_ans and self.is_speaking == False:
                    self.chat_perform(self.next_ans)
                    self.next_ans = None

                try:
                    if self.get_user_input() == False:
                        continue

                    if self.next_ans == None:
                        self.next_ans = self.get_model_ans()
                        if self.next_ans == False:
                            self.next_ans = None

                except Exception as e:
                    print(f"聊天流程异常: {e}")

                self.loading_gif.hide()

    def load(self):
        model_dir = basepath + "model_data/"
        model_path = None

        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith(".model3.json"):
                    model_path = model_dir + f
                    break
        if not model_path:
            print("错误: 未在 model_data/ 下找到 .model3.json 文件")
            sys.exit(1)
        print(f"加载模型: {model_path}")
        self.model = live2d.LAppModel()
        self.model.LoadModelJson(model_path)
        self.model.Resize(WIDTH, HEIGHT)
        self.load_model_expression_data()
        self.load_voice_categories()
        # self.model.SetScale()
        for i in range(self.model.GetParameterCount()):
            param = self.model.GetParameter(i)
            self.param_dt[param.id] = i
        self.AI_tool = AI_tools.SocketAI(
            username=get_arg("--username"),
            base_url=get_arg("--server-url"),
            pet_id=get_arg("--pet-id"),
        )

    def like_music(self):
        self.in_daily = False
        self.recommend_button.setChecked(False)
        self.in_like = not self.in_like
        self.on_change_music = True

    def daily_recommend(self):
        self.in_like = False
        self.like_button.setChecked(False)
        self.in_daily = not self.in_daily
        self.on_change_music = True

    #
    def wait_for_reset(self, t):
        time.sleep(t)
        self.in_shake_body = False

    def shake_body(self):
        shake_param_label = self.effect_label_dict["shakeParam"]
        factor_abs = 0.06  # 0.02

        factor = factor_abs
        while True:
            param_index = self.param_dt[shake_param_label]
            param = self.model.GetParameter(param_index)

            if self.in_shake_body:
                self.model.AddParameterValue(shake_param_label, factor * (self.shake_range - abs(param.value) + 0.1))
                if param.value >= self.shake_range:
                    factor = -factor_abs
                if param.value <= -self.shake_range:
                    factor = factor_abs
                time.sleep(1 / 60)
            else:
                if abs(param.value) > 0.1:

                    factor = -factor_abs if param.value > 0 else factor_abs
                    self.model.AddParameterValue(shake_param_label,
                                                 factor * abs(param.value))
                    time.sleep(1 / 60)
                else:
                    self.model.SetParameterValue(shake_param_label, 0)
                    time.sleep(0.1)

    def random_cloth(self):
        while True:
            num = random.choice(self.cloth_lt)
            if num != self.last_cloth:
                self.change_cloth(num)
                return

    def keyPressEvent(self, event):
        key_code = event.key()
        if key_code == 16777220:
            if self.is_voice_flag == False and self.is_on_chat:
                try:
                    self.latest_input = easygui.enterbox("有什么想对我说的呢？")
                    if self.latest_input == None:
                        self.latest_input = ""
                except Exception as e:
                    print(e)

    def mouseReleaseEvent(self, event):
        self.in_moving = False

    def mousePressEvent(self, event):
        point = event.pos()
        x = point.x()
        y = point.y()
        if x > 0.4 * WIDTH * SCALE_FACTOR and x < 0.6 * WIDTH * SCALE_FACTOR and y > 0.2 * HEIGHT * SCALE_FACTOR and y < 0.4 * HEIGHT * SCALE_FACTOR:
            self.get_random_sound()

    def mouseMoveEvent(self, event):
        if self.in_moving:
            self.move(self.window_origin_pos + QCursor.pos() - self.mouse_origin_pos)

    def wheelEvent(self, event):
        global SCALE_FACTOR
        delta = event.angleDelta().y()

        if delta > 0:
            SCALE_FACTOR += 0.1
            SCALE_FACTOR = min(SCALE_FACTOR, 3)
        else:
            SCALE_FACTOR -= 0.1
            SCALE_FACTOR = max(SCALE_FACTOR, 1)
        self.on_scale_slider()

    def paintGL(self):

        live2d.clearBuffer(0, 0, 0, 0)
        self.model.Update()

        self.model.Draw()
        self.draw_text()


class CustomMenu(QMenu):
    def mouseReleaseEvent(self, event):
        """重写鼠标释放事件，保持菜单不关闭"""

        action = self.actionAt(event.pos())

        if action:
            if action.isCheckable():
                now_state = not action.isChecked()
                action.setChecked(now_state)
                action.triggered.emit(now_state)
                action.toggled.emit(now_state)
                event.accept()

            else:
                action.triggered.emit(True)
            return
        super().mouseReleaseEvent(event)


class CustomMenu2(QMenu):
    def mouseReleaseEvent(self, event):
        """重写鼠标释放事件，保持菜单不关闭"""

        action = self.actionAt(event.pos())

        if action:
            action.setChecked(True)
            action.triggered.emit(True)
            action.toggled.emit(True)
            event.accept()


class AudioData:
    def __init__(self, pygame_sound, wave_data, channel):
        self.pygame_sound = pygame_sound
        self.wave_lt = wave_data[0]
        self.win_ms = wave_data[1]
        self.channel = channel
        self.start_pos = pygame.time.get_ticks()

    def get_realtime_react(self):
        return self.wave_lt[min((pygame.time.get_ticks() - self.start_pos) // self.win_ms, len(self.wave_lt) - 1)]

    def check_over(self):
        return not self.channel.get_busy()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    atri = GLWidget()
    atri.show()
    sys.exit(app.exec())
