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
    QLinearGradient, QImage, QCursor, QPalette, QMovie,QActionGroup
from PyQt6.QtCore import QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtWidgets import QApplication, QPushButton

import live2d.v3 as live2d
from OpenGL.GLUT import *
from threading import Thread

from harness_bridge import PetHarnessBridge
from voicerecver import *
from voice_producer import *
import audio_prepare
import pygame
import ncm
import AI_tools


def get_arg(name,defalut=""):
    """返回桌宠资源根路径：优先使用 --pet-dir 参数，否则用脚本所在目录"""
    for i, arg in enumerate(sys.argv):
        if arg == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return defalut

basepath = get_arg("--pet-dir")
pet_id=get_arg("--pet-id")
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

# Realtime provider used by this client. Use "minicpm" for the cloud-hosted
# MiniCPM service, or change it to "qwen" when connecting to the local Qwen
# realtime service.
REALTIME_PROVIDER = "qwen"

os.environ['SDL_VIDEO_WINDOW_POS'] = "{},{}".format(POSX, POSY)

pygame.init()
live2d.init()


def make_chat_message(role, text):
    return {"role": role, "parts": [{"text": text}]}




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
    heart_beat_seconds=str(payload.get("heart_beat_seconds") or "").strip()
    special=str(payload.get("special") or "").strip()
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
        "heart_beat_seconds":heart_beat_seconds,
        "special":special
    }


def history_messages_to_jsonl(messages, started_at, ended_at, marker, tags=None):
    session_tags = []
    if tags is not None:
        if not isinstance(tags, (list, tuple)):
            raise ValueError("tags must be a list of topic/memory-type pairs")
        for tag in tags:
            if not isinstance(tag, (list, tuple)) or len(tag) != 2:
                raise ValueError("each tag must contain a topic and a memory type")
            pair = list(tag)
            if pair not in session_tags:
                session_tags.append(pair)

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
        for key in ("action", "mood", "expression", "cn"):
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
        "tags": session_tags,
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


    def parse_dialogue_history_text(self,text):
        messages = []
        index=0
        self.tags_index_dict[0]=[]
        if not text or not text.strip():
            return messages

        for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            line = raw_line.strip()
            if not line:
                continue

            event = json.loads(line)
            next_turn=False
            if not isinstance(event, dict):
                return messages

            event_type = event.get("type") or ""
            started_at = event.get("started_at")
            ended_at = event.get("ended_at")
            marker = event.get("marker") or event.get("source") or ""
            tags = event.get("tags") or []
            item=None

            if event_type.endswith("session_start") and started_at:
                item={
                    "role": "developer",
                    "content": "【历史会话信息】一段历史对话开始于 {}，记录标记：{}。".format(started_at, marker),
                }
            if event_type in ("dialogue_message", "summary_message"):
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
                        item={"role": role, "content": content}


            if event_type.endswith("session_end") and ended_at:


                item={
                    "role": "developer",
                    "content": "【历史会话信息】上一段历史对话结束于 {}。".format(ended_at),
                }
                self.tags_dict[index]=tags
                next_turn=True

            self.tags_index_dict[index].append(item)
            messages.append(item)
            if next_turn:
                index += 1
                self.tags_index_dict[index] = []


        return messages

    def tags_match(self,target):

        result_lt=[]
        ans_lt=[]
        for i in self.tags_dict.keys():
            mark=0
            flag=False
            value=self.tags_dict[i]
            for u in target:
                for v in value:
                    if u==v:
                        mark+=1
                        break
            for j in range(0,len(result_lt)):
                if result_lt[j][1]<mark:
                    result_lt.insert(j,[i,mark])
                    flag=True
                    break
            if flag==False:
                result_lt.append([i,mark])
        if len(result_lt)==0:
            return [{"role": "developer", "content": "你请求了往期历史，但什么都没有获得。"}]
        else:
            for i in range(0,min(5,len(result_lt))):
                ans_lt.append(result_lt[i][0])
            return ans_lt
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

        self.upload_file_path=""
        self.shake_range = 0
        self.shake_reset_time=0
        # self.mouth_open_dict = {"shy": 0.3, "happy": 0.8, "得意": 0.8, "sad": 0.4, "normal": 0.6, "confuse": 0.6,
        #                         "气急败坏": 0.8, "赌气": 0.8, "委屈想哭": 0.3, "彻底坏掉": 0.3}  # 不同心情嘴张多大
        # self.body_shake_range_dict={"shy":2,"happy":8,"得意":8}
        self.hair_flying = True
        self.in_shake_body = False
        self.is_on_chat = False
        self.is_voice_flag = False
        self.in_initiative=False
        self.mood_active = False
        self.is_speaking = False
        self.is_user_turn=False
        self.is_saving_log=False
        self.is_evil_mode=False
        self.mood_status = ""
        self.now_expression=""
        self.now_text = ""
        self.all_text=""
        self.latest_input = ""
        self.all_prefab_audio = []
        self.all_music = {}
        self.now_music = []
        self.chunk_queue=queue.Queue()
        self.voice_queue=queue.Queue()
        self.realtime_chunk_queue=queue.Queue()
        self._realtime_current_turn_id = None
        self._realtime_user_history = {}
        self._realtime_assistant_history = {}

        self.dirty_turn_id=[]
        self._realtime_assistant_turns = set()
        self._realtime_history_lock = threading.Lock()
        self.sounds_queue=queue.Queue()
        self.music_on = True
        self.on_change_music = False
        self.is_ncm_loaded = False
        self.is_screen_monitor=False
        self.in_like = False
        self.in_daily = True
        self.on_auto_play = True
        self.now_playing_sound = None
        self.in_moving = False
        self.last_cloth = None

        self.cloth_lt = [18, 9, 19, 20]

        self.init_music()
        self.config_llm()
        self.set_harness()
        self.set_menu()
        self.update_ui()
        self.music_play_over_check()
        self.awake_threads()
        self.motions_lt = self.model.GetMotionGroups()
        self.set_model_deepseek()

        self.is_loading_end = True
        sys.excepthook=self.main_exception_handler
        threading.excepthook=self.threads_exception_handler
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
        print("实时语音对话:", "开启" if self.is_voice_flag else "关闭")
    def set_text(self, text):
        self.now_text = text

    def change_expression_motion(self):
        value=self.expression_effect_dict[self.now_expression]["shake"]
        if value!=0:
            self.in_shake_body=True
            self.shake_range=value

    def load_model_custom_data(self):
        with open(basepath+"config/custom.json","r",encoding="utf-8") as f:
            self.user_custom_dict=json.load(f)
    def load_model_expression_data(self):
        self.expression_effect_dict={}
        with open(basepath+"config/expression_effect.json","r",encoding="utf-8") as f:
            data=json.load(f)
            self.expression_effect_dict:dict=data["expressions"]
            self.effect_label_dict=data["paramID"]
        for i in self.expression_effect_dict.keys():
            self.expression_lt.append(i)
    def load_voice_categories(self):
        lt=os.listdir(basepath+"config/voice_refs/")
        self.voice_refs_lt=[]
        for i in lt:
            self.voice_refs_lt.append(i.split(".")[0])
        if len(self.voice_refs_lt)==0:
            print("无参考音频")
            sys.exit()
        self.ref_voice_format = lt[0].split(".")[1]

    def add_draw_text(self,text):

        self.all_text+=text
    def clear_draw_text(self):
        self.all_text=""
        self.now_text=""
    def draw_text(self):
        painter = QPainter(self)
        # painter.begin(self)
        font = QFont('Arial', int(16 * SCALE_FACTOR))

        if self.now_text != "":
            text_rect = QRect(0, int(50 * SCALE_FACTOR), int(150 * SCALE_FACTOR), int(550 * SCALE_FACTOR))#350
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
    def update_text(self):
        while True:
            if self.now_text!=self.all_text:
                self.now_text+=self.all_text[len(self.now_text)]
            time.sleep(0.1)
    def save_log(self,is_exit=True):

        if not self.chat_begin_time or self.is_saving_log:
            return

        save_start = self.last_saved_dialogue_history_end
        save_end = len(self.all_dialogue_history)
        new_messages = self.all_dialogue_history[save_start:save_end]
        if not new_messages:
            return

        print("准备保存历史")
        self.is_saving_log=True
        try:
            ended_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            txt = history_messages_to_jsonl(
                new_messages,
                self.chat_begin_time,
                ended_at,
                "唤醒model--PC端",
            )
            with open(basepath+"config/dialogue_history.txt" ,"a",encoding="utf-8") as f:
                f.write(txt)
            result = self.AI_tool.save_dialogue_history(txt)
            if result != "success":
                print("保存聊天历史失败:", result)
                return
            self.last_saved_dialogue_history_end = save_end
            self.current_session_history_start = save_end
            self.chat_begin_time = None
        finally:
            self.is_saving_log=False
            # with open(basepath + "config/chat_history.txt", "a", encoding="utf-8") as f:
            #     f.write(txt)
        if is_exit:
            sys.exit()

    def load_summary_history(self):
        ans = self.AI_tool.get_summary_history()
        self.summary_prompt=[]

        if ans[0] == "success":
            self.summary_prompt.append(
                {"role": "developer", "content": "接下来是我们对话历史的精炼版，你可以从中参考获得更详细的人设定位。"})
            self.summary_prompt.extend(self.parse_dialogue_history_text(ans[1]))

        else:
            print("bakaYou!")
            sys.exit()
    def load_dialogue_history(self):
        ans = self.AI_tool.get_dialogue_history()
        self.all_dialogue_history=[]

        if ans[0] == "success":
            self.all_dialogue_history.extend(self.parse_dialogue_history_text(ans[1]))
            self.current_session_history_start = len(self.all_dialogue_history)
            self.last_saved_dialogue_history_end = self.current_session_history_start
            print("加载聊天历史，共{}条".format(len(self.all_dialogue_history)))
        else:
            print("baka!")
            sys.exit()
    def get_preset_prompt(self):
        tags = self.AI_tool.get_dialogue_tags()
        self.tags_lt = []
        if tags[0] == "success":
            self.tags_lt = tags[1]
        self.tags_index_dict = {}
        self.tags_dict = {}
        self.tags_prompt = []
        with open(basepath + "config/rules.txt", "r", encoding="utf-8") as f:
            rules = f.read()

            self.rule_prompt=[{"role":"developer","content":"下面为规则说明/重申"}]
            self.rule_prompt.append({"role":"developer","content":rules})
            self.rule_prompt.append(
                {"role": "developer", "content": "这是可选心情列表————" + str(self.voice_refs_lt)})
            self.rule_prompt.append(
                {"role": "developer", "content": "这是可选表现列表————" + str(self.expression_lt)})

        self.preset_prompt = []

        self.preset_prompt+=self.rule_prompt
        with open(basepath + "config/personal_setting.txt", "r", encoding="utf-8") as f:
            self.preset_prompt.append({"role":"developer","content":f.read()})




    def create_prompt(self):

        # if is_all_history:
        #     return self.preset_prompt+self.summary_prompt+self.all_dialogue_history+[{"role":"user","content":"[系统消息]:你已经请求并且获得了全部对话历史"}]+self.rule_prompt
        #把前50条内容也一起发过去,当作近期记忆

        extra_info=[{"role": "developer", "content": "接下来是我们近期的聊天记录，有助于你了解最近发生了什么。"}]
        current_memory_num=50
        start_index=max(0,self.current_session_history_start-current_memory_num)

        return self.preset_prompt+self.summary_prompt+extra_info+self.all_dialogue_history[start_index:]+self.rule_prompt

        # return self.all_dialogue_history
    def set_harness(self):
        self.harness_bridge = PetHarnessBridge(
            search_callback=self.tags_match,
            tags_callback=lambda: self.tags_lt,
            username=get_arg("--username"),
            pet_id=pet_id,
        ).start()

        QApplication.instance().aboutToQuit.connect(
            self.harness_bridge.close
        )
    def config_llm(self):

        self.voice_msg_queue=queue.Queue()
        self.chat_begin_time = None
        self.get_preset_prompt()
        self.load_summary_history()
        self.load_dialogue_history()


    def play_sound(self, sound, analyze_data):

        self.now_playing_sound = AudioData(sound, analyze_data, sound.play())
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

    def add_menu(self,menu, name, solution, checkable=False, toggle=False):
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
            self.play_sound(self.all_prefab_audio[self.prefab_audio_index][1],audio_prepare.analyze_wav(self.all_prefab_audio[self.prefab_audio_index][2])
                            )

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
                file_path=easygui.fileopenbox(msg="选择文件")
                if file_path!="" and file_path:
                    self.upload_file_path=file_path
        except:
            print("上传文件失败！")
    def allow_move_window(self):
        self.in_moving = True
        self.window_origin_pos = self.pos()
        self.mouse_origin_pos = QCursor.pos()

    def save_log_async(self,flag1):
        Thread(target=self.save_log,args=(flag1,)).start()

    def toggle_initiative(self):
        self.last_chat_time = time.time()
        self.in_initiative=not self.in_initiative
        self.is_user_turn=False
    def evil(self):
        self.is_evil_mode=not self.is_evil_mode
        if self.is_evil_mode:
            self.AI_tool.model_name="gpt"
        else:
            self.AI_tool.model_name="deepseek"

    def set_model_deepseek(self):
        self.AI_tool.api_url="https://api.ai-pixel.online/v1"
        self.AI_tool.api_key=os.getenv("DEEPSEEK_API_KEY", "").strip()
        self.AI_tool.model_name="deepseek"
    def set_model_gemini(self):
        self.AI_tool.api_url="https://api.ai-pixel.online/v1"
        self.AI_tool.api_key=os.getenv("GEMINI_API_KEY", "").strip()
        self.AI_tool.model_name="gemini"
    def set_model_gpt(self):
        self.AI_tool.api_url="https://api.ai-pixel.online/v1"
        self.AI_tool.api_key=os.getenv("OPENAI_API_KEY", "").strip()
        self.AI_tool.model_name="gpt"
    def set_model_devin(self):
        self.AI_tool.api_url="https://api.ai-pixel.online/v1"
        self.AI_tool.api_key=os.getenv("DEVIN_API_KEY", "").strip()
        self.AI_tool.model_name="devin"

    def set_menu(self):
        self.menu = CustomMenu()
        self.add_menu(self.menu,"开启对话", self.set_chat_mode, True)
        self.add_menu(self.menu,"主动模式",self.toggle_initiative,True)
        self.add_menu(self.menu,"语音输入模式", self.set_voice_flag, True)
        self.add_menu(self.menu,"上传文件",self.send_file,False)
        self.add_menu(self.menu,"开启音乐", self.check_music, True, True)
        if self.is_ncm_loaded:
            self.like_button = self.add_menu(self.menu,"红心漫游", self.like_music, True)
            self.recommend_button = self.add_menu(self.menu,"日推抽奖", self.daily_recommend, True, True)
        self.add_menu(self.menu,"自动播放下一曲", self.auto_play, True, True)
        self.add_menu(self.menu,"随机切换音乐", self.change_music_clicked, False)
        self.add_menu(self.menu,"随机变换", self.random_expression, False)
        self.add_menu(self.menu,"保存对话历史",self.save_log_async,False)

        model_menu = CustomMenu2("模型选择",self.menu)
        self.menu.addMenu(model_menu)

        group=QActionGroup(self)
        group.addAction(self.add_menu(model_menu, "gpt", self.set_model_gpt,True,False))
        group.addAction(self.add_menu(model_menu, "gemini", self.set_model_gemini,True,False))
        group.addAction(self.add_menu(model_menu, "deepseek", self.set_model_deepseek,True,True))
        group.addAction(self.add_menu(model_menu, "devin", self.set_model_devin, True, False))
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



    def voice_recv(self, msg_queue=None, sample_rate=16000):
        """持续采集 16 kHz PCM16，并把音频帧推送到本地 MiniCPM 会话。"""
        frame_count = 320  # 16000 Hz 下的 20 ms 音频
        realtime_started = False
        previous_model_config = None
        speech_end_hold_seconds = 0.6
        calibration_seconds = 0.8
        initiative_changed = False
        in_speech = False
        last_voice_at = None
        idle_since = None
        calibration_until = None
        noise_samples = []
        energy_threshold_db = -45.0
        response_was_busy = False

        while True:
            if not self.is_voice_flag:
                if realtime_started:
                    try:
                        self.AI_tool.stop_realtime()
                    except Exception as e:
                        print("关闭实时语音会话失败:", e)
                    realtime_started = False
                    if previous_model_config is not None:
                        self.AI_tool.model_name, self.AI_tool.api_url, self.AI_tool.api_key = previous_model_config
                        previous_model_config = None
                in_speech = False
                last_voice_at = None
                idle_since = None
                calibration_until = None
                noise_samples = []
                response_was_busy = False
                time.sleep(1 / 60)
                continue

            if not realtime_started:
                try:
                    # 普通模型配置与实时模型配置分开；Qwen 使用独立的实时接口凭据。
                    previous_model_config = (
                        self.AI_tool.model_name,
                        self.AI_tool.api_url,
                        self.AI_tool.api_key,
                    )
                    self._realtime_current_turn_id = None
                    with self._realtime_history_lock:
                        self._realtime_user_history.clear()
                        self._realtime_assistant_history.clear()
                        self.dirty_turn_id.clear()
                        self._realtime_assistant_turns.clear()
                    self.set_model_gpt()
                    realtime_provider = str(REALTIME_PROVIDER).strip().lower()
                    if realtime_provider == "qwen":
                        qwen_url = self.AI_tool.realtime_url
                        qwen_api_key = self.AI_tool.realtime_api_key
                        if not qwen_url or not qwen_api_key:
                            raise RuntimeError(
                                "Qwen Realtime 未配置 QWEN_REALTIME_URL 或 QWEN_REALTIME_API_KEY"
                            )
                        response = self.AI_tool.start_realtime(
                            self.create_prompt(),
                            model=os.getenv(
                                "QWEN_REALTIME_MODEL",
                                "qwen-audio-3.0-realtime-plus",
                            ),
                            transcription_model=None,
                            provider="qwen",
                            realtime_url=qwen_url,
                            realtime_api_key=qwen_api_key,
                        )
                    elif realtime_provider in (
                        "minicpm",
                        "minicpm-o",
                        "minicpm_local",
                        "local",
                    ):
                        realtime_provider = "minicpm"
                        response = self.AI_tool.start_realtime(
                            self.create_prompt(),
                            model="MiniCPM-o-4_5-int4",
                            transcription_model=None,
                            provider="minicpm",
                        )
                    elif realtime_provider in (
                            "gemini",
                            "gemini-live",
                            "gemini3.8live",
                    ):
                        response = self.AI_tool.start_realtime(
                            self.create_prompt(),
                            model=os.getenv(
                                "GEMINI_REALTIME_MODEL",
                                "gemini-3.8-live",
                            ),
                            transcription_model=None,
                            provider="gemini-live",
                            realtime_url="",
                            realtime_api_key="",
                        )
                    else:
                        raise RuntimeError(
                            "不支持的实时模型提供方: {}".format(realtime_provider)
                        )
                    if not isinstance(response, dict) or not response.get("ok"):
                        raise RuntimeError(
                            (response or {}).get("error", "服务端未建立实时语音会话")
                            if isinstance(response, dict)
                            else response
                        )
                    realtime_started = True
                    now = time.monotonic()
                    calibration_until = now + calibration_seconds
                    noise_samples = []
                    energy_threshold_db = -45.0
                    in_speech = False
                    last_voice_at = None
                    idle_since = now
                    initiative_changed = bool(self.in_initiative)
                    response_was_busy = self.AI_tool.realtime_response_busy()
                    print(realtime_provider.upper() + " Realtime 已连接")
                except Exception as e:
                    print("启动实时语音会话失败:", e)
                    if previous_model_config is not None:
                        self.AI_tool.model_name, self.AI_tool.api_url, self.AI_tool.api_key = previous_model_config
                        previous_model_config = None
                    self.is_voice_flag = False
                    time.sleep(0.5)
                    continue

            try:
                with sd.InputStream(
                    samplerate=sample_rate,
                    channels=1,
                    dtype="int16",
                    blocksize=frame_count,
                ) as stream:
                    while self.is_voice_flag and realtime_started:
                        frame, overflowed = stream.read(frame_count)
                        if overflowed:
                            print("麦克风音频溢出，丢弃当前帧")
                        if not self.AI_tool.send_realtime_audio(
                            frame.tobytes(), sample_rate
                        ):
                            print("实时语音会话已断开")
                            realtime_started = False
                            self.is_voice_flag = False
                        else:
                            now = time.monotonic()
                            samples = np.asarray(frame, dtype=np.float32).reshape(-1)
                            rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
                            dbfs = 20.0 * np.log10(max(rms, 1.0) / 32768.0)

                            if calibration_until is not None and now < calibration_until:
                                noise_samples.append(dbfs)
                            elif calibration_until is not None:
                                if noise_samples:
                                    noise_floor_db = float(np.percentile(noise_samples, 60))
                                    energy_threshold_db = max(
                                        -50.0,
                                        min(-25.0, noise_floor_db + 10.0),
                                    )
                                calibration_until = None

                            start_threshold_db = energy_threshold_db + 3.0
                            if dbfs >= start_threshold_db:
                                in_speech = True
                                last_voice_at = now
                                idle_since = None
                            elif in_speech:
                                if last_voice_at is None:
                                    last_voice_at = now
                                elif now - last_voice_at >= speech_end_hold_seconds:
                                    in_speech = False
                                    last_voice_at = None
                                    idle_since = now

                            current_initiative = bool(self.in_initiative)
                            if current_initiative != initiative_changed:
                                initiative_changed = current_initiative
                                if current_initiative:
                                    idle_since = now

                            response_busy = self.AI_tool.realtime_response_busy()
                            if response_busy:
                                idle_since = None
                            elif response_was_busy:
                                idle_since = now
                            response_was_busy = response_busy

                            tts_busy = bool(
                                self.is_speaking
                                or not self.voice_queue.empty()
                                or not self.sounds_queue.empty()
                            )
                            if tts_busy:
                                idle_since = None
                            elif (
                                idle_since is None
                                and not in_speech
                                and not response_busy
                            ):
                                idle_since = now

                            heartbeat_seconds = self.AI_tool.get_realtime_heartbeat_seconds()
                            if heartbeat_seconds is None:
                                heartbeat_seconds = getattr(self, "heartbeat_time", 30)
                            try:
                                heartbeat_seconds = max(5.0, float(heartbeat_seconds))
                            except (TypeError, ValueError):
                                heartbeat_seconds = 30.0

                            if (
                                self.in_initiative
                                and not in_speech
                                and idle_since is not None
                                and now - idle_since >= heartbeat_seconds
                                and not response_busy
                                and not tts_busy
                            ):
                                try:
                                    proactive_response = self.AI_tool.request_realtime_response()
                                except Exception as proactive_error:
                                    print("实时主动回复请求失败:", proactive_error)
                                    proactive_response = None
                                if isinstance(proactive_response, dict) and proactive_response.get("ok"):
                                    idle_since = None
                                    self.is_user_turn = False
                                else:
                                    # Avoid retrying on every 20 ms frame when the
                                    # server rejects the request or is unavailable.
                                    idle_since = now

                            self.is_user_turn=False

            except Exception as e:
                print("实时录音失败:", e)
                self.is_voice_flag = False
            finally:
                if not self.is_voice_flag or not realtime_started:
                    try:
                        self.AI_tool.stop_realtime()
                    except Exception as e:
                        print("清理实时语音会话失败:", e)
                    realtime_started = False
                    if previous_model_config is not None:
                        self.AI_tool.model_name, self.AI_tool.api_url, self.AI_tool.api_key = previous_model_config
                        previous_model_config = None
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
        Thread(target=self.voice_recv,args=(self.voice_msg_queue,),daemon=True).start()
        Thread(target=self.chat, daemon=True).start()
        Thread(target=self.speaking, daemon=True).start()
        Thread(target=self.music_ctrl, daemon=True).start()
        Thread(target=self.shake_body, daemon=True).start()
        Thread(target=self.voice_chunk_get,daemon=True).start()
        Thread(target=self.realtime_chat_consume, daemon=True).start()
        Thread(target=self.update_text, daemon=True).start()
        Thread(target=self.sounds_play_thread, daemon=True).start()
    def sequence_add(self,chunk):
        self.chunk_queue.put(chunk)

    def realtime_sequence_add(self, chunk):
        self.realtime_chunk_queue.put(chunk)

    def realtime_event_add(self, event):
        if not isinstance(event, dict):
            return

        event_type = event.get("type")
        turn_id = event.get("turn_id")
        if turn_id:
            self._realtime_current_turn_id = turn_id

        if event_type == "speech_started":
            self.is_user_turn = False
            return

        if event_type == "user_transcript":
            text = str(event.get("text") or "").strip()
            if text and turn_id:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if self.chat_begin_time is None:
                    self.chat_begin_time = timestamp
                message = {
                    "role": "user",
                    "content": timestamp + " | " + text,
                    "turn_id": turn_id,
                }
                with self._realtime_history_lock:
                    self._realtime_user_history[turn_id] = message
                    if turn_id not in self.dirty_turn_id and turn_id in self._realtime_assistant_history.keys():
                        self.dirty_turn_id.append(turn_id)
                        self.all_dialogue_history.append(message)
                        self.all_dialogue_history.append(self._realtime_assistant_history[turn_id])

            return

        if event_type == "turn_ready" and event.get("user_text"):
            # 转写事件晚到时，使用 turn_ready 中的文本修正历史顺序。
            self.realtime_event_add({
                "type": "user_transcript",
                "turn_id": turn_id,
                "text": event.get("user_text"),
            })
            return

        if event_type == "assistant_started":
            self.loading_gif.show()
        elif event_type == "turn_cancelled":
            self.realtime_chunk_queue.put([
                "cancel",
                {"turn_id": turn_id},
            ])
            self.last_chat_time = time.time()
        elif event_type == "turn_completed":
            self.last_chat_time = time.time()
        elif event_type == "error":
            print("Realtime 事件错误:", event.get("message", event))

    def realtime_chat_consume(self):
        """把 Realtime 的整回合输出接入现有字幕、TTS 和表达系统。"""
        mood = "normal"
        expression = "normal"
        turn_id = None
        turn_action = "reply"

        while True:
            flag, reply = self.realtime_chunk_queue.get()
            reply = reply if isinstance(reply, dict) else {}
            reply_turn_id = reply.get("turn_id")
            if reply_turn_id:
                turn_id = reply_turn_id

            if flag == "params":
                action = str(reply.get("action") or "reply")
                turn_action = action
                if action == "reply":
                    mood = str(reply.get("mood") or "normal")
                    expression = str(reply.get("expression") or "normal")
                    special = str(reply.get("special") or "")
                    if special == "save":
                        self.save_log_async(False)
                    elif special == "conversation-over":
                        self.save_log_async(False)
                elif action == "wait":
                    self.all_dialogue_history.append({
                        "role": "assistant",
                        "action": "wait",
                        "content": "wait",
                        "turn_id": turn_id,
                    })
                continue

            if flag == "segment":
                jp_text = str(reply.get("jp") or "").strip()
                cn_text = str(reply.get("cn") or jp_text).strip()
                if not jp_text and not cn_text:
                    continue
                self.voice_queue.put({
                    "jp_text": jp_text or cn_text,
                    "cn_text": cn_text or jp_text,
                    "mood": mood + "." + self.ref_voice_format,
                    "model_name": pet_id,
                    "expression": expression,
                    "turn_id": turn_id,
                })
                continue

            if flag == "cancel":
                self.voice_queue.put({
                    "_realtime_cancel": True,
                    "turn_id": turn_id,
                })
                mood = "normal"
                expression = "normal"
                turn_id = None
                turn_action = "reply"
                continue

            if flag == "end":
                if turn_action == "reply":
                    self.voice_queue.put({
                        "_realtime_end": True,
                        "turn_id": turn_id,
                    })
                self.loading_gif.hide()
                mood = "normal"
                expression = "normal"
                turn_id = None
                turn_action = "reply"

    def sounds_play_thread(self):
        clear_flag = False
        while True:
            if self.is_speaking == False:
                if self.sounds_queue.empty()==False:
                    item = self.sounds_queue.get()
                    if clear_flag:
                        clear_flag=False
                        self.clear_draw_text()
                    if item=="end":
                        clear_flag=True
                        continue
                    # time.sleep(0.3)
                    self.loading_gif.hide()
                    self.show_expression(item[0])
                    self.change_expression_motion()
                    self.add_draw_text(item[2])
                    self.play_sound(item[1],item[3])
                    self.last_chat_time = time.time()
                    self.is_user_turn = True
                else:
                    self.in_shake_body=False
            time.sleep(1/60)


    def voice_chunk_get(self):

        all_cn_text=""
        all_jp_text=""
        realtime_turn_id = None
        realtime_end_turn_id = None


        while True:
            while self.voice_queue.empty()==False:
                data=self.voice_queue.get()
                if isinstance(data, dict) and data.get("_realtime_cancel"):
                    cancel_turn_id = data.get("turn_id")
                    if cancel_turn_id == realtime_turn_id or realtime_turn_id is None:
                        all_cn_text=""
                        all_jp_text=""
                        realtime_turn_id = None
                        realtime_end_turn_id = None
                        while not self.sounds_queue.empty():
                            self.sounds_queue.get()
                        self.sounds_queue.put("end")
                    continue
                if isinstance(data, dict) and data.get("_realtime_end"):
                    realtime_end_turn_id = data.get("turn_id")
                    data = "end"
                if data=="end":
                    message={
                        "role":"assistant",
                        "content":all_jp_text,
                        "cn":all_cn_text,
                        "action":"reply"
                    }
                    if realtime_end_turn_id:
                        with self._realtime_history_lock:
                            message["turn_id"] = realtime_end_turn_id
                            self._realtime_assistant_history[realtime_end_turn_id]=message
                            if realtime_end_turn_id not in self.dirty_turn_id and realtime_end_turn_id in self._realtime_user_history.keys():
                                self.dirty_turn_id.append(realtime_end_turn_id)
                                self.all_dialogue_history.append(self._realtime_user_history[realtime_end_turn_id])
                                self.all_dialogue_history.append(message)
                    else:
                        self.all_dialogue_history.append(message)
                    all_cn_text=""
                    all_jp_text=""
                    realtime_turn_id = None
                    realtime_end_turn_id = None
                    self.sounds_queue.put("end")
                    continue
                if isinstance(data, dict) and data.get("turn_id"):
                    realtime_turn_id = data["turn_id"]

                get_radio(data["jp_text"], "voice.wav", data["mood"], self.AI_tool, data["model_name"])

                all_jp_text+=data["jp_text"]
                all_cn_text+=data["cn_text"]
                self.sounds_queue.put([data["expression"],pygame.mixer.Sound("voice.wav"),data["cn_text"],audio_prepare.analyze_wav("voice.wav")])


            time.sleep(1/60)

    def speaking(self):
        mouth_param_label=self.effect_label_dict["mouthParam"]
        while True:
            if self.is_speaking:

                if self.now_playing_sound.check_over():


                    self.model.SetParameterValue(mouth_param_label, 0)

                    self.is_speaking = False
                else:
                    if self.now_expression=="":
                        self.model.SetParameterValue(mouth_param_label, 0.5
                            * self.now_playing_sound.get_realtime_react())
                    else:
                        self.model.SetParameterValue(mouth_param_label, self.expression_effect_dict[
                            self.now_expression]["mouth_open"] * self.now_playing_sound.get_realtime_react())
            else:
                time.sleep(1/60)

    def show_expression(self, expression):
        if expression!=self.now_expression:
            self.model.ResetExpression()
            self.model.SetExpression(expression)
            self.now_expression=expression

    def chat_solution(self):

        while self.chunk_queue.empty()==False:
            self.chunk_queue.get()
        mood=""
        expression=""
        

        if self.upload_file_path!="":
            self.AI_tool.upload_file(self.create_prompt(),self.upload_file_path)
            self.upload_file_path=""
        else:
            self.AI_tool.get_answer(self.create_prompt())

        while True:
            self.loading_gif.show()
            while self.chunk_queue.empty()==False:
                if self.chat_begin_time == None:
                    self.chat_begin_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                chunk=self.chunk_queue.get()
                print(chunk)
                flag=chunk[0]
                reply=chunk[1]
                if flag=="params":
                    action=reply["action"]
                    if action == "reply":
                        mood = reply["mood"]
                        expression = reply["expression"]
                        if reply["heart_beat_seconds"] == "":
                            reply["heart_beat_seconds"] = 30
                        else:
                            self.heartbeat_time = max(self.min_heart_beat_time, int(reply["heart_beat_seconds"]))
                    else:

                        if action== "wait":
                            print("等待")
                            self.loading_gif.hide()
                            self.all_dialogue_history.append(
                                {
                                    "role":"assistant",
                                    "action":"wait",
                                    "content":"wait",
                                }

                            )
                            return False


                if flag=="segment":#日文文本
                    self.voice_queue.put({"jp_text":reply["jp"],"cn_text":reply["cn"],"mood":mood+"."+self.ref_voice_format,"model_name":pet_id,"expression":expression})
                if flag=="end":
                    # 在服务端进行拼接校对成完整json格式
                    self.voice_queue.put("end")

                    return True
            time.sleep(1/60)



    def set_user_input_prompt(self,text):
        if text!="heart_beat":
            self.is_user_turn=False
        self.all_dialogue_history.append({"role": "user","content":datetime.now().strftime("%Y-%m-%d %H:%M:%S")+" | "+ text})

    def get_user_input(self):
        if self.upload_file_path!="":
            return True

        text = self.latest_input
        self.latest_input = ""
        if text == "":
            if self.in_initiative and self.is_user_turn and time.time() - self.last_chat_time > self.heartbeat_time:
                text = "heart_beat"
                self.last_chat_time=time.time()
                print("心跳")
            else:
                return False
        self.set_user_input_prompt(text)
        return True

    def get_model_ans(self):
        self.loading_gif.show()

        self.chat_solution()


    def chat(self):
        self.last_chat_time=time.time()
        self.min_heart_beat_time=5
        self.heartbeat_time=self.min_heart_beat_time

        while True:
            time.sleep(1 / 60)
            if self.is_on_chat and self.is_voice_flag==False:

                try:
                    if self.get_user_input()==False:
                        continue
                    self.get_model_ans()

                except Exception as e:
                    print(f"聊天流程异常: {e}")



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
            client_type="pc",
            chunk_sink=self.sequence_add,
            realtime_chunk_sink=self.realtime_sequence_add,
            realtime_event_sink=self.realtime_event_add,
        )
        self.AI_tool.set_history_search_callback(self.tags_match)
        self.AI_tool.set_conversation_action_callback(self.save_log_async)
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


    def shake_body(self):
        shake_param_label=self.effect_label_dict["shakeParam"]
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
        if x > 0.4 * WIDTH *SCALE_FACTOR and x < 0.6 * WIDTH *SCALE_FACTOR and y > 0.2 * HEIGHT*SCALE_FACTOR and y < 0.4 * HEIGHT*SCALE_FACTOR:
            self.get_random_sound()

    def mouseMoveEvent(self, event):
        if self.in_moving:
            self.move(self.window_origin_pos + QCursor.pos() - self.mouse_origin_pos)

    def wheelEvent(self, event):
        global SCALE_FACTOR
        delta = event.angleDelta().y()

        if delta > 0:
            SCALE_FACTOR+=0.1
            SCALE_FACTOR=min(SCALE_FACTOR,3)
        else:
            SCALE_FACTOR-=0.1
            SCALE_FACTOR=max(SCALE_FACTOR,1)
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



