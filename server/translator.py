# -*- coding:utf-8 -*-
#
#   author: iflytek
#
#  鏈琩emo娴嬭瘯鏃惰繍琛岀殑鐜涓猴細Windows + Python3.7
#  鏈琩emo娴嬭瘯鎴愬姛杩愯鏃舵墍瀹夎鐨勭涓夋柟搴撳強鍏剁増鏈涓嬶紝鎮ㄥ彲鑷閫愪竴鎴栬€呭鍒跺埌涓€涓柊鐨則xt鏂囦欢鍒╃敤pip涓€娆℃€у畨瑁咃細
#   cffi==1.12.3
#   gevent==1.4.0
#   greenlet==0.4.15
#   pycparser==2.19
#   six==1.12.0
#   websocket==0.2.1
#   websocket-client==0.56.0
#
#  璇煶鍚啓娴佸紡 WebAPI 鎺ュ彛璋冪敤绀轰緥 鎺ュ彛鏂囨。锛堝繀鐪嬶級锛歨ttps://doc.xfyun.cn/rest_api/璇煶鍚啓锛堟祦寮忕増锛?html
#  webapi 鍚啓鏈嶅姟鍙傝€冨笘瀛愶紙蹇呯湅锛夛細http://bbs.xfyun.cn/forum.php?mod=viewthread&tid=38947&extra=
#  璇煶鍚啓娴佸紡WebAPI 鏈嶅姟锛岀儹璇嶄娇鐢ㄦ柟寮忥細鐧婚檰寮€鏀惧钩鍙癶ttps://www.xfyun.cn/鍚庯紝鎵惧埌鎺у埗鍙?-鎴戠殑搴旂敤---璇煶鍚啓锛堟祦寮忥級---鏈嶅姟绠＄悊--涓€у寲鐑瘝锛?
#  璁剧疆鐑瘝
#  娉ㄦ剰锛氱儹璇嶅彧鑳藉湪璇嗗埆鐨勬椂鍊欎細澧炲姞鐑瘝鐨勮瘑鍒潈閲嶏紝闇€瑕佹敞鎰忕殑鏄鍔犵浉搴旇瘝鏉＄殑璇嗗埆鐜囷紝浣嗗苟涓嶆槸缁濆鐨勶紝鍏蜂綋鏁堟灉浠ユ偍娴嬭瘯涓哄噯銆?
#  璇煶鍚啓娴佸紡WebAPI 鏈嶅姟锛屾柟瑷€璇曠敤鏂规硶锛氱櫥闄嗗紑鏀惧钩鍙癶ttps://www.xfyun.cn/鍚庯紝鎵惧埌鎺у埗鍙?-鎴戠殑搴旂敤---璇煶鍚啓锛堟祦寮忥級---鏈嶅姟绠＄悊--璇嗗埆璇鍒楄〃
#  鍙坊鍔犺绉嶆垨鏂硅█锛屾坊鍔犲悗浼氭樉绀鸿鏂硅█鐨勫弬鏁板€?
#  閿欒鐮侀摼鎺ワ細https://www.xfyun.cn/document/error-code 锛坈ode杩斿洖閿欒鐮佹椂蹇呯湅锛?
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import _thread as thread
import os
import time
from time import mktime
import sounddevice as sd
import numpy as np
import asyncio
import queue

import websocket

import base64
import datetime
import hashlib
import hmac
import json
import ssl
from datetime import datetime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

STATUS_FIRST_FRAME = 0  # 绗竴甯х殑鏍囪瘑
STATUS_CONTINUE_FRAME = 1  # 涓棿甯ф爣璇?
STATUS_LAST_FRAME = 2  # 鏈€鍚庝竴甯х殑鏍囪瘑


ans_dict={}
is_closed=False
output_queue=queue.Queue()
class Ws_Param(object):
    # 鍒濆鍖?
    def __init__(self, APPID, APIKey, APISecret, AudioFile):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.AudioFile = AudioFile
        self.CommonArgs = {"app_id": self.APPID}
        self.BusinessArgs = {"domain": "iat", "language": "zh_cn", "accent": "mandarin", "vinfo": 1, "vad_eos": 10000}#"dwa":"wpgs"} #zh_cn en_us
        self.iat_params = {
            "domain": "slm", "language": "zh_cn", "accent": "mandarin", "result":
                {
                    "encoding": "utf8",
                    "compress": "raw",
                    "format": "plain"
                }
        }

    # 鐢熸垚url
    def create_url(self):
        url = 'wss://ws-api.xfyun.cn/v2/iat'
        # 鐢熸垚RFC1123鏍煎紡鐨勬椂闂存埑
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 鎷兼帴瀛楃涓?
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/iat " + "HTTP/1.1"
        # 杩涜hmac-sha256杩涜鍔犲瘑
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        # 灏嗚姹傜殑閴存潈鍙傛暟缁勫悎涓哄瓧鍏?
        v = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        # 鎷兼帴閴存潈鍙傛暟锛岀敓鎴恥rl
        url = url + '?' + urlencode(v)


        # 姝ゅ鎵撳嵃鍑哄缓绔嬭繛鎺ユ椂鍊欑殑url,鍙傝€冩湰demo鐨勬椂鍊欏彲鍙栨秷涓婃柟鎵撳嵃鐨勬敞閲婏紝姣斿鐩稿悓鍙傛暟鏃剁敓鎴愮殑url涓庤嚜宸变唬鐮佺敓鎴愮殑url鏄惁涓€鑷?

        return url


# 鏀跺埌websocket娑堟伅鐨勫鐞?
def on_message(ws, message):

    # print(message)
    message = json.loads(message)
    code = message["code"]
    # status = message["header"]["status"]

    if code != 0:
        ws.close()

    else:
        result:dict=message["data"]["result"]
        data = result["ws"]
        status = message["data"]["status"]
        # pgs=result["pgs"]
        sn=result["sn"]



        ans=""
        for i in data:
            for w in i["cw"]:
                ans += w["w"]
        ans_dict[sn]=ans
        add_ans(sn)
        # if pgs == "rpl":
        #     for i in range(result["rg"][0],result["rg"][1]+1):
        #         ans_dict[i]=""

        if status==2:
            # add_ans(sn)
            ws.close()

def add_ans(sn):
    ans = ""
    for i in range(1, sn + 1):
        ans += ans_dict[i]
    output_queue.put(ans)
# 鏀跺埌websocket閿欒鐨勫鐞?
def on_error(ws, error):
    pass


# 鏀跺埌websocket鍏抽棴鐨勫鐞?
def on_close(ws, close_status_code, close_msg):
    global is_closed
    is_closed=True


#
# # 鏀跺埌websocket杩炴帴寤虹珛鐨勫鐞?
# def on_open(ws):
#     global is_closed
#     print("open!")
#     is_closed=False
#     def run(*args):
#         frameSize = 1280  # 姣忎竴甯х殑闊抽澶у皬
#         intervel = 0.04  # 鍙戦€侀煶棰戦棿闅?鍗曚綅:s)
#         status = STATUS_FIRST_FRAME  # 闊抽鐨勭姸鎬佷俊鎭紝鏍囪瘑闊抽鏄涓€甯э紝杩樻槸涓棿甯с€佹渶鍚庝竴甯?
#
#         with open(wsParam.AudioFile, "rb") as fp:
#             while True:
#
#                 buf = fp.read(frameSize)
#                 audio = str(base64.b64encode(buf), 'utf-8')
#
#                 # 鏂囦欢缁撴潫
#                 if not buf:
#                     status = STATUS_LAST_FRAME
#                 # 绗竴甯у鐞?
#                 if status == STATUS_FIRST_FRAME:
#
#                     d = {"common": wsParam.CommonArgs,
#                          "business": wsParam.BusinessArgs,
#                          "data": {"status": 0, "format": "audio/L16;rate=16000",
#                                   "audio": str(base64.b64encode(buf), 'utf-8'),
#                                   "encoding": "raw"}}
#                     d = json.dumps(d)
#                     ws.send(d)
#                     status = STATUS_CONTINUE_FRAME
#                 # 涓棿甯у鐞?
#                 elif status == STATUS_CONTINUE_FRAME:
#                     d = {"data": {"status": 1, "format": "audio/L16;rate=16000",
#                                   "audio": str(base64.b64encode(buf), 'utf-8'),
#                                   "encoding": "raw"}}
#                     if is_closed:
#                         break
#                     ws.send(json.dumps(d))
#                 # 鏈€鍚庝竴甯у鐞?
#                 elif status == STATUS_LAST_FRAME:
#                     d = {"data": {"status": 2, "format": "audio/L16;rate=16000",
#                                   "audio": str(base64.b64encode(buf), 'utf-8'),
#                                   "encoding": "raw"}}
#                     if is_closed:
#                         break
#                     ws.send(json.dumps(d))
#                     break
#
#                 # 妯℃嫙闊抽閲囨牱闂撮殧
#                 time.sleep(intervel)
#         print("缁撴潫锛?)
#
#     thread.start_new_thread(run, ())

def translate(audio_data,id):
    # global result
    # path="users_temp/"+id
    # if os.path.exists(path)==False:
    #     os.makedirs(path, exist_ok=True)
    # path+="/upload_voice.pcm"
    # with open(path,"wb") as f:
    #     f.write(audio_data)


    # result=""
    wsParam.AudioFile="users_temp/superpai/upload_voice.pcm"#path
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = on_open
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    # print(result)
    # return result


# translate(1,1)

# 鍚啓 WebSocket 鎵撳紑鍚庡惎鍔ㄥ彂閫佺嚎绋?
def on_open(ws):
    def run(*args):
        frameSize = 8000  # 鍙牴鎹渶瑕佽皟鏁达紝浣嗗彂閫佹椂涓嶈兘鎷嗗紑闊抽甯э紝杩欓噷绠€鍖栦负鎸?q 鍙栫殑鍧楀彂閫?
        interval = 0.04
        status = STATUS_FIRST_FRAME

        while True:
            time.sleep(1/60)
            # 闃诲绛夊緟闊抽鏁版嵁
            if input_queue.empty():
                if is_on==False:
                    break

            else:
                pcm = input_queue.get()  # 姣忔鑾峰彇涓€涓?CHUNK 鐨勫瓧鑺傚潡
                # 涓€娆″洖璋冪殑鏁版嵁鍙兘姣?frameSize 澶ф垨灏忥紝鍙牴鎹渶瑕佹媶鍒嗘垨鍚堝苟
                # 杩欓噷绠€鍗曡捣瑙侊細姣忔鍥炶皟浜у嚭鐨勬暟鎹綔涓轰竴甯у彂閫?
                if status == STATUS_FIRST_FRAME:
                    d = {"common": wsParam.CommonArgs,
                         "business": wsParam.BusinessArgs,
                         "data": {"status": 0, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(pcm), 'utf-8'),
                                  "encoding": "raw"}}
                    ws.send(json.dumps(d))
                    status = STATUS_CONTINUE_FRAME
                else:
                    d = {"data": {"status": 1, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(pcm), 'utf-8'),
                                  "encoding": "raw"}}
                    ws.send(json.dumps(d))

            # 涓嶉渶瑕?sleep锛屽洜涓?q.get() 鑷劧闃诲锛屽疄鏃舵€ф洿濂?

        # 寰幆缁撴潫鍚庡彂閫佺粨鏉熷抚
        d = {"data": {"status": 2, "format": "audio/L16;rate=16000",
                      "audio": "",
                      "encoding": "raw"}}
        ws.send(json.dumps(d))
        time.sleep(1)
        ws.close()

    thread.start_new_thread(run, ())


def close():
    global is_on
    is_on = False

def start(queue):
    global output_queue
    output_queue=queue
    SAMPLE_RATE = 16000
    CHUNK_MS = 100
    CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)

    with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            callback=audio_callback
    ):
        wsUrl = wsParam.create_url()
        ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
        ws.on_open = on_open
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        while is_on:
            time.sleep(1 / 60)

def audio_callback(indata, frames, time_info, status):
    pcm = (indata * 32767).astype(np.int16).tobytes()
    input_queue.put(pcm)

wsParam = Ws_Param(APPID=os.getenv('XFYUN_APP_ID', '').strip(), APISecret=os.getenv('XFYUN_API_SECRET', '').strip(),
                       APIKey=os.getenv('XFYUN_API_KEY', '').strip(),
                       AudioFile="")
websocket.enableTrace(False)
input_queue = queue.Queue()
is_on=False
# 褰曢煶鍥炶皟锛氬皢闊抽鏁版嵁鏀惧叆闃熷垪
