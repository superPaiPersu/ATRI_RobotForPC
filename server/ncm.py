import json
import os
import time
from pathlib import Path

import requests

try:
    from MusicLibrary.neteaseCloudMusicApi import NeteaseCloudMusicApi, NcmProcessEnv
    _NCM_IMPORT_ERROR = None
except Exception as exc:
    NeteaseCloudMusicApi = None
    NcmProcessEnv = None
    _NCM_IMPORT_ERROR = exc


class _UnavailableNcm:
    def __getattr__(self, name):
        raise RuntimeError(
            "NetEase Music integration is unavailable on this server: "
            f"{_NCM_IMPORT_ERROR}"
        )


def _require_ncm():
    if NeteaseCloudMusicApi is None:
        raise RuntimeError(
            "NetEase Music integration is unavailable on this server: "
            f"{_NCM_IMPORT_ERROR}"
        )
    return ncm

ncm = NeteaseCloudMusicApi() if NeteaseCloudMusicApi is not None else _UnavailableNcm()
cookie = {}

BASE_DIR = Path(__file__).resolve().parent.parent
COOKIE_DIR = BASE_DIR / "data" / "cookies"


def _get_cookie_path(cellphone_num: str) -> Path:
    return COOKIE_DIR / f"{cellphone_num}.json"


def save_cookie(cellphone_num: str, cookie_dict: dict):
    """鎸夋墜鏈哄彿淇濆瓨 cookie 鍒?JSON 鏂囦欢"""
    os.makedirs(COOKIE_DIR, exist_ok=True)
    path = _get_cookie_path(cellphone_num)
    data = {"cellphone": cellphone_num, "cookie": cookie_dict}
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_cookie(cellphone_num: str) -> dict | None:
    """璇诲彇缂撳瓨鐨?cookie锛屼笉瀛樺湪鍒欒繑鍥?None"""
    path = _get_cookie_path(cellphone_num)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("cookie")
    except (json.JSONDecodeError, IOError):
        return None


def init():
    global ncm,id,user_name,player_list,all_lt_dict,cookie

    _require_ncm()
    ncm = NeteaseCloudMusicApi()
    # print(ncm.recommend_resource())
    id = "5059494246"
    user_name=ncm.user_detail(id).body["profile"]["nickname"]

    cookie= parse_cookies([])
    cookie=parse_cookies(ncm.login_refresh(cookie=cookie).cookies)
    # cookie = r.cookies

    # print(cookie)
    # ncm.login_qr_create(ncm.login_qr_key())
    # print(ncm.login_status())
    # print(ncm.login_status(cookie=cookie))
    # print(ncm.cookie)
    # id=response.data["account"]["id"]
    # print(id)
    # print(cookie)
    # #
    player_list = ncm.user_playlist(id).body["playlist"]
    all_lt_dict = {}
    for i in player_list:
        playlist = ncm.playlist_detail(i["id"]).body["playlist"]
        lt_name = playlist["name"]
        lt_id = playlist["id"]
        track_lt = playlist["tracks"]  # 鍗曚釜姝屽崟鐨勫叏閮ㄦ瓕鏇?
        all_lt_dict[lt_name] = track_lt

    all_lt_dict["鏃ユ帹姝屽崟"]=ncm.recommend_songs(cookie=cookie).body["data"]["dailySongs"]

# 鍒涘缓 API 瀹炰緥
def download_music(url):
    response=requests.get(url)
    tail=response.headers["Content-Disposition"].split(".")[1].strip('"')
    data=response.content
    name="NT."+tail
    with open(name,"wb") as file:
        file.write(data)

    return name

def user_login(cellphone_num):
    user_cookie=load_cookie(cellphone_num)
    if user_cookie==None:
        user_login_send_code(cellphone_num)

    else:
        user_login_by_cookie(cellphone_num,user_cookie)


def user_login_by_cookie(cellphone_num,user_cookie):
    user_cookie = parse_cookies(_require_ncm().login_refresh(cookie=user_cookie).cookies)
    save_cookie(cellphone_num=cellphone_num,cookie_dict=user_cookie)

def user_login_send_code(cellphone_num):#瀛楃涓茬被鍨?

    _require_ncm().captcha_sent(cellphone_num)

def user_login_check_code(cellphone_num,code):
    response=_require_ncm().login_cellphone(cellphone_num, captcha=code)
    if response.status==200:
        cookie=parse_cookies(response.headers['Set-Cookie'])

        return {"status":"success","cookie":cookie}
    else:
        return {"status":"failed","code":response.status}
def get_music(id):
    url=_require_ncm().song_url_v1(id=id,cookie=cookie,level="standard").body["data"][0]["url"]
    if url!=None:
        return download_music(url)


def read_cookie():
    with open("cookie.txt","r",encoding="utf-8") as f:
        return f.read()
def write_cookie():
    with open("cookie.txt","w",encoding="utf-8") as f:
        f.write(str(cookie))
def parse_cookies(cookieLt):
    cookie = {}
    for item in cookieLt:
        item = item.split("; ")[0]
        # 鎸夌収绗竴涓瓑鍙峰垎鍓诧紝鑾峰彇閿€煎
        k, v = item.split("=", 1)
        cookie[k.strip()] = v.strip()
    return cookie

def get_like_music_lt():
    return get_music_lt(user_name)
cnt=0
def get_music_lt(name):
    global cnt
    lt=[]
    all_id=""
    for i in all_lt_dict[name]:
        # print(i)
        song_id=str(i["id"])

        lt.append([song_id,i["name"],0])



    #
    #
    #     if ncm.check_music(song_id).body["success"]==True:
    #         all_id+=song_id+","
    #     else:
    #         print("姝屾洸鑾峰彇澶辫触")
    #         print(song_id)
    #
    # try:
    #     all_id=all_id.strip(",")
    #
    #     response=ncm.song_url_v1(all_id,level="standard",cookie=cookie)
    #
    #     for item in response.body["data"]:
    #         song_url=item["url"]
    #     # song_url=response.body["data"][0]["url"]
    #         if song_url!=None:
    #             lt.append(song_url)
    #
    # except Exception as e:
    #     print("閿欒锛?)
    #     print(e)
    #

    return lt
#
#
