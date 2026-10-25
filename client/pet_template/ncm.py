import os
import sys
import time
from pathlib import Path

import requests


def _configure_musiclibrary_dll_path():
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return

    candidates = []
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            candidates.append(Path(meipass) / "MusicLibrary" / "lib")
        candidates.append(Path(sys.executable).resolve().parent / "_internal" / "MusicLibrary" / "lib")
    else:
        candidates.append(Path(__file__).resolve().parent / "_internal" / "MusicLibrary" / "lib")

    for dll_dir in candidates:
        if dll_dir.exists():
            os.add_dll_directory(str(dll_dir))
            os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
            break


_configure_musiclibrary_dll_path()

from MusicLibrary.neteaseCloudMusicApi import NeteaseCloudMusicApi, NcmProcessEnv

cookie={}
ncm_cnt=0
def init(cookie_parm):
    global ncm,id,user_name,player_list,all_lt_dict,cookie
    ncm = NeteaseCloudMusicApi()
    result=ncm.user_account(cookie=cookie_parm).body
    id=result["account"]["id"]
    user_name=result["profile"]["nickname"]
    cookie=cookie_parm

    player_list = ncm.user_playlist(id).body["playlist"]
    all_lt_dict = {}
    for i in player_list:
        playlist = ncm.playlist_detail(i["id"]).body["playlist"]
        lt_name = playlist["name"]

        lt_id = playlist["id"]
        track_lt = playlist["tracks"]  # 单个歌单的全部歌曲
        all_lt_dict[lt_name] = track_lt

    all_lt_dict["日推歌单"]=ncm.recommend_songs(cookie=cookie).body["data"]["dailySongs"]

# 创建 API 实例
def download_music(url):
    response=requests.get(url)
    tail=response.headers["Content-Disposition"].split(".")[1].strip('"')
    data=response.content
    name="NT."+tail
    with open(name,"wb") as file:
        file.write(data)

    return name

def get_music(id):
    url=ncm.song_url_v1(id=id,cookie=cookie,level="standard").body["data"][0]["url"]
    if url!=None:
        return download_music(url)

def get_like_music_lt():
    return get_music_lt(user_name+"喜欢的音乐")

def get_music_lt(name):
    global ncm_cnt
    lt=[]
    all_id=""
    for i in all_lt_dict[name]:
        # print(i)
        song_id=str(i["id"])

        lt.append([song_id,i["name"],0])


    return lt
#
# ncm=NeteaseCloudMusicApi()
# print(ncm.user_account({
#     "MUSIC_R_T": "1624346318561",
#     "MUSIC_A_T": "1624346318551",
#     "NMTID": "00ONyRwUr1HdsZi1UAxo7CMmd9O9IwAAAGebshz8A",
#     "__csrf": "105cc6195f038ebc82b8cd1f915d84b8",
#     "MUSIC_U": "00C1B359D900B794E12CBE494E8FDB6AE24C92881041A57C152AD8E0CDFCAC09C1F8E4BFCA442D693C737C4428BA5DE2BE78080974B4E4FB6618E0D4E14804FD238560E5EE5B87C9038B1ED0D7B59D1A96D2DBB322889F315358D48448BDE52D4826F8799E8016B500C7C9413B6E3D86B4620B5C450AF5C8BBB8B0CC1F575393DF46B009212ECC27E0668802BB2D419188AFD5266207A8E2562BB0C840D6DFD5B5DE5C289B70EBC9EC990DDBE4A48BF8F866D4D697A870879D3B117EF23E09E2A88AC75D4FE991E28EEE7B539FA59877E88D192AD9E7CA7BA618EF236D47824A6D217C80253B0162799C48374AACE0B8416C9BE248EF628952F74F6F6BE0C9F071A4082ABF5F1B1B27220ADD393315E7410917981BD29756EB50A8988C26407C2C6F82F791E949AD0041CDC52D43856FF55C5B97B27A2F567B2CE249EB02A61AF7DD5EAFC3FECE536018F2A4EB750C3ADC5F9E469E234AFFAE7FC4D0A406ED55BB7973DAB1861B56D64993B0E275D5D004",
#     "MUSIC_SNS": "",
#     "MUSIC_R_U": "00F1725477E84AF7921EABB2AF6996BAF8885AA09234EB1F7EAAEFAA57F81B1ED6D10DDA092032187DE3326CE2A5A9D5897A47318E98765EBC27E8AC4A79530CF74320A5D9FBAB1282290F329300D4C50E"
#   }))
