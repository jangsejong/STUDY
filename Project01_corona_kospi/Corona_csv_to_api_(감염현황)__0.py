import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup 
import requests
import csv
import time
import sys
from urllib.parse import urlencode
import xmltodict

#1) 데이터 로드하기

import requests
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from urllib.parse import urlencode, quote_plus, unquote
import json

# 코로나 API 가져오기
serviceKey = 'lKyyDlz+TK1rYYop8yeEkCqE//5+z/ZEugxxiNdsaG+ozoMwypoTgDOnBETRWj3kIVF3xvEHScN6OJ6zHzauAA==' # 공공데이터포털에서 발급받은 서비스키 입력
params = {'ServiceKey' : serviceKey,
          'pageNo' : '1',
          'numOfRows' : '10',
          'startCreateDt' : '20200210', # 데이터 호출범위지정(시작일)
          'endCreateDt' : '20211230' # 데이터 호출범위지정(종료일)
         }

# 서비스URL
url = 'http://openapi.data.go.kr/openapi/service/rest/Covid19/getCovid19InfStateJson?'
res = requests.get(url, params=params) # url 뒤에 params 정보를 붙임 
soup = BeautifulSoup(res.text, 'lxml')

# xml -> 데이터프레임 전환
items = soup.find_all('item') # item의 하위 항목들만 모두 가져옵니다. items에 보관


# 최상단에 있는 item만 선택해서 컬럼명을 가져옴
columns = []
for item in items[0]:
    columns.append(item.name) # item의 하위 항목 이름을 가져옵니다.


# 각 항목의 데이터 수집
final_data = []
for item in items:
    data = []
    for i in item: # item 하위 항목들을 한 줄씩 가져옵니다.

        # 21년 11월 24일 데이터 부터 accdefrate(누적 확진률) 데이터가 제외되므로 아예 수집에서 제외합니다.
        if i.name == 'accdefrate':
            pass
        else:
            data.append(i.text) # 항목에 해당하는 데이터를 가져옵니다.
    final_data.append(data)

# 데이터프레임으로 전환
df = pd.DataFrame(final_data, columns=columns)
df



# 영문 이름을 한글로 바꿉니다.
df = df.rename(columns={
    'accexamcnt': '사망자 수', #
    'createdt': '등록일시분초', 
    'deathcnt': '게시글번호', #
    'decidecnt': '확진자 수', #
    'seq': '게시글번호(감염현황 고유값)', #
    'statedt': '기준일', #
    'statetime': '기준시간', #
    'updatedt': '수정일시분초', #
})
# 원하는 순서대로 항목을 나열합니다.
df = df[[
    '게시글번호', '기준일', '기준시간', '확진자 수', '사망자 수', '게시글번호(감염현황 고유값)',
    '등록일시분초', '수정일시분초'
]]

# 기준일을 추후 분석의 편의를 위해 날짜 타입으로 변경합니다.
df['기준일'] = pd.to_datetime(df['기준일'])

# 엑셀 저장 (해당 .py또는 .ipynb 파일이 있는 폴더)
df.to_csv("../개인프로젝트/keras1000_Corona(감염현황).csv", mode='w',encoding='euc-kr')
df

print(df.info)


