"""
raw:
{
    "id": 137334,
    "claim": "Fox 2000 Pictures released the film Soul Food.",
    "label": "SUPPORTS",
    "gold_evidence": [
      [
        "Soul Food is a 1997 American comedy-drama film produced by Kenneth `` Babyface '' Edmonds , Tracey Edmonds and Robert Teitel and released by Fox 2000 Pictures . Robert Teitel Robert Teitel comedy-drama film comedy-drama film Tracey Edmonds Tracey Edmonds Fox 2000 Pictures Fox 2000 Pictures"
      ]
    ]
  },
  {
    "id": 111897,
    "claim": "Telemundo is a English-language television network.",
    "label": "REFUTES",
    "gold_evidence": [
      [
        "Telemundo ( [ tele\u02c8mundo ] ) is an American Spanish-language terrestrial television network owned by Comcast through the NBCUniversal division NBCUniversal Telemundo Enterprises . NBCUniversal NBCUniversal Spanish Spanish language American United States Spanish-language Spanish language terrestrial Terrestrial television television network television network Comcast Comcast"
      ],
      [
        "It is the second largest provider of Spanish content nationwide behind American competitor Univision , with programming syndicated worldwide to more than 100 countries in over 35 languages . Spanish Spanish language American United States Univision Univision"
      ],
      [
        "The channel broadcasts programs and original content aimed at Hispanic and Latino American audiences in the United States and worldwide , consisting of telenovelas , sports , reality television , news programming , and films -- either imported or Spanish-dubbed . United States United States Spanish Spanish language American United States Hispanic and Latino American Hispanic and Latino Americans sports sports reality television reality television",
        "Hispanic Americans and Latino Americans ( hispanos [ is\u02c8panos ] ) are American descendants from Spain and the Spanish speaking countries of Latin America . Spanish Spain American Americans Spain Spain Latin America Latin America Hispanic Hispanic Latino Latino (demonym)"
      ],
      [
        "In addition , Telemundo operates NBC Universo , a separate channel directed towards young Hispanic audiences ; Telemundo Digital Media , which distributes original programming content across mass media , the Telemundo and NBC Universo websites ; Puerto Rico telestation WKAQ-TV ; and international distribution arm Telemundo Internacional . Puerto Rico Puerto Rico NBC NBC NBC Universo NBC Universo mass media mass media telestation Television station WKAQ-TV WKAQ-TV"
      ]
    ]
  },
converted_data:
{
    "id":  "642ef759-aec2-4cec-97cc-7ecdc104dfca",
    "claim":  "Mark O\u0027Connor is an Amercian bluegrass singer who performed the song Restless with The New Nashville Cats.",
    "label":  "supports",
    "num_hops":  2,
    "evidence":  "\"Restless\" is a 1968 song written by Carl Perkins and released as a single on Columbia Records. The song was recorded on September 27, 1968, and released as a 45 single, 4-44723, on Columbia, in December, 1968, backed with \"11-43\", reaching no. 20 on the \"Billboard\" country chart. The recording, produced by Bill Denny and Larry Butler, also appeared on the May, 1969 Columbia LP \"Carl Perkins\u0027 Greatest Hits\". The song also appeared on the 1992 Carl Perkins compilation album \"Restless: The Columbia Recordings\". The song became a major hit again in 1991 in a new all-star recording by Mark O\u0027Connor and The New Nashville Cats. Carl Perkins performed the song on the Kraft Music Hall episode hosted by Johnny Cash on April 16, 1969.\nMark O\u0027Connor (born August 5, 1961, Seattle) is an American bluegrass, jazz and country violinist, fiddler, composer and music teacher. O\u0027Connor has received numerous awards for both his playing and his composition."
},

"""

import json
import re

raw_path = 'data/FEVER/raw/symmetric_generated.json'
converted_path = 'data/FEVER/converted_data/symmetric.json'

from collections import OrderedDict


def clean_evidence(raw_evidence):
    """
    最小清洗 evidence：
    只清除 LLM 不易识别/容易干扰的字符，其余内容尽量不改动。

    主要处理：
    1. 删除 IPA 音标、特殊音标符号
    2. 删除非 ASCII 外文字符，例如阿拉伯文、部分特殊拉丁字符
    3. 删除孤立的方括号、异常括号符号
    4. 规范多余空格
    """

    if raw_evidence is None:
        return ""

    text = str(raw_evidence)

    # 1. 删除方括号中的音标内容
    # 例如 [ ˌmɔːɡəˈdiːʃuː ]
    text = re.sub(r"\[[^\[\]]*\]", " ", text)

    # 2. 删除明显的 IPA / 音标特殊字符
    # 保留普通英文、数字、标点
    text = re.sub(
        r"[ˌˈːəɔɡʃʉɪɒɑæɛɜɞɨɯɵøœŋðθʒʤʧʊʌɚɝɾʔ]",
        " ",
        text,
    )

    # 3. 删除非 ASCII 字符
    # 会去掉阿拉伯文、特殊变音字母等
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # 4. 删除孤立的括号符号
    # 不删除括号内英文内容，只去掉括号本身
    text = text.replace("(", " ").replace(")", " ")
    text = text.replace("[", " ").replace("]", " ")

    # 5. 规范标点前后的空格
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([,.!?;:])(?=\S)", r"\1 ", text)

    # 6. 压缩多余空格
    text = re.sub(r"\s+", " ", text).strip()

    return text

with open(raw_path, 'r') as f:
    raw_data = json.load(f)

converted_data = []
for item in raw_data:
    evidence = clean_evidence(item.get('gold_evidence', []))
    label = item['label'].lower()
    converted_data.append({
        'id': item['id'],
        'claim': item['claim'],
        'label': label,
        'num_hops': -1,
        'evidence': evidence
    })

with open(converted_path, 'w') as f:
    json.dump(converted_data, f, indent=4)