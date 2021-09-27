import json
import xml.etree.ElementTree as ET
from lxml import etree
from tqdm import trange
from harvest import *
import pyhanlp
import re
import os


def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print('%s -> data over' % file)
    return data


def read_xml(file):
    # xmlstring = open(file).read()
    # parser = etree.XMLParser(recover=True)
    # tree = etree.fromstring(xmlstring, parser=parser)
    parser = ET.XMLParser(encoding="utf-8")
    tree = ET.parse(file, parser=parser)
    root = tree.getroot()
    data = []
    for doc in root:
        doc_txt = ''
        for sent in doc:
            sent_txt = sent.text
            doc_txt = doc_txt + sent_txt + ' '
        for sent in doc:
            sent_txt = sent.text
            if 'Train' in file or 'Dev' in file:
                if sent.get('label'):
                    data.append([sent.get('label'), sent_txt, doc_txt])
                continue
            if 'Test' in file:
                data.append([sent_txt, doc_txt])
    if 'Train' in file or 'Dev' in file:
        data = [['label', 'text_a', 'text_b']] + data
    if 'Test' in file:
        data = [['text_a', 'text_b']] + data
    return data


def save_json(data, file, indent=1):
    with open(file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, indent=1, ensure_ascii=False))
    print('data -> %s over' % file)


def save_file(filepath, lst):
    with open(filepath, 'w', encoding='utf-8') as f:
        for i in range(len(lst)):
            f.write(lst[i])
            if i != len(lst) - 1:
                f.write('\n')


def save_tsv(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        for i in range(len(data)):
            f.write('\t'.join(data[i]))
            if i != len(data) - 1:
                f.write('\n')


def remove_url(src):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b',
                   '', src, flags=re.MULTILINE)
    return vTEXT

# file: 数据文件


def clean_text(file, save_dir):
    # ht = HarvestText()
    CharTable = pyhanlp.JClass('com.hankcs.hanlp.dictionary.other.CharTable')
    data = read_xml(file)
    columns = {}
    num_null = 0
    cleaned_data = []
    neural_idx_in_test = []
    for i in trange(len(data)):
        if i == 0:
            for i, column_name in enumerate(data[i]):
                columns[column_name] = i
            cleaned_data.append(data[0])
            continue
        content_a = CharTable.convert(data[i][columns["text_a"]])
        cleaned_content_a = remove_url(harvest_clean_text(
            content_a, emoji=False)).strip()  # 过滤@后最多6个字符
        content_b = CharTable.convert(data[i][columns["text_b"]])
        cleaned_content_b = remove_url(harvest_clean_text(
            content_b, emoji=False)).strip()
        num_null += 1 if cleaned_content_a == '' else 0
        # 删除train中的自带的空数据或清洗后出现的空数据
        if ('Train' in file or 'Dev' in file) and (not content_a or not cleaned_content_a):
            continue
        if 'Test' in file:
            data[i][columns["text_a"]
                    ] = content_a if not cleaned_content_a else cleaned_content_a
            if not cleaned_content_a:
                neural_idx_in_test.append(str(i-1))
            data[i][columns["text_b"]] = cleaned_content_b
            cleaned_data.append(data[i])
        else:
            data[i][columns["text_a"]] = cleaned_content_a
            data[i][columns["text_b"]] = cleaned_content_b
            cleaned_data.append(data[i])
    filename = file.split('/')[-1].split('.')[0]
    save_tsv(cleaned_data, os.path.join(save_dir, filename + '.tsv'))
    if 'Test' in file:
        save_file('neural_idx_in_test.txt', neural_idx_in_test)
    print('num data: ', num_null)


clean_text('./data/raw/SMP2019_ECISA_Train.xml', './data/nb_clean')
clean_text('./data/raw/SMP2019_ECISA_Dev.xml', './data/nb_clean')
clean_text('./data/raw/ECISA2021-Test.xml', './data/nb_clean')
