# -*- coding:utf-8 -*-
import requests
import bs4
import time
import random
from tqdm import *
from bs4 import BeautifulSoup


def get_english_word():
    try:
        word_list = []
        for i in tqdm(range(66)):
            base_url = 'http://www.kuakao.com/english/ch/39' + str(183 + i) + '.html'
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.1 Safari/537.36'}
            r = requests.get(base_url, headers=headers)

            r.encoding = r.apparent_encoding

            soup = BeautifulSoup(r.content, 'lxml')

            all_words = soup.find('div', class_='artTxt')

            words = all_words.find_all('p')

            for i in range(len(words)):
                if type(words[i].string) == bs4.element.NavigableString:
                    if words[i].string != '跨考教育':
                        for string in words[i].stripped_strings:
                            word_list.append(string)
            time.sleep(5)
        time.sleep(.01)
        return word_list
    except Exception as e:
        print(e)


def write_word_to_file(word_list):
    for word in word_list:
        with open('word.txt', 'w') as f:
            f.write(word)
            f.write('\n')


word_list = get_english_word()

print(word_list)
