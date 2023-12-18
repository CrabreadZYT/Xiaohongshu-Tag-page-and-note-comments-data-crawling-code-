import jieba
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
from snownlp import SnowNLP
import numpy as np
df1=pd.read_csv('Filtered RED Tag Page Data.csv')
text = ' '.join(df1['Title'].dropna().astype(str))
words = jieba.cut(text)
stopwords_file = r'hit_stopwords.txt'
with open(stopwords_file, 'r', encoding='utf-8') as f:
    stopwords = set([line.strip() for line in f.readlines()])
words = [word.replace('\n', ' ') for word in words if word not in stopwords]
word_freq = Counter(words)

plt.rcParams['font.sans-serif']=['Songti SC','STFongsong']
plt.rcParams['axes.unicode_minus']=False
wordcloud = WordCloud(width=800, height=400, background_color='white',font_path ="Songti.ttc")
wordcloud.generate_from_frequencies(word_freq)
plt.figure(figsize=(16, 12))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('标题词云图', fontsize=16,loc='left',)
plt.savefig('标题词云图.png')
plt.show()
plt.close('all')

import warnings
warnings.filterwarnings('ignore')

top_20_notes = df1.nlargest(20, 'Likes')

# 获取点赞数最高的二十条笔记的标题和点赞数
top_20_titles = top_20_notes['Title']
top_20_likes = top_20_notes['Likes']

# 绘制条形图
plt.figure(figsize=(16,10))
plt.barh(top_20_titles, top_20_likes, color='skyblue')
plt.xlabel('Likes')
plt.title('Top 20 Notes by Likes')
plt.gca().invert_yaxis()  # 反转y轴，使点赞数最多的笔记显示在顶部
plt.tight_layout()
plt.savefig('点赞数前20标题图.png')
plt.show()
plt.close('all')

df=pd.read_csv('Filtered RED Tag Page Data.csv')
df['Create Time'] = pd.to_datetime(df['Create Time'])
df.set_index('Create Time', inplace=True)

filtered_data = df[df.index > '2023-10']
notes_count_by_date = filtered_data.resample('D').size()

# 绘制折线图
plt.figure(figsize=(16, 10))
notes_count_by_date.plot(marker='o', linestyle='-', color='b')
plt.xlabel('日期')
plt.ylabel('数量')
plt.title('每日笔记数量走势图')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('每日笔记数量走势图.png')
plt.show()
plt.close('all')

def analyze_sentiment(comment):
    s = SnowNLP(comment)
    sentiment_score = s.sentiments
    if sentiment_score > 0.60:
        return '积极'
    elif sentiment_score < 0.40:
        return '消极'
    else:
        return '中性'


df['sentiment'] = df['Title'].apply(analyze_sentiment)
sentiment_counts = df['sentiment'].value_counts()

colors = plt.cm.Set3.colors
labels = sentiment_counts.index

plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)

plt.title('笔记情感分布饼图', fontsize=16, loc='left')
plt.axis('equal')
plt.tight_layout()
plt.savefig('笔记情感分布饼图.png')
plt.show()
plt.close('all')

def analyze_sentiment(comment):
    comment=str(comment)
    s = SnowNLP(comment)
    sentiment_score = s.sentiments
    if sentiment_score > 0.60:
        return '积极'
    elif sentiment_score < 0.40:
        return '消极'
    else:
        return '中性'


df2=pd.read_csv('Filtered RED Note Comments Data.csv')
df2['sentiment'] = df2['Content'].apply(analyze_sentiment)
sentiment_counts = df2['sentiment'].value_counts()

colors = plt.cm.Set2.colors
labels = sentiment_counts.index

plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)

plt.title('评论情感分布饼图', fontsize=16, loc='left')
plt.axis('equal')
plt.tight_layout()
plt.savefig('评论情感分布饼图.png')
plt.show()
plt.close('all')

ip_location_counts = df2['IP Location'].value_counts()

plt.figure(figsize=(10, 10))

categories = ip_location_counts.index
values = ip_location_counts.values

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()

values = np.concatenate((values, [values[0]]))
angles += angles[:1]

ax = plt.subplot(111, polar=True)
plt.xticks(angles[:-1], categories, color='black', size=10)
ax.plot(angles, values, linewidth=1, linestyle='solid')
ax.fill(angles, values, 'b', alpha=0.1)

plt.title('IP地址分布雷达图',loc='left')
plt.savefig('IP地址分布雷达图.png')
plt.show()
plt.close('all')

text = ' '.join(df2['Content'].dropna().astype(str))
words = jieba.cut(text)
stopwords_file = r'hit_stopwords.txt'
with open(stopwords_file, 'r', encoding='utf-8') as f:
    stopwords = set([line.strip() for line in f.readlines()])
words = [word.replace('\n', ' ') for word in words if word not in stopwords]
word_freq = Counter(words)

wordcloud = WordCloud(width=800, height=400, background_color='white',font_path ="Songti.ttc")
wordcloud.generate_from_frequencies(word_freq)
plt.figure(figsize=(16, 12))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('关键词和短语提取', fontsize=16,loc='left')
plt.savefig('关键词和短语提取图.png')
plt.show()
plt.close('all')

import warnings
warnings.filterwarnings('ignore')

top_20_notes = df2.nlargest(20, 'Like Count')
print(top_20_notes[['Content','Like Count']])
top_20_Content = top_20_notes['Content']
top_20_likes = top_20_notes['Like Count']


plt.figure(figsize=(60, 40))
plt.barh(top_20_Content, top_20_likes, color='skyblue')
plt.xlabel('Like Count', fontsize=12)
plt.title('Top 20 Content by Like Count', fontsize=14)
plt.xticks(fontsize=20)
plt.yticks(fontsize=50)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('点赞数前20评论图.png')
plt.show()