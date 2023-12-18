import pandas as pd
data = pd.read_csv('RED Tag Page Data.csv', encoding='utf-8')
keywords = ["张桂梅", "校长", "我本是高山", "海清", "电影", "华坪", "女子高中", "教育", "大山", "山区", "纪录片"]
filtered_data = data[data['Title'].apply(lambda x: any(keyword in x for keyword in keywords))]
filtered_data.to_csv('/Users/wangsifan/PycharmProjects/pythonProject27/Filtered RED Tag Page Data.csv', index=False, encoding='utf-8')
print(filtered_data.head())
