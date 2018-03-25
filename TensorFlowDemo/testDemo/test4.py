import pandas as pd
import numpy as np

print(pd.__version__)

# 定义列数据
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
city_names += '(new)'
population += 1
# 把列添加到表格
city_data = pd.DataFrame({'City name': city_names, 'Population': population})
city_data['新列'] = city_data['Population']+1

print(city_data['City name'][0])

# print(city_data.describe())

print(type(city_data[0:2]))
print(city_data[0:2])

print(city_data['City name'].apply(lambda name: name.startswith('San')))

# 随机排序
print(city_data.reindex(np.random.permutation(city_data.index)))

print(city_data.reindex([6,5,4]))