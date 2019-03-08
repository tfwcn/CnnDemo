import pandas as pd
import numpy as np
import h5py
import scipy.io as scio
data_path = 'E:\\imdb_crop\\imdb.mat'
data = scio.loadmat(data_path)
# file = h5py.File(datapath,'r')
print("keys",data.keys())
imdb=data["imdb"]

dob=imdb['dob'][0,0][0]
photo_taken=imdb['photo_taken'][0,0][0]
full_path=imdb['full_path'][0,0][0]
gender=imdb['gender'][0,0][0]
name=imdb['name'][0,0][0]
face_location=imdb['face_location'][0,0][0]
face_score=imdb['face_score'][0,0][0]
second_face_score=imdb['second_face_score'][0,0][0] 
celeb_names=imdb['celeb_names'][0,0][0]
celeb_id=imdb['celeb_id'][0,0][0]


dob = pd.Series(dob)
photo_taken = pd.Series(photo_taken)
full_path = pd.Series(full_path)
gender = pd.Series(gender)
name = pd.Series(name)
face_location = pd.Series(face_location)
face_score = pd.Series(face_score)
second_face_score = pd.Series(second_face_score)
celeb_names = pd.Series(celeb_names)
celeb_id = pd.Series(celeb_id)
# 把列添加到表格
data = pd.DataFrame({'出生日期': dob,'拍照年份':photo_taken,'图片路径':full_path,
'性别':gender,'姓名':name,'人脸坐标':face_location,'人脸得分':face_score,'人脸得分2':second_face_score,
'名人名单':celeb_names,'名人id':celeb_id})

# print('data',data)

data.to_excel('imdb.xlsx', sheet_name='Sheet1')
# data = pd.read_excel('imdb.xlsx', 'Sheet1', index_col=None, na_values=['NA'])
