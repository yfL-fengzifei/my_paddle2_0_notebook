import os
import zipfile
import shutil
import numpy

PATH=os.getcwd()
path=os.path.join(PATH,'car_licence.zip')
unzip_path=os.path.join(PATH,'car_licence')

# #解压
# z=zipfile.ZipFile(path,'r')
# for name in z.namelist():
#     z.extract(name,unzip_path)
# """
# z=zipfile.ZipFile(src_path,'r)
# z.extrctall(path=target_path)
# z.close
# """

#删除无关文件
# shutil.rmtree(os.path.join(unzip_path,'__MACOSX'))

data_folders=os.listdir(unzip_path)
# print(len(data_folders),data_folders)

"""
===version 1
"""
"""
# if os.path.exists('./train_data.list'):
#     os.remove('./train_data.list')
# if os.path.exists('./test_data.list'):
#     os.remove('./test_data.list')


# label=0
# label_map={}

# for data_folder in data_folders:
#     with open('./train_data.list','a') as f_train:
#         with open('./test_data.list','a') as f_test:

#             if data_folder == '.DS_Store' or data_folder == '.ipynb_checkpoints' or data_folder == 'car_licence.zip':
#                 continue

#             label_map[label]=data_folder
#             print('data_folder:{} --> label:{} '.format(data_folder,label))

#             char_img_path=os.listdir(os.path.join(unzip_path,data_folder))

#             for i,img_path in enumerate(char_img_path):
#                 if img_path=='.DS_Store':
#                     continue
#                 if i%10==0:
#                     f_test.write(os.path.join(unzip_path,data_folder,img_path)+'\t'+str(label)+'\n')
#                 else:
#                     f_train.write(os.path.join(unzip_path,data_folder,img_path)+'\t'+str(label)+'\n')
#     label+=1
"""

"""
===version 2
"""
"""
# wrong
# label_map = {10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',19:'J',20:'K',21:'L',22:'M',23:'N',
#         24:'O',25:'P',26:'Q',27:'R',28:'S',29:'T',30:'U',31:'V',32:'W',33:'X',34:'Y',35:'Z',
#         36:'yun',37:'cuan',38:'hei',39:'zhe',40:'ning',41:'jin',42:'gan',43:'hu',44:'liao',45:'jl',46:'qing',47:'zang',
#         48:'e1',49:'meng',50:'gan1',51:'qiong',52:'shan',53:'min',54:'su',55:'xin',56:'wan',57:'jing',58:'xiang',59:'gui',
#         60:'yu1',61:'yu',62:'ji',63:'yue',64:'gui1',65:'sx',66:'lu',
#         0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9'}
"""

#标签
label_map = {10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'J', 19: 'K',
             20: 'L', 21: 'M',22: 'N',23: 'P', 24: 'Q', 25: 'R', 26: 'S', 27: 'T', 28: 'U', 29: 'V',
             30: 'W', 31: 'X', 32: 'Y', 33: 'Z',34: 'yun', 35: 'cuan', 36: 'hei', 37: 'zhe', 38: 'ning', 39: 'jin',
             40: 'gan', 41: 'hu', 42: 'liao',43: 'jl', 44: 'qing', 45: 'zang',46: 'e1', 47: 'meng', 48: 'gan1',
             49: 'qiong', 50: 'shan', 51: 'min', 52: 'su', 53: 'xin', 54: 'wan',55: 'jing', 56: 'xiang', 57: 'gui',
             58: 'yu1', 59: 'yu', 60: 'ji', 61: 'yue', 62: 'gui1', 63: 'sx', 64: 'lu',
             0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

#逆标签
label_map_inv = {}
for k in label_map.keys():
    label_map_inv[label_map[k]] = k
# print(label_map_inv)
# print(type(label_map_inv.keys()))

#删除数据文件
train_list_path=os.path.join(PATH,'train_data.list')
test_list_path=os.path.join(PATH,'test_data.list')
if os.path.exists(train_list_path):
    os.remove(train_list_path)
if os.path.exists(test_list_path):
    os.remove(test_list_path)
"""
清空文件，不是删除再创建
with open(train_list_path,'w') as f:
    f.seek(0)  #指针指向的位置
    f.truncate(size) #size指定文件截断位置，空表示从当前位置截断，后面的数据被删除
"""

for data_folder in data_folders:
    with open(train_list_path, 'a',encoding='utf-8') as f_train:
        with open(test_list_path, 'a',encoding='utf-8') as f_test:

            if data_folder in label_map_inv:
                print('data_folder:{} --> label:{} '.format(data_folder,label_map_inv[data_folder]))

                char_img_path = os.listdir(os.path.join(unzip_path, data_folder))

                for i, img_path in enumerate(char_img_path):
                    if img_path == '.DS_Store':
                        continue

                    if i % 10 == 0:
                        f_test.write(os.path.join(unzip_path, data_folder, img_path) + '\t' + str(label_map_inv[data_folder]) + '\n')
                    else:
                        f_train.write(os.path.join(unzip_path, data_folder, img_path) + '\t' + str(label_map_inv[data_folder]) + '\n')