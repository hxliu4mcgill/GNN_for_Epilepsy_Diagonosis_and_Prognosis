import os
import shutil

c = 'HC'
path = 'D:\data\ZD\BOLD'
path = os.path.join(path, c)
path_funraw = os.path.join(path, 'FunRaw')
path_funimg = os.path.join(path, 'FunImg')
path_temp = os.path.join(path, 'TEMP')
dis = 'F:\Data\ZD\RAW'
dis = os.path.join(dis, c)

dir_list = os.listdir(path)
more = []
for idx, sub in enumerate(os.listdir(path_funimg)):
    sub_path = os.path.join(path_funimg, sub)
    dcm_list = os.listdir(sub_path)

    if(len(dcm_list) == 1 and dcm_list[0].endswith('.nii')):
        more.append(dcm_list[0])
        # shutil.move(sub_path, path_temp)

    # for dcm in dcm_list:
    #     if dcm.endswith('.dcm'):
    #         shutil.move(os.path.join(sub_path, dcm), os.path.join(path, sub))
    # shutil.rmtree(sub_path)

move_list = []
for i in dir_list:
    sub_path = os.path.join(path, i)
    if len(os.listdir(sub_path)) > 1:
        move_list.append(sub_path)

for i in move_list:
    os.makedirs(os.path.join(dis, i.split('\\')[-1]), exist_ok=True)
    shutil.copytree(sub_path, dis)

