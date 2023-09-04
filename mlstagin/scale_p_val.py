import numpy as np
import xlrd
import xlwt
import pandas as pd
import math

# 读取Excel文件
from scipy.stats import chi2_contingency, ttest_ind

path = 'F:\KeYan\癫痫\covariates.xlsx'
HC_data = pd.read_excel(path, sheet_name='XY_HC')
TLE_data = pd.read_excel(path, sheet_name='XY_TLE')

gender_HC = HC_data['性别'].tolist()
gender_TLE = TLE_data['Sex(rest)\nmale:1\nfemale:0'].tolist()
gender_TLE = [int(x) for x in gender_TLE if not math.isnan(x)]

F_HC = gender_HC.count(0)
M_HC = gender_HC.count(1)
F_TLE = gender_TLE.count(0)
M_TLE = gender_TLE.count(1)
print('HC: M/F: {}/{}, TLE: M/F: {}/{}'.format(M_HC, F_HC, M_TLE, F_TLE))

# chi square test
# 将频数数据转换为列联表
observed = np.array([[F_HC, M_TLE], [M_HC, F_TLE]])
chi2, p, dof, expected = chi2_contingency(observed)
print('T-val: {}, p-val: {}'.format(chi2, p))


age_HC = HC_data['年龄'].tolist()
age_TLE = TLE_data['Age'].tolist()
age_TLE = [int(x) for x in age_TLE if not math.isnan(x)]

# 执行卡方检验
age_HC_ = np.array(age_HC)
age_TLE_ = np.array(age_TLE)

print('Age: TLE: {} +- {}, HC: {} +- {}'.format(np.average(age_TLE_), np.std(age_TLE_), np.average(age_HC_), np.std(age_HC_)))
t_statistic, p = ttest_ind(age_HC_, age_TLE_)
print('chi2: {}, p-val: {}'.format(t_statistic, p))
print()
