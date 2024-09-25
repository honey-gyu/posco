#!/usr/bin/env python
# coding: utf-8

# 1. 도메인 정보 파악
#     - 명주쌤이 해주심!
# 2. 데이터 특성 파악 후 전처리
#     1) 결측치
#         - isnull().sum() -> 조회 안되지만 0값 존재하므로 확인 필요
#     2) 이상치
#         -
#     3) 변수간 상관관계
#         - 압연온도, 균열대 재로시간
# 3. 그래프를 통한 데이터 분석
# 4. 모델링/요약
# 5. 핵심인자 도출
# 6. 경쟁력 확보 방안 도출
# 7. 배운점/느낀점

# ![image.png](attachment:image.png)

# In[20]:


# 데이터 구성:Series, DataFrame
import pandas as pd
# 행렬 연산
import numpy as np
# 데이터 시각화
import matplotlib.pyplot as plt
import matplotlib
# scaling
from sklearn.preprocessing import StandardScaler
# 데이터 분할:train, test
from sklearn.model_selection import train_test_split
# 로지스틱 회귀
from statsmodels.api import Logit
# 분류모델 평가 함수
from sklearn.metrics import accuracy_score, f1_score 
from sklearn.metrics import confusion_matrix, classification_report


# In[21]:


df = pd.read_csv("/home/piai/다운로드/2주차 실습파일/2. Big Data 분석/SCALE불량.csv",encoding='euc-kr')
df.head()


# In[22]:


df['scale'].replace({"양품":0,"불량":1},inplace=True)


# In[23]:


df.columns


# ###  컬럼 제거

# In[24]:


df.drop(columns=['plate_no','spec_long','spec_country'],axis=1,inplace=True)


# In[25]:


df.head()


# ### 압연 온도 -> 0 제거

# In[26]:


df = df[df['rolling_temp'] != 0]


# ### boxplot

# In[27]:


# 이상치 확인
variables = ['pt_thick', 'pt_width', 'pt_length', 'fur_heat_temp', 'fur_heat_time', 'fur_soak_temp', 'fur_soak_time',
             'fur_total_time', 'rolling_temp', 'descaling_count']

# Boxplot
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8))  # 3행 4열의 subplot 생성
axes = axes.flatten()  # 2차원 배열을 1차원으로 펼치기

for i, variable in enumerate(variables):
    axes[i].boxplot(df[variable])
    axes[i].set_title(f"Boxplot of {variable}")

# subplot 간 간격 조정
plt.tight_layout()

# 그래프 출력
plt.show()


# ### 설명변수간 상관관계

# In[28]:


import seaborn as sns
corr = df.corr()
sns.heatmap(corr,annot=True)


# In[29]:


sns.pairplot(df)


# In[30]:


df[df['fur_soak_time'] > 120]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[31]:


pd.pivot_table(data=df,index='scale',columns='rolling_method',values='work_group',aggfunc='count')


# In[32]:


pd.pivot_table(data=df,index='scale',columns='work_group',values='plate_no',aggfunc='count')


# In[ ]:


# df['rolling_date'] = pd.to_datetime(df['rolling_date'], format='%d%b%Y:%H:%M:%S')

# df['month'] = df['rolling_date'].dt.month


# In[33]:


df.head()


# In[34]:


pd.pivot_table(data=df,index='scale',columns='month',values='work_group',aggfunc='count')


# In[ ]:


# df['hour'] = df['rolling_date'].dt.hour


# In[ ]:


# df_h = pd.DataFrame(df['hour'].value_counts())
# df_h.sort_indexdex(inplace=True)


# In[ ]:


df_h


# In[ ]:


pd.pivot_table(data=df,index='scale',columns='hour',values='work_group',aggfunc='count').T


# In[ ]:


df.columns


# In[35]:


pd.pivot_table(data=df,index='scale',columns=['fur_no','fur_input_row'],values='hsb',aggfunc='count')


# In[36]:


corr = df.corr()
sns.heatmap(corr,annot=True)


# In[37]:


# 사람에 관련된 부분에선 지양(개선안 도출 어려움)


# In[38]:


pd.pivot_table(data=df,index='scale',columns='hsb',values='pt_thick',aggfunc='count')


# In[39]:


pd.pivot_table(data=df,index='scale',columns='descaling_count',values='pt_thick',aggfunc='count')


# ### 모델링

# 1. Logistic regression

# In[40]:


# train_test_split(데이터, test_size = test 데이터 비율, random_state: 랜덤)
df_train, df_test = train_test_split(df, # 데이터
                                     test_size = 0.3, # test 데이터의 비율
                                     random_state = 1234)  # random state

print("train data size : {}".format(df_train.shape))
print("test data size : {}".format(df_test.shape))


# In[41]:


df.info()


# In[42]:


df_train.shape


# In[46]:


# from_formula 함수를 이용하여 변수 역할 지정
formula = """scale ~ C(steel_kind) + pt_thick + pt_length + C(hsb) + C(fur_no) + C(fur_input_row) +
            fur_heat_temp + fur_heat_time + fur_soak_temp + fur_soak_time + fur_total_time +
            C(rolling_method) + rolling_temp + descaling_count + C(work_group)"""

log_model = Logit.from_formula(formula, df_train)
# 적합
log_result = log_model.fit()

# 결과 출력
print(log_result.summary())


# In[49]:


# 회귀계수가 유의한 변수만 사용한 모델
# 회귀계수 유의성 기준 제외변수: REASON, LOAN, MORTDUE, VALUE, YOJ 
log_model = Logit.from_formula("""scale ~ rolling_temp + descaling_count + fur_heat_temp + C(fur_no)""", df_train)
# 적합
log_result = log_model.fit()

# 결과 출력
print(log_result.summary())


# In[50]:


# train 데이터 예측
y_pred_train = log_result.predict(df_train)
# 0과 1의 값을 가진 class로 변환
y_pred_train_class = (y_pred_train > 0.5).astype(int)  # 0.5 : “1/0” 판정 임계값(1 발생 확률) 변경 가능 
print("Train 예측 결과 \n", y_pred_train_class.head(), "\n")
print("Confusion Matrix: \n{}".format(confusion_matrix(df_train["scale"],y_pred_train_class)),"\n")

# test 데이터 예측
y_pred_test = log_result.predict(df_test)
# 0과 1의 값을 가진 class로 변환
y_pred_test_class = (y_pred_test > 0.5).astype(int)
print("Test 예측 결과 \n", y_pred_test_class.head(),"\n")
print("Confusion Matrix: \n{}".format(confusion_matrix(df_test["scale"],y_pred_test_class)),"\n")


# In[52]:


# 실제 train 데이터와 예측 결과 비교
print("Train 예측/분류 결과")
print("Accuracy: {0:.3f}\n".format(accuracy_score(df_train["scale"], y_pred_train_class)))
print("Confusion Matrix: \n{}".format(confusion_matrix(df_train["scale"],y_pred_train_class)),"\n")
print(classification_report(df_train["scale"], y_pred_train_class, digits=3))

# 실제 train 데이터와 예측 결과 비교
print("Test 예측/분류 결과")
print("Accuracy: {0:.3f}\n".format(accuracy_score(df_test["scale"], y_pred_test_class)))
print("Confusion Matrix: \n{}".format(confusion_matrix(df_test["scale"],y_pred_test_class)),"\n")
print(classification_report(df_test["scale"], y_pred_test_class, digits=3))


# In[53]:


# 목표변수의 빈도 불균형 : f1 score로 모델 평가 
print("Train 예측/분류 결과")
# print(classification_report(df_test["BAD"], y_pred_test_class, digits=3))
print(classification_report(df_test["scale"], y_pred_test_class, target_names=['승인', '거절'], digits=3))

print("Test 예측/분류 결과")
# 목표변수의 빈도 불균형 : f1 score로 모델 평가 
# print(classification_report(df_test["BAD"], y_pred_test_class, digits=3))
print(classification_report(df_test["scale"], y_pred_test_class, target_names=['승인', '거절'], digits=3))


# In[54]:


# 설명변수 중요도
df_logistic_coef = pd.DataFrame({"Coef": log_result.params.values[1:]}, index = log_model.exog_names[1:])
df_logistic_coef.plot.barh(y = "Coef")
# df_logistic_coef.plot.barh(y = "Coef", figsize=(10,6))


# In[55]:


# 0번째=Intercept..
print(log_result.params.values[0:1])
print(log_model.exog_names[0:1])


# In[57]:


# select_dtypes: 특정 변수 타입을 선택/제외하여 데이터 추출
df_char = df.select_dtypes(include = "object")
df_numeric = df.select_dtypes(exclude = "object")

# Data Scaling
scaler = StandardScaler()
np_numeric_scaled = scaler.fit_transform(df_numeric)
df_numeric_scaled = pd.DataFrame(np_numeric_scaled, columns = df_numeric.columns)

# 문자 데이터 + 숫자 데이터
df_scaled = pd.concat([df_numeric_scaled, df_char],axis = 1)
df_scaled.head()


# In[58]:


# BAD 데이터를 0과 1로 변환(정수형), np.where(조건, 조건을 만족하는 경우, 만족하지 않는 경우)
df_scaled["scale"] = np.where(df_scaled["scale"]> 0, 1, 0)
df_scaled.head()


# In[59]:


# 데이터 분할
df_scaled_train, df_scaled_test = train_test_split(df_scaled, # 데이터
                               test_size = 0.3, # test 데이터의 비율
                               random_state = 1234)  # random state


# In[60]:


# from_formula 함수를 이용하여 변수 역할 지정
# scaled_log_model = Logit.from_formula("""BAD ~ LOAN + MORTDUE + VALUE + C(REASON) + C(JOB) + YOJ + 
#         DEROG + DELINQ + CLAGE + NINQ + CLNO + DEBTINC""", df_scaled_train)

# 선정된 설명변수 기준
scaled_log_model = Logit.from_formula("""scale ~ rolling_temp + descaling_count + fur_heat_temp + C(fur_no)""", df_scaled_train)
# 적합
scaled_log_result = scaled_log_model.fit()
# 결과 출력
print(scaled_log_result.summary())


# In[63]:


# 설명변수 중요도
df_log_scaled_coef = pd.DataFrame({"Coef": scaled_log_result.params.values[1:]}, index = scaled_log_model.exog_names[1:])
df_log_scaled_coef.plot.barh(y = "Coef", legend = False)


# In[62]:


import matplotlib as mpl
import matplotlib.pyplot as plt
# 유니코드 깨짐현상 해결
mpl.rcParams['axes.unicode_minus'] = False
# 나눔고딕 폰트 적용
plt.rcParams["font.family"] = 'NanumGothic'


# In[ ]:




