
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(layout="wide")
st.title('データ分析アプリ')

st.sidebar.write('Settings')
graph_h = st.sidebar.slider('height',1,2000,1600)
graph_w = st.sidebar.slider('height',1,2000,1400)


st.write('SiriusRAW')
SiriusRaw = st.file_uploader("ファイルアップロード", type='csv',key="1")

st.write('SiriusWaveband')
WaveBand = st.file_uploader("ファイルアップロード", type='csv',key="2")


df = pd.read_csv(SiriusRaw,encoding = 'UTF8')
df_w = pd.read_csv(WaveBand,encoding = 'UTF8')

Title = ['Delta','Theta','Alpha','Beta']
titles = [' EMG',' EMG.1',' EMG.2',' EOG',' EOG.1',' 6-REF']
df_w['Unnamed: 0'] = df_w['Unnamed: 0']*600

#x_600 = np.arange(len(df['Time']))/600



fig = make_subplots(
  rows=6, cols=1,
  #horizontal_spacing=0.05,
  vertical_spacing=0.04,
subplot_titles=("SiriusRaw", "Waveband", "Delta", "Theta", "Alpha", "Beta"),
    #row_heights=[0.01,0.01,0.01,0.01],
  # column_widths=[0.001],
  shared_xaxes=True,
  
  )


fig.add_trace(go.Scatter#描画
  (
    y= df['RawDataEdit_uV'],
    name = 'RawDataEdit_uV',
    marker=dict()
    
  )  , row=1, col=1) 

for i in range(4):
  fig.add_trace(go.Scatter#描画
    (
      x=df_w['Unnamed: 0'],
      y= df_w[Title[i]],
      name = 'Waveband_' + Title[i],
      marker=dict()
    )  , row=2, col=1) 


fig.update_layout(template='plotly_dark',margin=dict(t=60,b=40),
                  title=dict(font=dict(size=20,color='white'),yanchor='top'),
                  barmode='group')


####

fillkind = 'tonexty'

df = [[]]*8

for i in range(8):
  df[i] = pd.read_csv('/Users/msys/Google Drive/共有ドライブ/52_開発(共有Drive)/5211_実験/活性度実験（オフィスの集中筋電確認）Work_APP_20220705-/20220713-14_APM早打ちテスト（ポリメイトとSiriusで）/PolymateData/Arikawa/analysis/waveband_power_'+ str(i) + '.csv',encoding = 'UTF8')
  df[i]['Unnamed: 0'] = df[i]['Unnamed: 0']*600
Titles = ["Delta","Theta","Alpha","Beta"]
EEG = ['Fpz','T8','O1','O2']


for j in range(4):# j : 周波数帯域別(それぞれ別グラフ)
  for k in range(4): #k : チャンネル別(それぞれ同じグラフ)
    if k == 0:
      case0 = df[k+4][Titles[j]] / (df[4][Titles[j]] + df[5][Titles[j]] + df[6][Titles[j]]+ df[7][Titles[j]])
      fig.add_trace(go.Scatter#描画
        (
          x = df[j]['Unnamed: 0'],
          y= case0,
          name = Titles[j] + EEG[k],
          fill=fillkind,
          marker=dict()
        )  , row=int(j+3), col=1) 
    elif k == 1:
      case1 = df[k+4][Titles[j]]/ (df[4][Titles[j]] + df[5][Titles[j]] + df[6][Titles[j]]+ df[7][Titles[j]])
      fig.add_trace(go.Scatter#描画
        (
          x = df[j]['Unnamed: 0'],
          y= case1 + case0,
          name = Titles[j] + EEG[k],
          fill=fillkind,
          marker=dict()
        )  , row=int(j+3), col=1) 
    elif k == 2:
      case2 = df[k+4][Titles[j]] / (df[4][Titles[j]] + df[5][Titles[j]] + df[6][Titles[j]]+ df[7][Titles[j]])
      fig.add_trace(go.Scatter#描画
        (
          x = df[j]['Unnamed: 0'],
          y= case2 + case1 + case0,
          name = Titles[j] + EEG[k],
          fill=fillkind,
          marker=dict()
        )  , row=int(j+3), col=1) 
    elif k == 3:
      case3 = df[k+4][Titles[j]] / (df[4][Titles[j]] + df[5][Titles[j]] + df[6][Titles[j]]+ df[7][Titles[j]])
      fig.add_trace(go.Scatter#描画
        (
          x = df[j]['Unnamed: 0'],
          y= case0 + case1 + case2 + case3,
          name = Titles[j] + EEG[k],
          fill=fillkind,
          marker=dict()
        )  , row=int(j+3), col=1) 
      
fig.update_layout(template='plotly_dark',margin=dict(t=60,b=40),
                  title=dict(font=dict(size=20,color='white'),yanchor='top'),
                  barmode='group',
                        width=graph_w,
      height=graph_h,
                  )

fig.show()
