from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import re
from scipy import signal, stats
import math


#init
layout_black = False #initial background
whether_render = False
wether_title_judge = True
option_graph = ['Raw','Waveband']
spectre= ['Delta','Theta','Alpha','Beta']


mind_raw = False

st.set_page_config(layout="wide")

markdown = """
# ぶしおAnalytics (ver1.0)

"""

st.markdown(markdown)

if "df" not in st.session_state:#0番台
  st.session_state.df = [0]*20
if "df_2" not in st.session_state:#3000番台
  st.session_state.df_2 = [0]*20

if "df_stats" not in st.session_state:#1000番台
  st.session_state.df_stats = [0]*20

if "df_polymate_stats" not in st.session_state:#2000番台
  st.session_state.df_polymate_stats = [0]*20

df_r = [0]*20 #df for render

FigureTitles = [0]*20
sum_df_stats = 1



#---AboutPandas

def judge_csv(raw):#Judge What File Imported
#return   0:None 1:Sirius 2:Polymate 3:Error(idontknow)

  if(raw != None):
    if(re.search('mind_raw',raw.name)):
      return 1
    elif(re.search('Raw',raw.name)):
      return 1
    elif(re.search('CSV',raw.name)):
      return 2
    else:
      return 3
  return 0




#---AboutFFT
def bp_filter(input_sig: np.ndarray, smpl_freq: int = 600, hpf_freq: int = 3, lpf_freq: int = 60, bef_freq: int = 50, qval: float = 2.0, order: int = 2) -> np.ndarray:
    """
    Band-pass Filter
    Input
        - input_sig: Raw EEG Signal [μV]
        - smpl_freq: Original Sampling Rate [Hz]
        - hpf_freq: Lower cutoff frequency [Hz] (optional)
        - lpf_freq: Upper cutoff frequency [Hz] (optional)
        - bef_freq: Notch Filter frequency [Hz] (optional)
        - qval: Quality factor (optional)
        - order: The order of the filter (optional)
    Output:
        - output_sig: Filtered Signal [μV]
    """
    bh, ah = signal.butter(order, hpf_freq, 'high', fs=smpl_freq)
    bl, al = signal.butter(order, lpf_freq, 'low', fs=smpl_freq)
    bn, an = signal.iirnotch(bef_freq,qval, fs=smpl_freq)
    output_sig = signal.lfilter(bh, ah, input_sig)
    output_sig = signal.lfilter(bl, al, output_sig)
    output_sig = signal.lfilter(bn, an, output_sig)
    return output_sig

def nextpow2(n: int) -> int:
# Finding the next Power of 2

  m_f = np.log2(n)
  m_i = np.ceil(m_f)

  return int(np.log2(2**m_i))

def power_spectrum(sig: np.ndarray, fs: int, win_sec: int = 3, fq_lim: int = 40) -> np.ndarray:
# Spectrum Analysis (Short-time Fourier Transform & Power Spectrum)
# Input:
#  - sig: EEG Signal [μV]
#  - fs: Sampling Rate [Hz]
#  - win_sec: Short-time Fourier Transformation Window Size [sec]
#  - fq_lim: Upper Limitation Frequency [Hz]
# Output:
#  - spctgram: Spectrogram
#  - freq: Frequency Label
#  - t: Timeseries Label

    # Parameter Setting
    dt = 1. / fs # Sampling Interval [sec]
    n  = fs * win_sec # Data Length [points]
    nfft = 2 ** nextpow2(n) # next Power of 2
    df   = 1. / (nfft*dt) # Frequency Interval [/sec]​
    # Spectrogram (Short-time Fourier Transformation)
    freq, t, spctgram = signal.spectrogram(sig, fs=fs, nperseg=n, nfft=nfft, noverlap=(win_sec-1)*fs, window='hamming', mode='psd', return_onesided=True)
    freq = freq[:math.ceil(fq_lim/df)]
    spctgram = np.abs(spctgram[:math.ceil(fq_lim/df),:])
    #db = 10 * np.log10(spctgram)

    return spctgram, freq, t










#---AboutSidebar
st.sidebar.title('Settings')
graph_h = st.sidebar.slider('Graph height',1000,2000,1600)
graph_w = st.sidebar.slider('Graph width',1000,2000,1400)

#makeSomeday
n = st.sidebar.number_input(label='How many graphes?',#Count CreateGraph Num
                        min_value=1,
						            value=1
                        )


#あとで可変長個のグラフに対応したいけど、fileuploaderのkeys問題が解決しないのでとりあえず10個まで対応しとく
#st.write("dfの中身",st.session_state.df)

def inport_file_widget(num):

  st.subheader("%dファイル目:" %(num+1))
  st.session_state.df[num] = st.file_uploader("", type='csv',key = num)
  if(judge_csv(st.session_state.df[num]) == 1):
    st.write("インポートされたデータ:Siriusデータ")
  elif(judge_csv(st.session_state.df[num]) == 2):
    st.write("インポートされたデータ:Polymateデータ")
    tmp_df =pd.read_csv(st.session_state.df[num],skiprows = 3,encoding = 'UTF8')
    st.session_state.df_polymate_stats[num] = st.selectbox(label="表示したい列を選択してください",options=tmp_df.columns,key = 2000 + num)
    st.session_state.df_2[num] = tmp_df
  elif(judge_csv(st.session_state.df[num]) == 3):
    st.write("インポートされたデータ:登録されていないファイル形式")
  else:
    st.write("インポートされたデータがありません")
  st.session_state.df_stats[num] = st.multiselect(label="表示したい内容を選択してください",options=option_graph,key = 1000 + num)
 
  st.write("")
  st.markdown("---")


inport_file_widget(0)
inport_file_widget(1)
inport_file_widget(2)
inport_file_widget(3)
inport_file_widget(4)
inport_file_widget(5)
inport_file_widget(6)
inport_file_widget(7)
inport_file_widget(8)
inport_file_widget(9)


#---AboutPlotly
def plot_graph_init(n):#init graph
	fig = make_subplots(
	rows=sum_df_stats, cols=1,
	#horizontal_spacing=0.05,
	vertical_spacing=0.02,
	#row_heights=[0.01,0.01,0.01,0.01],
	# column_widths=[0.001],
  subplot_titles=(FigureTitles),
	shared_xaxes=True,
  )
	return fig

def plot_graph(fig,df,Name,i,df_stats_list,file_type,polymate_stats):#Add graph

  if(file_type == 0):
    if(mind_raw == True):
      df_colum = 'brainWave'
    else:
      df_colum = 'RawDataEdit_uV'
    fig.add_trace(go.Scatter
    (
      y= df[df_colum],
      name = "SiriusRaw",
      marker=dict()
    )  
    , row=i, col=1)

  if(file_type == 1):
    fs = 600
    if(mind_raw == True):
      df_colum = 'brainWave'
    else:
      df_colum = 'RawDataEdit_uV'
    
    sig = np.zeros((df[df_colum].values.shape[0], 1))
    sig[:,0] = df[df_colum].values
    win_sec = 3
    fq_lim = 60
    dt = 1./fs # Sampling Interval [sec]
    n  = win_sec*fs # Data Length [points]
    nfft = 2 ** nextpow2(n) # next Power of 2
    dfq   = 1./(nfft*dt) # Frequency Interval [/sec]
    dplt = round(5/dfq)
    n_conv = 30
    spctgram, freq, t = power_spectrum(sig[:,0], fs, win_sec, fq_lim)
    idx_fq = np.array(np.round(freq, 2), dtype=str)
    delta = np.abs(np.mean(spctgram[math.ceil(0/dfq):math.floor(4/dfq),:],axis=0))
    theta = np.abs(np.mean(spctgram[math.ceil(4/dfq):math.floor(8/dfq),:],axis=0))
    low_alpha = np.abs(np.mean(spctgram[math.ceil(8/dfq):math.floor(10/dfq),:],axis=0))
    high_alpha = np.abs(np.mean(spctgram[math.ceil(10/dfq):math.floor(12/dfq),:],axis=0))
    smr = np.abs(np.mean(spctgram[math.ceil(12/dfq):math.floor(15/dfq),:],axis=0))
    med_beta = np.abs(np.mean(spctgram[math.ceil(15/dfq):math.floor(20/dfq),:],axis=0))
    high_beta = np.abs(np.mean(spctgram[math.ceil(20/dfq):math.floor(30/dfq),:],axis=0))
    low_gamma = np.abs(np.mean(spctgram[math.ceil(30/dfq):math.floor(40/dfq),:],axis=0))
    med_gamma = np.abs(np.mean(spctgram[math.ceil(40/dfq):math.floor(50/dfq),:],axis=0))
    columns_wave=['delta' ,'theta', 'low_alpha', 'high_alpha', "smr", 'med_beta', 'high_beta', 'low_gamma', 'med_gamma']
    #df_waveband = pd.DataFrame(np.array([delta, theta, low_alpha, high_alpha, smr, med_beta, high_beta, low_gamma, med_gamma]).T, columns=columns_wave)
    alpha = np.abs(np.mean(spctgram[math.ceil(8/dfq):math.floor(12/dfq),:],axis=0))
    beta = np.abs(np.mean(spctgram[math.ceil(12/dfq):math.floor(30/dfq),:],axis=0))
    labels = ["Delta", "Theta", "Alpha", "Beta"]
    df_waveband = pd.DataFrame(np.array([delta, theta, alpha,beta]).T, columns=labels)
    fftspectre = [delta,theta,alpha,beta]
    #x_sirius = np.zeros((df["RawDataEdit_uV"].values.shape[0], 1))*600
    x_sirius = np.arange(len(df[[df_colum]]))*600
    for k in range(4):
      fig.add_trace(go.Scatter
      (
        x = x_sirius,
        y= fftspectre[k],
        name = spectre[k],
        marker=dict()
        
      )  , row=i, col=1)

  if(file_type == 2):
    x_polymate = (np.arange(len(df[polymate_stats]))*1.2)
    fig.add_trace(go.Scatter
    (
      x=x_polymate,
      y= df[polymate_stats],
      name = Name,
      marker=dict()
    )  , row=i, col=1)

  if(file_type == 3):
    fs = 500
    sig = np.zeros((df[polymate_stats].values.shape[0], 1))
    sig[:,0] = df[polymate_stats].values
    win_sec = 3
    fq_lim = 60
    dt = 1./fs # Sampling Interval [sec]
    n  = win_sec*fs # Data Length [points]
    nfft = 2 ** nextpow2(n) # next Power of 2
    dfq   = 1./(nfft*dt) # Frequency Interval [/sec]
    dplt = round(5/dfq)
    n_conv = 30
    spctgram, freq, t = power_spectrum(sig[:,0], fs, win_sec, fq_lim)
    idx_fq = np.array(np.round(freq, 2), dtype=str)
    delta = np.abs(np.mean(spctgram[math.ceil(0/dfq):math.floor(4/dfq),:],axis=0))
    theta = np.abs(np.mean(spctgram[math.ceil(4/dfq):math.floor(8/dfq),:],axis=0))
    low_alpha = np.abs(np.mean(spctgram[math.ceil(8/dfq):math.floor(10/dfq),:],axis=0))
    high_alpha = np.abs(np.mean(spctgram[math.ceil(10/dfq):math.floor(12/dfq),:],axis=0))
    smr = np.abs(np.mean(spctgram[math.ceil(12/dfq):math.floor(15/dfq),:],axis=0))
    med_beta = np.abs(np.mean(spctgram[math.ceil(15/dfq):math.floor(20/dfq),:],axis=0))
    high_beta = np.abs(np.mean(spctgram[math.ceil(20/dfq):math.floor(30/dfq),:],axis=0))
    low_gamma = np.abs(np.mean(spctgram[math.ceil(30/dfq):math.floor(40/dfq),:],axis=0))
    med_gamma = np.abs(np.mean(spctgram[math.ceil(40/dfq):math.floor(50/dfq),:],axis=0))
    columns_wave=['delta' ,'theta', 'low_alpha', 'high_alpha', "smr", 'med_beta', 'high_beta', 'low_gamma', 'med_gamma']
    #df_waveband = pd.DataFrame(np.array([delta, theta, low_alpha, high_alpha, smr, med_beta, high_beta, low_gamma, med_gamma]).T, columns=columns_wave)
    alpha = np.abs(np.mean(spctgram[math.ceil(8/dfq):math.floor(12/dfq),:],axis=0))
    beta = np.abs(np.mean(spctgram[math.ceil(12/dfq):math.floor(30/dfq),:],axis=0))
    labels = ["Delta", "Theta", "Alpha", "Beta"]
    df_waveband = pd.DataFrame(np.array([delta, theta, alpha,beta]).T, columns=labels)
    fftspectre = [delta,theta,alpha,beta]
    x_polymate = (np.arange(len(df[["POINT"]]))*1.2)*500
    for k in range(4):
        fig.add_trace(go.Scatter
        (
          x=x_polymate,
          y= fftspectre[k],
          name = spectre[k]+ Name,
          marker=dict()
          
        )  , row=i, col=1)


#---AboutUI
def plot_layout(fig):#Update Layout for Black
  if(layout_black == True):
    fig.update_layout(
            
            template='plotly_dark',margin=dict(t=60,b=40),
            title=dict(font=dict(size=20,color='white'),yanchor='top'),
          	barmode='group',
            width=graph_w,
      			height=graph_h,
            hovermode='closest'
                  )
  if(layout_black == False):
    fig.update_layout(
              width=graph_w,
    			height=graph_h,
          hovermode='closest'
                )
  return fig




if(st.sidebar.checkbox('Black Background')):#background checkbox
	layout_black = True

if(st.sidebar.checkbox('brain wave mode')):#background checkbox
	mind_raw = True
if(st.sidebar.button('Render')):#RenderButton
  whether_render = True


for ij in range(n):
  sum_df_stats += len(st.session_state.df_stats[ij])




#---AboutPlottingMove
if(whether_render == True):
  
  howmanyplots = 0
  for i in range(n): #judgeTitle
    if(st.session_state.df[i] != None):
      
      if(judge_csv(st.session_state.df[i]) == 1):
        if("Raw" in str(st.session_state.df_stats[i])):
          FigureTitles[howmanyplots] = f"Fig{howmanyplots}:Sirius Raw"
          howmanyplots += 1
        if("Waveband" in str(st.session_state.df_stats[i])):
          FigureTitles[howmanyplots] = f"Fig{howmanyplots}:Sirius Waveband"
          howmanyplots += 1

      elif(judge_csv(st.session_state.df[i]) == 2):
        if("Raw" in str(st.session_state.df_stats[i])):
          FigureTitles[howmanyplots] = f"Fig{howmanyplots}:Polymate Raw" + st.session_state.df_polymate_stats[i]
          howmanyplots += 1
        if("Waveband" in str(st.session_state.df_stats[i])):
          FigureTitles[howmanyplots] = f'Fig{howmanyplots}:Polymate Waveband'+ st.session_state.df_polymate_stats[i]
          howmanyplots += 1

  howmanyplots = 0
  i=0
  fig = plot_graph_init(n)
  for i in range(n):
    if(st.session_state.df[i] != None):
      
      if(judge_csv(st.session_state.df[i]) == 1):
        df_r[i] = pd.read_csv(st.session_state.df[i],encoding = 'UTF8')
        if("Raw" in str(st.session_state.df_stats[i])):
          plot_graph(fig,df_r[i],'SiriusRaw',1+howmanyplots,st.session_state.df_stats[i],0,0)
          howmanyplots += 1
        if("Waveband" in str(st.session_state.df_stats[i])):
          plot_graph(fig,df_r[i],'None',1+howmanyplots,st.session_state.df_stats[i],1,0)
          howmanyplots += 1
      elif(judge_csv(st.session_state.df[i]) == 2):
        df_r[i] =st.session_state.df_2[i]
        if("Raw" in str(st.session_state.df_stats[i])):
          plot_graph(fig,df_r[i],"Polymate Raw_"+ st.session_state.df_polymate_stats[i],1+howmanyplots,st.session_state.df_stats[i],2,st.session_state.df_polymate_stats[i])
          howmanyplots += 1
        if("Waveband" in str(st.session_state.df_stats[i])):
          plot_graph(fig,df_r[i],str(st.session_state.df_polymate_stats[i]),1+howmanyplots,st.session_state.df_stats[i],3,st.session_state.df_polymate_stats[i])
          howmanyplots += 1
  
  st.write(FigureTitles)
  plot_layout(fig)
  st.plotly_chart(fig, use_container_width=True)


#---test
#st.write(st.session_state.df_stats[:len(st.session_state.df_stats[0])])

#st.write("Raw" in st.session_state.df_stats[0])

