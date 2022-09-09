#import gradio as gr
import pickle
#from neuralprophet import NeuralProphet
import pandas as pd
n = pd.read_pickle("fore.pkl")
date= pd.to_datetime("07-06-2200")+ pd.DateOffset(1)
#date1=date[:3]+str(int(date[3:5])+1)+date[5:]
k = pd.DataFrame({"ds": [pd.to_datetime("07-06-2200"),pd.to_datetime("07-07-2200")], "y": [None, None]})
fore = n.predict(k)
print(fore.iloc[0]["yhat1"])
print(date)
print("done")
