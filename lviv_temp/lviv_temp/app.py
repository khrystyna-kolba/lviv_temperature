from neuralprophet import NeuralProphet
import gradio as gr
import pickle
import pandas as pd
def greet(name):
    return "Hello " + name + "!!"
def predict(date):
    n = pd.read_pickle("fore.pkl")
    date= pd.to_datetime(date)
    date1=date + pd.DateOffset(1)
    k = pd.DataFrame({"ds": [date,date1], "y": [None, None]})
    fore = n.predict(k)
    #print(fore)
    return "{} C degrees".format(round(fore.iloc[0]["yhat1"], 2))


#print(predict("07-07-2100"))
iface = gr.Interface(fn=predict, inputs="text", outputs="text")
iface.launch()
