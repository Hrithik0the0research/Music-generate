import streamlit as st
import random 
import numpy as np
from music_read import read_midi,convert_to_midi
from sklearn.model_selection import train_test_split
from keras.models import load_model
import  matplotlib.pyplot as plt
from glob import glob
from collections import Counter
import os
ran_number=random.randint(1,10000)
file_delete=glob("./audio/*")
for i in file_delete:
    os.remove(i)
def predict_stacked_model(model, inputX):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # make prediction
    return model.predict(X, verbose=0)
option = st.selectbox(
    'Models',
    ('','LSTM', 'RNN', 'CNN','Hybrid','STACKED'))
if option=="LSTM":
    model=load_model("lstm_model.h5")
elif option=="RNN":
    model=load_model("rnn_model.h5")
elif option=="CNN":
    model=load_model("cn_model.h5")
elif option=="Hybrid":
    model=load_model("hybrid_model.h5")
else:
    model=load_model("stacked.h5")
st.write("music generate")

st.write("Download demo files")
demo=glob("./audio1/*")

demo_file = st.selectbox("download demo audio for using it as prediction",demo)
print(demo_file)
with open(demo_file, "rb") as file:
        btn = st.download_button(
                label="Download audio",
                data=file,
                file_name="demo.midi")

uploaded_file = st.file_uploader("Choose a file")

val = st.text_input('how much part you want to predict', '10%')
val1=""
for j in val:
    if j!="%":
        val1=val1+j
val=int(val1)
print(val)
st.write("It takes some minutes")
if uploaded_file is not None:
    
    for j in range(3):
        audio_name='./audio/audio'+str(ran_number)+str(j)+'.midi'
        with open(audio_name, mode='bx') as f:
            f.write(uploaded_file.getvalue())
    backup_name='./audio1/audio'+str(ran_number)+'.midi'
    with open(backup_name, mode='bx') as f:
        f.write(uploaded_file.getvalue())
    file=glob("./audio/*")
    print(file)
    #files=[i for i in file if i.endswith(".mid")]
    #print(files)
    notes_array = np.array([read_midi(i) for i in file])
    print(notes_array)
    #print(notes_array)
    #print(notes_array)
    notes_ = [element for note_ in notes_array for element in note_]

#No. of unique notes
    unique_notes = list(set(notes_))
    notes_ = [element for note_ in notes_array for element in note_]

#No. of unique notes
    unique_notes1 = list(set(notes_))
    print(len(unique_notes1))
    freq = dict(Counter(notes_))

#library for visualiation


#consider only the frequencies
    no=[count for _,count in freq.items()]
    frequent_notes = [note_ for note_, count in freq.items() if count>=20]
    new_music=[]

    for notes in notes_array:
        temp=[]
        for note_ in notes:
            if note_ in frequent_notes:
                temp.append(note_)            
        new_music.append(temp)
        
    new_music = np.array(new_music)
    print(new_music.shape)
    no_of_timesteps = 30
    x = []
    y = []

    for note_ in new_music:
        for i in range(0, len(note_) - no_of_timesteps, 1):
            
            #preparing input and output sequences
            input_ = note_[i:i + no_of_timesteps]
            output = note_[i + no_of_timesteps]
            
            x.append(input_)
            y.append(output)
            
    x=np.array(x)
    y=np.array(y)

    unique_x = list(set(x.ravel()))
    x_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_x))
    x_seq=[]
    for i in x:
        temp=[]
        for j in i:
            #assigning unique integer to every note
            temp.append(x_note_to_int[j])
        x_seq.append(temp)
        
    x_seq = np.array(x_seq)
    unique_y = list(set(y))
    y_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_y)) 
    y_seq=np.array([y_note_to_int[i] for i in y])
    x=x_seq
    y=y_seq
    x=np.array(x)
    y=np.array(y)
    x=x.reshape(x.shape[0],x.shape[1],1)
    print(x.shape)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)
    print(x_test.shape)
    import random
    ind = np.random.randint(0,len(x_test)-1)

    random_music = x_test[ind]

    predictions=[]
    
    p=st.empty()
    if option=="STACKED":
        for i in range(len(x_test)):
            p.write("wait for some moments....")
            p.write("wait for some moments..")
            p.write("wait for some moments......")
            p.write("wait for some moments.")
            random_music = random_music.reshape(1,no_of_timesteps)
            print(random_music.shape)
            prob=predict_stacked_model(model,random_music)[0]
            y_pred= np.argmax(prob,axis=0)
            predictions.append(y_pred)
            p.write("wait for some moments..")
            random_music = np.insert(random_music[0],len(random_music[0]),y_pred)
            random_music = random_music[1:]
    else:
        for i in range(len(x_test)):
            p.write("wait for some moments....")
            p.write("wait for some moments..")
            p.write("wait for some moments......")
            p.write("wait for some moments.")
            random_music = random_music.reshape(1,no_of_timesteps)
            print(random_music.shape)
            prob  = model.predict(random_music)[0]
            y_pred= np.argmax(prob,axis=0)
            predictions.append(y_pred)
            p.write("wait for some moments..")
            random_music = np.insert(random_music[0],len(random_music[0]),y_pred)
            random_music = random_music[1:]
        
    plt.plot(y_test,label="y_test",color="red")
    plt.plot(predictions,label="predicted",color="blue")
    plt.legend()
    plt.savefig("plot.png")
    st.image("plot.png")
    x_int_to_note = dict((number, note_) for number, note_ in enumerate(unique_x)) 
    #predicted_notes = [x_int_to_note[i] for i in predictions]
    predicted_notes=[]
    for i in predictions:
        if i in x_int_to_note.keys():
            predicted_notes.append(x_int_to_note[i])
    #print(predictions)
    convert_to_midi(predicted_notes)
    with open("new_music.mid", "rb") as file:
        btn = st.download_button(
                label="Download audio",
                data=file,
                file_name="music.mid",
            
                )
   # for f in file:
    #    os.remove(f)
    
    file_delete=glob("./audio/*")
    for i in file_delete:
        os.remove(i)
