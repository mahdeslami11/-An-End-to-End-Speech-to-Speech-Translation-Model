import streamlit as st
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import soundfile as sf
import os
import librosa
import glob
from helper import draw_embed, create_spectrogram, read_audio, record, save_record
import SessionState
from Speechtotext import convert
np.set_printoptions(threshold=np.inf)
"# An End-to-End Speech-to-Speech Translation Model"

model_load_state = st.text("Loading pretrained models...")

num_generated = 0
enc_model_fpath = Path("encoder/saved_models/pretrained.pt")
syn_model_dir = Path("synthesizer/saved_models/pretrained/pretrained.pt")
voc_model_fpath = Path("vocoder/saved_models/pretrained/pretrained.pt")
encoder.load_model(enc_model_fpath)
synthesizer = Synthesizer(
    syn_model_dir
)
vocoder.load_model(voc_model_fpath)

model_load_state.text("Loaded pretrained models!")

session_state=SessionState.get(path_myrecording=None,embed=None,path_myoutput=None,text=None,myrecording=None)
st.header("1. Record your own voice for sampling")

st.text("Read the paragraph written below for 20s")
'''
"A computer is a machine that can be programmed to carry out sequences of arithmetic or logical operations automatically. Modern computers can perform generic sets of operations known as programs. These programs enable computers to perform a wide range of tasks. A computer system is a "complete" computer that includes the hardware, operating system (main software), and peripheral equipment needed and used for "full" operation. This term may also refer to a group of computers that are linked and function together, such as a computer network or computer cluster."
'''
try:
    if st.button(f"Click to Record",key=0):
            
            record_state = st.text("Recording...")
            duration = 20 # seconds
            fs = 16000
            session_state.myrecording = record(duration, fs)
            
            record_state.text(f"Saving sample as Record.wav")
            session_state.path_myrecording = f"./samples/Record.wav"
            
            save_record(session_state.path_myrecording, session_state.myrecording, fs)
            record_state.text(f"Done! Saved sample as Record.wav")

            st.audio(read_audio(session_state.path_myrecording))

            fig = create_spectrogram(session_state.path_myrecording)
            st.pyplot(fig)
            
    #original_wav, sampling_rate = librosa.load(session_state.path_myrecording)
    #preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
    session_state.embed = encoder.embed_utterance(
        encoder.preprocess_wav(
            np.array(
                session_state.myrecording).flatten(), 16000))
    #session_state.embed = encoder.embed_utterance(preprocessed_wav)
    st.success("Created the embedding")
    fig = draw_embed(session_state.embed, "myembedding", None)
    st.pyplot(fig)


    st.header("2. Record your voice for translation")

    duration1=st.text_input("Choose time duration for translation")
    duration1=int(duration1)
    if st.button(f"Click to Record",key=1):

        
            record_state1 = st.text("Recording...")
            # seconds
            fs1 = 16000
            myrecording1 = record(duration1, fs1)
            record_state1.text(f"Saving sample as output.wav")
            session_state.path_myoutput = f"./samples/output.wav"

            save_record(session_state.path_myoutput, myrecording1, fs1)
            record_state1.text(f"Done! Saved sample as output.wav")

            st.audio(read_audio(session_state.path_myoutput))

            fig1 = create_spectrogram(session_state.path_myoutput)
            st.pyplot(fig1)
            session_state.text=convert()
    
    
    "## 3. Synthesize text."


    def pgbar(i, seq_len, b_size, gen_rate):
        mybar.progress(i / seq_len)

    
    if st.button("Click to synthesize"):
        texts = [session_state.text]
        embeds = [session_state.embed]

        # generate waveform
        with st.spinner("Generating your speech..."):
            specs = synthesizer.synthesize_spectrograms(texts, embeds)
            spec = specs[0]
            synthesize_state = st.text("Created the mel spectrogram")
            synthesize_state.text("Generating the waveform...")
            mybar = st.progress(0)
            generated_wav = vocoder.infer_waveform(spec, progress_callback=pgbar)
            generated_wav = np.pad(
                generated_wav, (0, synthesizer.sample_rate), mode="constant"
            )
            generated_wav = encoder.preprocess_wav(generated_wav)
            synthesize_state.text("Synthesized the waveform")
            st.success("Done!")
            
        # Save it on the disk
        filename = "demo_output_%02d.wav" % num_generated
        sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
        num_generated += 1
        synthesize_state.text("\nSaved output as %s\n\n" % filename)
        st.audio(read_audio(filename))
except:
    num_generated=num_generated