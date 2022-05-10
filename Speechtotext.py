import streamlit as st
from translator import translator1
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource 
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
apikey = ''
url = 'https://api.us-east.speech-to-text.watson.cloud.ibm.com/instances/d4a60e7a-3e07-4b3d-b389-1b237a0a8840'
authenticator = IAMAuthenticator(apikey)
stt = SpeechToTextV1(authenticator=authenticator)
stt.set_service_url(url)
def convert():
    with open("./samples/output.wav", 'rb') as f:
        res = stt.recognize(audio=f, content_type='audio/wav', model='hi-IN_Telephony').get_result()
    text = res['results'][0]['alternatives'][0]['transcript']
    print(text)
    html_str = f"""
    <style>
    p.a {{
    font: bold {30}px Courier;
    }}
    </style>
    <p class="a">{text}</p>
    """
    st.markdown(html_str, unsafe_allow_html=True)
    
    text1=translator1(text)
    html_str = f"""
    <style>
    p.a {{
    font: bold {30}px Courier;
    }}
    </style>
    <p class="a">{text1}</p>
    """
    st.markdown(html_str, unsafe_allow_html=True)
    
    return text1