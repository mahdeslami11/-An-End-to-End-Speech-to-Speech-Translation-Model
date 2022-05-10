
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import SessionState
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
session_state=SessionState.get(s="")
def translator1(en_text):
  session_state.s=""
  tokenizer.src_lang = "hi"
  
  encoded_en = tokenizer(en_text, return_tensors="pt")
  generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.get_lang_id("en"))
  list =[]
  list=tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
  
  for i in range(0,len(list)):
    session_state.s=session_state.s+list[i]+" "
  print(session_state.s)  
  return session_state.s

"""
trans(en_text)
"""
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from text_to_speech import speech
#loading the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")

def translator1(text):
  # function to translate english text to hindi
  input_ids = tokenizer.encode(text, return_tensors="pt", padding=True)
  outputs = model.generate(input_ids)
  decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
  speech(decoded_text)
  #return decoded_text
#text you want translate
#text = "Dont hesitate to ask questions"
#print("Hindi Translation: ", translator(text))
"""