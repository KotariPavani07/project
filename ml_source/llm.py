import google.generativeai as genai

genai.configure(api_key="AIzaSyCm5aAY9Rw7R089qufDCAxXe7jLJaOTk2s")

model=genai.GenerativeModel(model_name="gemini-2.0-flash")

chat=model.start_chat(history=[])
while True:
    prmt=input()
    if(prmt=="exit"):
        break


    res=chat.send_message(prmt)
    print(res.text)
