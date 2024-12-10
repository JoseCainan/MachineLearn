import speech_recognition as sr
import pyttsx3
import datetime
import wikipedia
import pywhatkit
from mensagens import sms_chess, apresentaçao, ia_adaptativa, ia_compositiva, tarefas_operacionais
import webbrowser

audio = sr.Recognizer()
machine = pyttsx3.init()

def listar_vozes():
    voices = machine.getProperty('voices')
    print("Vozes disponíveis:")
    for index, voice in enumerate(voices):
        print(f"{index}: {voice.name} - {voice.languages}")

def mudar_para_voz_feminina():
    voices = machine.getProperty('voices')
    for voice in voices:
        if "Microsoft Maria" in voice.name:
            machine.setProperty('voice', voice.id)
            return
    machine.say("Desculpe, não encontrei uma voz feminina disponível.")
    machine.runAndWait()

listar_vozes()
mudar_para_voz_feminina()

def execução_comand():
    try:
        with sr.Microphone() as source:
            print("Aguardando seu comando...")  
            audio.adjust_for_ambient_noise(source)  
            my_voice = audio.listen(source, timeout=None)
            comand = audio.recognize_google(my_voice, language='pt-BR')
            comand = comand.lower()  

            if 'ellen' in comand:
                comand = comand.replace("ellen", "") 

            return comand

    except sr.UnknownValueError:
        print("Desculpe, não consegui entender o que você disse.")
        return None

    except sr.RequestError:
        print("Erro ao se comunicar com o serviço de reconhecimento de fala.")
        return None

    except:
        print("falha no microfone")
        return None

def funcao_usuario():
    comand = execução_comand()

    if comand is None:
        return True

    if 'horas' in comand:
        hora = datetime.datetime.now().strftime('%H:%M')
        machine.say('Agora são ' + hora)
        machine.runAndWait()
    
    elif "busque por" in comand:
        busca = comand.replace("busque por", "")
        wikipedia.set_lang('pt')
        resultado = wikipedia.summary(busca, 3)
        print(resultado)
        machine.say(resultado)
        machine.runAndWait()
    
    elif "o que é" in comand:
        busca = comand.replace("o que é", "")
        wikipedia.set_lang('pt')
        resultado = wikipedia.summary(busca, 2)
        print(resultado)
        machine.say(resultado)
        machine.runAndWait()

    elif "quem é" in comand:
        busca = comand.replace("quem é", "")
        wikipedia.set_lang('pt')
        resultado = wikipedia.summary(busca, 2)
        print(resultado)
        machine.say(resultado)
        machine.runAndWait()

    elif 'toque' in comand:
        musica = comand.replace('toque', '')
        pywhatkit.playonyt(musica)
        machine.say('tocando música')
        machine.runAndWait()
    
    elif 'abrir lichess' in comand:
        machine.say('Abrindo o Lichess')
        machine.runAndWait()
        webbrowser.open('https://lichess.org')
    
    elif 'bom dia' in comand.strip():  
        machine.say(apresentaçao)
        machine.runAndWait()
    
    elif "defina" in comand:
        machine.say(ia_compositiva)
        machine.runAndWait()
    
    elif "fala aí" in comand:
        machine.say(ia_compositiva)
        machine.runAndWait()

    elif "responde" in comand:
        machine.say(tarefas_operacionais)
        machine.runAndWait()
    
    elif 'encerrar' in comand:
        machine.say("vou dormir então. Até logo!")
        machine.runAndWait()
        return False
    
    else:
        machine.say("Desculpe, não entendi o comando.")
        machine.runAndWait()

    return True

while True:
    if not funcao_usuario():
        break
