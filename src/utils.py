import tkinter as tk
from tkinter import Canvas, PhotoImage
from gtts import gTTS
from playsound import playsound
import pyperclip as clipboard
from pathlib import Path
from os import remove
from spellchecker import SpellChecker

OUTPUT_PATH = Path(__file__).parent.parent
ASSETS_PATH = OUTPUT_PATH / Path("./assets/frame0")

def relative_to_assets(path: str) -> Path:
    return str(ASSETS_PATH / Path(path)).replace("\\", "/")

def show_toast(message, x, y, automatic_delete: bool = True):
    toast = tk.Toplevel()
    toast.geometry(f"300x100+{x}+{y}")
    
    toast.configure(bg = "#FFFFFF")
    
    canvasToast = Canvas(
        toast,
        bg = "#FFFFFF",
        height = 100,
        width = 300,
        bd = 0,
        highlightthickness = 0,
        relief = "ridge"
    )

    canvasToast.place(x = 0, y = 0)
    canvasToast.create_rectangle(
        0.0,
        0.0,
        300.0,
        100.0,
        fill="#FFFFFF",
        outline="")

    canvasToast.create_rectangle(
        8.0,
        7.0,
        18.0,
        93.0,
        fill="#4ECB71",
        outline="")

    image_image_1 = PhotoImage(
        file=relative_to_assets("yes.png"))
    
    image_1 = canvasToast.create_image(
        50.0,
        50.0,
        image=image_image_1
    )

    canvasToast.create_text(
        45.0,
        43.0,
        anchor="nw",
        text=message,
        fill="#000000",
        font=("Helvetica", 14 * -1, 'bold')
    )
    
    if automatic_delete == True:
        toast.after(1700, toast.destroy) # Cierra la ventana después de 3 segundos

def copy_text(sent_entrybox):
    clipboard.copy(sent_entrybox.get("1.0", "end"))
    show_toast("Texto copiado en el portapapeles", "370", "340")

def delete_text(sent_entrybox):
    sent_entrybox.delete("1.0", "end")
    show_toast("Se ha eliminado el texto" , "370", "340")
    
def eliminate_last_word(sent_entrybox):
    contenido = sent_entrybox.get('1.0', 'end-1c')
    palabras = contenido.split()
    if palabras:
        palabras.pop()
        nuevo_contenido = " ".join(palabras)
        sent_entrybox.delete('1.0', 'end')
        sent_entrybox.insert('1.0', nuevo_contenido+" ")
        show_toast("Se elimino la ultima palabra", "370", "340")
    
def text_to_speech(sent_entrybox):
    show_toast("Reproduciendo audio...", "370", "340")
    
    FILE_NAME = "sonido_generado.mp3"
    FILE_PATH = f"./audios_generados/{FILE_NAME}"
    
    tts = gTTS(sent_entrybox.get("1.0", "end"), lang='es-us')
    
    with open(FILE_PATH, "wb") as archivo:
        tts.write_to_fp(archivo)
    
    playsound(FILE_PATH)
    
    remove(FILE_PATH)

def auto_correct(word):
    spell = SpellChecker(language='es')
    correction =  spell.correction(word)
    
    if correction != None:
        return correction
    return "a"
