
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path
import time

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"D:\Documentos\b\build\assets\frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


ventana = Tk()

ventana.geometry("416x352")
ventana.configure(bg = "#FFFFFF")


canvas = Canvas(
    ventana,
    bg = "#FFFFFF",
    height = 352,
    width = 416,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(
    file=relative_to_assets("advertencia.png"))
image_1 = canvas.create_image(
    208.0,
    53.0,
    image=image_image_1
)

canvas.create_text(
    195.0,
    110.0,
    anchor="nw",
    text="5",
    fill="#000000",
    font=("Inter Bold", 40 * -1),
    tags='contador'
)

image_image_2 = PhotoImage(
    file=relative_to_assets("recomendaciones.png"))
image_2 = canvas.create_image(
    189.0,
    256.0,
    image=image_image_2
)
ventana.resizable(False, False)

for i in range(5, -1, -1):
    canvas.delete('contador')
    canvas.create_text(
    195.0,
    110.0,
    anchor="nw",
    text=f"{i}",
    fill="#000000",
    font=("Inter Bold", 40 * -1),
    tags='contador'
    )
    
    canvas.update()
    
    if i == 0:
        ventana.destroy()
        from interfaz import iniciarInterfazHappyHand
        iniciarInterfazHappyHand()
        
    time.sleep(2)

    


ventana.mainloop()