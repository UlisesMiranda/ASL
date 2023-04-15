import tkinter as tk
import cv2
from PIL import Image, ImageTk

class MyGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Pantalla con tres columnas")

        # Creamos un frame para la primera sección con los botones
        self.frame1 = tk.Frame(self.master)
        self.frame1.grid(row=0, column=0, padx=10, pady=10)

        # Creamos los tres botones y los añadimos al frame1
        self.button1 = tk.Button(self.frame1, text="Botón 1")
        self.button1.pack(side=tk.TOP, pady=10)
        self.button2 = tk.Button(self.frame1, text="Botón 2")
        self.button2.pack(side=tk.TOP, pady=10)
        self.button3 = tk.Button(self.frame1, text="Botón 3")
        self.button3.pack(side=tk.TOP, pady=10)

        # Creamos un frame para la segunda sección con el cuadro de texto
        self.frame2 = tk.Frame(self.master)
        self.frame2.grid(row=0, column=1, padx=10, pady=10)

        # Creamos el cuadro de texto y lo añadimos al frame2
        self.textbox = tk.Text(self.frame2, height=10, width=50)
        self.textbox.insert(tk.END, "Aquí aparecerá su mensaje")
        self.textbox.config(state=tk.DISABLED)
        self.textbox.pack(side=tk.TOP, pady=10)

        # Creamos un frame para la tercera sección con la cámara web
        self.frame3 = tk.Frame(self.master)
        self.frame3.grid(row=0, column=2, padx=10, pady=10)

        # Creamos un label con el título "Letra detectada" y lo añadimos al frame3
        self.label = tk.Label(self.frame3, text="Letra detectada")
        self.label.pack(side=tk.TOP, pady=10)

        # Creamos un lienzo para mostrar la imagen de la cámara web
        self.canvas = tk.Canvas(self.frame3, width=640, height=480)
        self.canvas.pack()

        # Iniciamos la cámara web y mostramos el vídeo en tiempo real en el canvas
        self.cap = cv2.VideoCapture(0)
        self.show_frame()

    def show_frame(self):
        ret, frame = self.cap.read()
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2image = cv2.flip(cv2image, 1)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.master.after(10, self.show_frame)

root = tk.Tk()
app = MyGUI(root)
root.mainloop()
