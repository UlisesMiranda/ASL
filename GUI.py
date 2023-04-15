from tkinter import *
import cv2
from PIL import ImageTk, Image

class GUI:
    
    def __init__(self, title, size):
        self.root = Tk()
        self.root.title(title)
        # Obtiene las dimensiones de la pantalla
        width = self.root.winfo_screenwidth()
        height = self.root.winfo_screenheight()
        self.root.geometry(f"{width}x{height}+0+0")
        self.root.configure(background='white')

    def create_frame(self, width, height, anchor, relx, rely, background='white'):
        frame = Frame(self.root, bg=background, width=width, height=height)
        frame.place(anchor=anchor, relx=relx, rely=rely)
        return frame
        
    def create_labels(self, label_num, labels, anchor, relx, rely, x_spacing=0, y_spacing=0, create_entrybox_per_label=False):
        entry_labels = {}
        entry_boxes = {}
        relx = relx
        rely = rely

        longest_label_spacing = len(max(labels, key=len))/100.0
        
        for i in range(label_num):
            label = Label(self.root, text = labels[i]+": ",
                           font = ("Century Gothic", 18, 'bold'), background='white')
            label.place(anchor=anchor, relx=relx, rely=rely)
            
            entry_labels[labels[i]] = label
            if create_entrybox_per_label:
                entry_box = Text(self.root, font=("Century Gothic", 15), height=1, width=10, borderwidth=0.1)
                entry_box.place(anchor=anchor, relx=relx+longest_label_spacing+0.1, rely=rely)
                
                entry_boxes[labels[i]+'_entrybox'] = entry_box
            rely += y_spacing
            relx += x_spacing
        return entry_labels, entry_boxes

    def create_buttons(self, button_num, text, anchor, relx, rely, command=None, x_spacing=0, y_spacing=0):
        buttons = {}
        relx = relx
        rely = rely
        
        for i in range(button_num):
            btn = Button(self.root, command=command, text=text[i], width=10, height=2, background='pink', font=("Century Gothic", 18, "bold"))
            btn.place(anchor=anchor, relx=relx, rely=rely)

            buttons[text[i]+' button'] = btn
            
            rely += y_spacing
            relx += x_spacing

        return buttons
