import mediapipe as mp
from PIL import ImageTk, Image
import cv2
import tkinter as tk
import numpy as np
from pathlib import Path
from tkinter import Tk, Canvas, Text, Button, PhotoImage,Frame,Label
from src.cnn import HandClassifierModel
from utils import show_toast, copy_text, delete_text, eliminate_last_word, text_to_speech, auto_correct


class InterfazHappyHand():
    def __init__(self, MODEL_PATH="../models/grayscale_classifier.h5"):
        self.model = HandClassifierModel()
        self.model.load_trained_model(MODEL_PATH)
        print("este es el modelooooooooooooooooooooo", self.model)
        self.mediapipe_drawing = mp.solutions.drawing_utils
        self.mediapipe_hands = mp.solutions.hands
        self.videoCapture = cv2.VideoCapture(0)
        
        self.OUTPUT_PATH = Path(__file__).parent.parent
        self.ASSETS_PATH = self.OUTPUT_PATH / Path("./assets/frame0")
        
        self.window = Tk()
        self.window.geometry("1138x720+200+0")
        self.window.configure(bg = "#F5FDF8")
        
        self.canvas = Canvas(
            self.window,
            bg = "#F5FDF8",
            height = 720,
            width = 1138,
            bd = 0,
            highlightthickness = 0,
            relief = "ridge"
        )
        self.canvas.place(x = 0, y = 0)
        
        self.image_image_1 = PhotoImage(
            file=self.relative_to_assets("image_1.png")
        )

        self.xd = self.canvas.create_image(
            821.6666259765625,
            296.0,
            image=self.image_image_1
        )

        self.entry_image_1 = PhotoImage(
            file=self.relative_to_assets("entry_1.png"))
        self.entry_bg_1 = self.canvas.create_image(
            316.7642364501953,
            357.955322265625,
            image=self.entry_image_1
        )

        # Input de texto
        self.sent_entrybox = Text(
            bd=0,
            bg="#FFFFFF",
            fg="#000716",
            highlightthickness=0,
            font=('Helvetica', 10)
        )
        self.sent_entrybox.place(
            x=163,
            y=45.0,
            width=310,
            height=630,
        )

        self.canvas.create_text(
            164.60975646972656,
            70.25201416015625,
            anchor="nw",
            text="Tu mensaje aparecera aqui...\n",
            fill="#858383",
            font=("Inter", 16 * -1)
        )

        self.canvas.create_rectangle(
            566.0,
            608.0,
            1080.0,
            692.0,
            fill="#B80090",
            outline="")

        self.canvas.create_text(
            583.0,
            633.0,
            anchor="nw",
            text="Palabra detectada: ",
            fill="#FFFFFF",
            font=("Helvetica", 30 * -1, 'bold')
        )

        self.canvas.create_text(
            850.0,
            633.0,
            anchor="nw",
            text="",
            fill="#FFFFFF",
            font=("OpenSansRoman CondensedBold", 32 * -1),
            tags="palabra"
        )

        self.canvas.create_rectangle(
            14.0,
            27.0,
            95.86991882324219,
            688.91064453125,
            fill="#FFFFFF",
            outline="")

        self.button_image_1 = PhotoImage(
            file=self.relative_to_assets("button_1.png"))
        self.button_1 = Button(
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: print("button_1 clicked"),
            relief="flat"
        )
        self.button_1.place(
            x=25.0,
            y=41.0,
            width=62.0,
            height=104.0
        )

        # Copiar
        self.button_image_2 = PhotoImage(
            file=self.relative_to_assets("button_2.png"))
        self.button_2 = Button(
            image=self.button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: copy_text(self.sent_entrybox),
            relief="flat"
        )
        self.button_2.place(
            x=30,
            y=233,
            width=48.658538818359375,
            height=54.83740234375
        )

        # Borrar la ultima palabra
        self.button_image_6 = PhotoImage(
            file=self.relative_to_assets("borrar ultima.png"))
        self.button_6 = Button(
            image=self.button_image_6,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: eliminate_last_word(self.sent_entrybox),
            relief="flat"
        )
        self.button_6.place(
            x=29.0,
            y=315.0,
            width=48.658538818359375,
            height=54.83740234375
        )

        # Borrar
        self.button_image_3 = PhotoImage(
            file=self.relative_to_assets("button_3.png"))
        self.button_3 = Button(
            image=self.button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: delete_text(self.sent_entrybox),
            relief="flat"
        )
        self.button_3.place(
            x=30.219512939453125,
            y=397,
            width=48.658538818359375,
            height=54.83740234375
        )
        # texto a voz
        self.button_image_4 = PhotoImage(
            file=self.relative_to_assets("button_4.png"))
        self.button_4 = Button(
            image=self.button_image_4,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: text_to_speech(self.sent_entrybox),
            relief="flat"
        )
        self.button_4.place(
            x=30.99188232421875,
            y=479,
            width=48.658538818359375,
            height=54.83740234375
        )
            

    def relative_to_assets(self, path: str) -> Path:
        return str(self.ASSETS_PATH / Path(path)).replace("\\", "/")

    def update_image_frame(self, image, video_label):
        image_fromarray = Image.fromarray(image)
        imgtk = ImageTk.PhotoImage(image=image_fromarray)
        
        video_label.imgtk = imgtk
        video_label.config(image=imgtk)
        
    def create_frame(self, width, height, anchor, relx, rely, background='white'):
        frame = Frame(self.window, bg=background, width=width, height=height)
        frame.place(anchor=anchor, relx=relx, rely=rely)
        return frame
    
    def isSpace(self, current_char):
        return current_char == "space"
    
    def isDelete(self, current_char):
        return current_char == "del"
        
    def isValidChar(self, current_char):
        return current_char != "nothing"
    
    def addCharToWord(self, word, current_char):
        temp_word = word
        
        if self.isSpace(current_char):
            show_toast("Palabra agregada" , "750", "450")
            temp_word = ""
            
        elif self.isDelete(current_char):
            temp_word = temp_word[0:-1]
            show_toast('El último caracter ha sido borrado', "750", "450")
            self.canvas.delete("word")
            self.canvas.create_text(
                860.0,
                633.0,
                anchor="nw",
                text=f"{temp_word}",
                fill="#FFFFFF",
                font=("Helvetica", 32 * -1),
                tags="word"
            )
            
        elif self.isValidChar(current_char):
            temp_word += current_char.lower()
            print('Character added: ', current_char.lower())

        return temp_word, current_char

    def get_char(self, gesture):
        return self.model.predict_gesture(gesture)

    def flip_image_to_selfie(self, frame):
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (620, 560))
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def update_new_frame_entry(self, video_label):
        success, frame = self.videoCapture.read()
        
        selfie_image = self.flip_image_to_selfie(frame)
        self.update_image_frame(selfie_image, video_label)
        
        return selfie_image
            
    def process_for_HandsMediapipe(self, image, hands_object):
        image.flags.writeable = False
        results_mediapipe = hands_object.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return results_mediapipe, image

    def hands_detection_drawing_marks(self, hand_landmarks, image_width, image_height): 
        x = [landmark.x for landmark in hand_landmarks.landmark]
        y = [landmark.y for landmark in hand_landmarks.landmark]

        center = np.array([np.mean(x) * image_width, np.mean(y) * image_height]).astype('int32')
        
        return x, y, center

    def get_draw_region(self, center, image):
        full_image = cv2.rectangle(image, 
                                    (center[0] - 130, center[1] - 130),
                                    (center[0] + 130, center[1] + 130),
                                    (184, 0, 144),
                                    2)
        
        rectangle = full_image[center[1]-130:center[1]+130, center[0]-130:center[0]+130]
        
        return full_image, rectangle

    

    """VIDEO"""
    def frame_video_stream(self, video_label: Label, current_char, previous_char, word, hands_object, threshold=0.9):
        
        image = self.update_new_frame_entry(video_label)
        
        results_mediapipe, image = self.process_for_HandsMediapipe(image, hands_object)
        image_height, image_width, _ = image.shape

        if results_mediapipe.multi_hand_landmarks:
            
            for hand_landmarks in results_mediapipe.multi_hand_landmarks:
                
                x, y, center = self.hands_detection_drawing_marks(hand_landmarks, image_width, image_height)
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                full_img, rectangle = self.get_draw_region(center, image)
                self.update_image_frame(full_img, video_label)


                try:
                    # Transform the cropped hand into gray image
                    gray_image = cv2.cvtColor(rectangle, cv2.COLOR_BGR2GRAY)
                    
                    current_char, confidence = self.get_char(gray_image)
                    
                    char_text = cv2.putText(
                        full_img,
                        current_char,
                        (center[0]-135, center[1]-135),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (184, 0, 144),
                        2,
                        cv2.LINE_AA
                    )
                    
                    char_prob_text = cv2.putText(
                        full_img,
                        '{0:.2f}'.format(np.max(confidence)),
                        (center[0]+60, center[1]-135),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, 
                        (184, 0, 144),
                        2, 
                        cv2.LINE_AA
                    )

                    self.update_image_frame(full_img, video_label)

                    if self.validEntryChar(current_char, previous_char, confidence, threshold):
                        
                        temporal_word, current_char = self.addCharToWord(word, current_char)
                        
                        if self.validEntryWord(temporal_word, current_char, word):
                            corrected_word = auto_correct(word) 
                            index = self.sent_entrybox.index(tk.INSERT)
                            self.sent_entrybox.insert(index, corrected_word + " ")
                        
                        word = temporal_word
                        
                        self.canvas.delete("palabra")
                        self.canvas.create_text(
                            860.0,
                            633.0,
                            anchor="nw",
                            text=f"{word}",
                            fill="#FFFFFF",
                            font=("Helvetica", 32 * -1),
                            tags="palabra"
                        )
                        
                        previous_char = current_char
                except:
                    print("Algo salio mal")
                    pass
                
        video_label.after(1, self.frame_video_stream, video_label, current_char, previous_char, word, hands_object, threshold)
    
    
    def validEntryChar(self, current_char, previous_char, confidence, threshold):
        return (current_char != previous_char) and (confidence > threshold)
    
    def validEntryWord(self, temporal_word, current_char, word):
        return (temporal_word == "") and (current_char != "del") and (word != "")

    def pipe_cam(self, current_char=None, previous_char=None, word=""):   
        
        gui_frame = self.create_frame(650, 600, 'ne', 1, 0, 'green')
        video_label = Label(gui_frame)
        video_label.grid()
            
        with self.mediapipe_hands.Hands(min_detection_confidence=0.4, min_tracking_confidence=0.5, max_num_hands=1) as hands_object:
            
                self.frame_video_stream(video_label=video_label, current_char=current_char, previous_char=previous_char, word=word, hands_object=hands_object)
                
                self.window.resizable(False, False)
                self.window.mainloop()


    def startHappyHandApp(self):
        self.pipe_cam()
        self.window.mainloop()
    
    
if __name__ == "__main__":
    happyHand = InterfazHappyHand()
    happyHand.startHappyHandApp()
