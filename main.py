import cv2
import numpy as np
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog, messagebox
import mahotas


path = None
root = Tk()


def get_path():
    global path
    path = filedialog.askopenfilename(title='open')
    messagebox.showinfo("Success", "File was changed!")


def prepare_image(img):
    img = img.resize((500, 500), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    return img


def show_image():
    source_image = prepare_image(Image.open(path))

    source_panel = Label(root, image=source_image)
    source_panel.image = source_image
    source_panel.place(x=120, y=200)

    processed_image = prepare_image(Image.open('out.png'))
    processed_panel = Label(root, image=processed_image)
    processed_panel.image = processed_image
    processed_panel.place(x=900, y=200)


def process_niblack_image():
    global path
    if path is None:
        messagebox.showerror("Error", "Please, choose file!")
        return
    window_size = 51
    k = -0.2
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = cv2.blur(img_gray, (window_size, window_size))
    mean_sq = cv2.blur(img_gray ** 2, (window_size, window_size))
    std_dev = np.sqrt(mean_sq - mean ** 2)
    threshold = mean + k * std_dev
    binary = img_gray > threshold
    result = binary.astype(np.uint8) * 255
    cv2.imwrite('out.png', result)
    show_image()


def process_bernsen_image():
    global path
    if path is None:
        messagebox.showerror("Error", "Please, choose file!")
        return
    img = mahotas.imread(path)
    img = img[:, :, 0]
    img = mahotas.thresholding.bernsen(img, 5, 100)
    result = img.astype(np.uint8) * 255
    cv2.imwrite('out.png', result)
    show_image()


def process_high_pass_image():
    global path
    if path is None:
        messagebox.showerror("Error", "Please, choose file!")
        return
    img = cv2.imread(path)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    result = cv2.filter2D(img, -1, kernel)
    cv2.imwrite('out.png', result)
    show_image()


def process_adaptive_image():
    global path
    if path is None:
        messagebox.showerror("Error", "Please, choose file!")
        return
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 199, 5)
    cv2.imwrite('out.png', result)
    show_image()


def main():
    root.geometry("1500x750+300+150")
    root['background'] = '#856ff8'
    root.resizable(width=True, height=True)
    btn1 = Button(root, text='Choose Image', command=get_path, fg='white', bg='blue')
    btn2 = Button(root, text='Niblack Threshold', command=process_niblack_image,  fg='white', bg='blue')
    btn3 = Button(root, text='Bernsen Threshold', command=process_bernsen_image, fg='white', bg='blue')
    btn4 = Button(root, text='High Pass Filter', command=process_high_pass_image, fg='white', bg='blue')
    btn5 = Button(root, text='Adaptive Threshold', command=process_adaptive_image, fg='white', bg='blue')

    btn1.place(x=300, y=50)
    btn2.place(x=500, y=50)
    btn3.place(x=700, y=50)
    btn4.place(x=900, y=50)
    btn5.place(x=1100, y=50)
    root.mainloop()


if __name__ == "__main__":
    main()


