import tkinter as tk
from tkinter import *

master = tk.Tk()


# root.withdraw()
# # the input dialog
# USER_INP = simpledialog.askstring(title="Test",
#                                   prompt="What's your Name?:")

# # check it out
# print("Hello", USER_INP)
def step2():
    print(var1.get())
    print(var2.get())
    # Toplevel object which will
    # be treated as a new window
    newWindow = Toplevel(master)
 
    # sets the title of the
    # Toplevel widget
    newWindow.title("New Window")
 
    # A Label widget to show in toplevel
    Label(newWindow,
          text ="This is a new window").grid(row=1, sticky=W)
    Button(newWindow, text='Next', command=newWindow.destroy).grid(row=2, sticky=W, pady=4)


def var_states():
   print("male: %d,\nfemale: %d" % (var1.get(), var2.get(),var3.get()))
###### Main window
Label(master, text="Training").grid(row=0, sticky=W)
var1 = IntVar()
Checkbutton(master, text="SVM", variable=var1).grid(row=1,sticky=W)
var2 = IntVar()
Checkbutton(master, text="KNN", variable=var2).grid(row=2, sticky=W)
var3 = IntVar()
Checkbutton(master, text="RF", variable=var3).grid(row=3, sticky=W)
var4 = IntVar()
Checkbutton(master, text="Ensemble", variable=var4).grid(row=4, sticky=W)
Label(master, text="Features (more than 1 will combine them)").grid(row=5, sticky=W)
var5 = IntVar()
Checkbutton(master, text="LBP", variable=var5).grid(row=6, sticky=W)
var6 = IntVar()
Checkbutton(master, text="Histogram", variable=var6).grid(row=7, sticky=W)
var7 = IntVar()
Checkbutton(master, text="Sobel Edge", variable=var7).grid(row=8, sticky=W)
var8 = IntVar()
Checkbutton(master, text="Canny Edge", variable=var8).grid(row=9, sticky=W)
var9 = IntVar()
Checkbutton(master, text="pixel_intensity", variable=var9).grid(row=10, sticky=W)





Button(master, text='Quit', command=master.quit).grid(row=11, sticky=W, pady=4)
Button(master, text='Next', command=step2).grid(row=12, sticky=W, pady=4)
mainloop()










