# -*- coding:utf-8 -*-
"""
This is the GUI of COVID-19 prediction model.
Author: Jingren Wang,Xiaocheng Chen,Jiayi Chen,Yinuo Wang
Date: Apr 18, 2022

Description:
* To run the gui successfully in Linux system, run following command to install used library
        sudo apt install tk-dev python-tk
* To run the GUI:
        python3 main.py

"""

import tkinter
import tkinter.filedialog
from tkinter import *
import matplotlib as plt
from matplotlib.pyplot import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

plt.use('TkAgg')
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from scipy import ndimage as nd
from tkinter import messagebox
import json

# import COVID19Py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import Model, Parameters
from scipy.integrate import odeint

###########################################
#            Variables                  ###
###########################################
prediction_days = 100
country = "US"

modelName = "SEIRD"
# modelName = "SEIR"
# modelName = "SIR"
initial_param = [1,0.5,0.0001,0.99,0.0001]
fit_model = {}

###########################################
#              GUI                      ###
###########################################
# create main window
root = tkinter.Tk()
root.title('COVID-19 Prediction System')

# window config
root.geometry('1000x500+100+50')  # width x height + widthoffset + heightoffset
root.configure(bg='white')
root.resizable(False, False)
root.focusmodel()

# logo
logo = ImageTk.PhotoImage(Image.open('./logo.png'))
label = Label(image=logo, border=False)
label.place(x=730, y=0)

# select Model
path = StringVar()

v1 = tkinter.IntVar()
loadModel1 = tkinter.Radiobutton(root, text='SIR', variable=v1, value=1, bg="white", fg="black")
loadModel1.place(x=720, y=150, w=80, h=30)
loadModel2 = tkinter.Radiobutton(root, text='SEIR', variable=v1, value=2, bg="white", fg="black")
loadModel2.place(x=800, y=150, w=80, h=30)
loadModel2 = tkinter.Radiobutton(root, text='SEIRD', variable=v1, value=3, bg="white", fg="black")
loadModel2.place(x=880, y=150, w=80, h=30)

# select country
v2 = tkinter.IntVar()
loadCountry1 = tkinter.Radiobutton(root, text='U.S', variable=v2, value=1, bg="white", fg="black")
loadCountry1.place(x=720, y=200, w=80, h=30)

loadCountry2 = tkinter.Radiobutton(root, text='India', variable=v2, value=2, bg="white", fg="black")
loadCountry2.place(x=800, y=200, w=80, h=30)

loadCountry3 = tkinter.Radiobutton(root, text='Mexico', variable=v2, value=3, bg="white", fg="black")
loadCountry3.place(x=880, y=200, w=80, h=30)

# set prediction duration
label1 = tkinter.Label(root, text="Prediction Duration:", bg="white", fg="black", font=('verdana', 10))
label1.place(x=720, y=252)
duration = tkinter.Text(root, width=12, height=1)
duration.place(x=860, y=250)

# panel
file_entry = Entry(root, state='readonly', text=path)
file_entry.pack()
image_label = Label(root, bg='gray')
image_label.place(x=0, y=0, width=700, height=500)

###########################################
#            Function                   ###
###########################################
"""
exit the GUI
"""


def exit_callback():
    root.destroy()
    sys.exit(0)


"""
generate fit data set
"""


def get_fit_data():
    import json
    global prediction_days
    # Opening JSON file
    f = open("../data/"+str(country) + "_data.json")

    # returns JSON object as a dictionary
    data = json.load(f)
    fit_days = len(data["date"]) - prediction_days
    # print(len(data["date"]))
    data["date"] = data["date"][0:fit_days]
    data["confirmed"] = data["confirmed"][0:fit_days]
    # print(len(data["date"]))
    # print(data)

    # Closing file
    f.close()

    # create json object from dictionary
    json = json.dumps(data)

    # open file for writing, "w"
    f = open("../data/fit_" + str(country) + "_data.json", "w")

    # write json object to file
    f.write(json)

    # close file
    f.close()


"""
Disease Prediction Model
(SIR, SEIR, SEIRD)

S: number of susceptible people on day t
I: number of infected people on day t
R: number of recovered people on day t

beta: expected amount of people an infected person infects per day
gamma: the proportion of infected recovering per day
ro: the total number of people an infected person infects 

delta:
alpha:
"""


def model_we_use(x, r0, gamma, delta, alpha, rho, population, fit, initial_infected):
    global modelName

    def deriv(y, x, r0, gamma, delta, alpha, rho):
        global modelName
        beta = r0 * gamma
        if modelName == "SIR":
            S, I, R = y
            dSdt = -beta * S * I / population
            dIdt = beta * S * I / population - gamma * I
            dRdt = gamma * I
            return dSdt, dIdt, dRdt
        elif modelName == "SEIR":
            S, E, I, R = y
            dSdt = -beta * S * I / population
            dEdt = beta * S * I / population - delta * E
            dIdt = delta * E - gamma * I
            dRdt = gamma * I
            return dSdt, dEdt, dIdt, dRdt

        elif modelName == "SEIRD":
            S, E, I, R, D = y
            dSdt = -beta * S * I / population
            dEdt = beta * S * I / population - delta * E
            dIdt = delta * E - (1 - alpha) * gamma * I - alpha * rho * I
            dRdt = (1 - alpha) * gamma * I
            dDdt = alpha * rho * I
            return dSdt, dEdt, dIdt, dRdt, dDdt

    if modelName == "SIR":
        I0 = initial_infected
        S0 = population - I0
        R0 = 0.0
        y0 = [S0, I0, R0]
        S, I, R = odeint(deriv, y0, x, args=(r0, gamma, delta, alpha, rho)).T
        return I if fit == True else [S, I, R]
    elif modelName == "SEIR":
        I0 = initial_infected
        R0 = 0.0
        E0 = I0
        S0 = population - I0 - R0
        y0 = [S0, E0, I0, R0]
        S, E, I, R = odeint(deriv, y0, x, args=(r0, gamma, delta, alpha, rho)).T
        return I if fit == True else [S, E, I, R]
    elif modelName == "SEIRD":
        I0 = initial_infected
        R0 = 0.0
        S0 = population - R0 - I0
        E0 = I0
        D0 = 0.0
        y0 = [S0, E0, I0, R0, D0]
        S, E, I, R, D = odeint(deriv, y0, x, args=(r0, gamma, delta, alpha, rho)).T
        return I if fit == True else [S, E, I, R, D]


"""
    Fit the real data with model
"""


def fit():
    global modelName
    global country
    global image_label
    global initial_param
    # Opening JSON file
    f = open('../data/fit_' + country + '_data.json')

    # returns JSON object as a dictionary
    data = json.load(f)

    # Closing file
    f.close()

    # normalized_cases = np.divide(np.array(data["confirmed"]), data["country_population"])
    x = np.linspace(0.0, len(np.array(data["confirmed"])), len(np.array(data["confirmed"])))

    params = Parameters()
    params.add("r0", value=initial_param[0], min=0.0)
    params.add("gamma", value=initial_param[1], min=0.0, max=1)
    params.add("delta", value=initial_param[2], min=0.0, max=1)
    params.add("alpha", value=initial_param[3], min=0.0, max=1)
    params.add("rho", value=initial_param[4], min=0.0, max=1)
    params.add("population", value=data["country_population"], vary=False)
    params.add("fit", value=True, vary=False)
    params.add("initial_infected", value=1000, vary=False)

    model_return = Model(model_we_use)
    model_return = model_return.fit(np.array(data["confirmed"]), params, x=x)

    image_label = Label(root, bg='gray')
    image_label.place(x=0, y=0, width=700, height=500)

    fig = plt.figure(figsize=(12, 6))
    plt.plot_date(pd.to_datetime(data["date"]), data["confirmed"], "-")
    plt.plot(pd.to_datetime(data["date"]), model_return.best_fit)
    # print(model_return.best_fit)
    plt.legend(["real data", "fit data", ])
    if modelName == "SIR":
        plt.title("SIR model fit")
    elif modelName == "SEIR":
        plt.title("SEIR model fit")
    else:
        plt.title("SEIRD model fit")

    plt.xlabel("time")
    plt.ylabel("population")
    canvas = FigureCanvasTkAgg(fig, master=image_label)
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
    canvas.draw()
    # plt.show()
    # plt.close()
    # print("r0 " + str(model_return.best_values["r0"]) + "  gamma " + str(
    #     model_return.best_values["gamma"]) + "  delta " + str(model_return.best_values["delta"]) + "  alpha " + str(
    #     model_return.best_values["alpha"]) + " rho " + str(model_return.best_values["rho"]))
    return {"r0": model_return.best_values["r0"], "gamma": model_return.best_values["gamma"],
            "delta": model_return.best_values["delta"], "alpha": model_return.best_values["alpha"],
            "rho": model_return.best_values["rho"]}


"""
fit button callback function
"""


def fit_callback():
    global country
    global modelName
    global prediction_days
    global fit_model
    global initial_param

    country_idx = v2.get()
    model_idx = v1.get()

    # parse the model type
    if model_idx == 1:
        modelName = "SIR"
        initial_param = [0.5, 0.01, 0.1, 0.1, 0.1]
    elif model_idx == 2:
        modelName = "SEIR"
        initial_param = [0.5, 0.01, 0.001, 0.1, 0.1]
    elif model_idx == 3:
        modelName = "SEIRD"
        initial_param = [1,0.5,0.0001,0.99,0.0001]
    else:
        messagebox.showerror('Error', 'Please select a model!!')
        return

    # parse the country to predict
    if country_idx == 1:
        country = "US"
    elif country_idx == 2:
        country = "IN"
    elif country_idx == 3:
        country = "MX"
    else:
        messagebox.showerror('Error', 'Please select a country!!')
        return

    # parse the prediction duration
    if duration.get(1.0) == '\n':
        messagebox.showerror('Error', 'Invalid Prediction Duration!!')
        return
    elif 0 < int(duration.get(1.0)) < 365:
        prediction_days = int(duration.get(1.0, END))
        get_fit_data()

    else:
        messagebox.showerror('Error', 'Invalid Prediction Duration!!')
        return

    # fit data
    fit_model = fit()


"""
predict and print the model
"""


def predict_callback():
    global modelName
    global fit_model
    global image_label
    if fit_model == {}:
        messagebox.showerror('Error', 'Please fit data first!!')
        return
    # Opening JSON file
    f_sim = open('../data/' + country + '_data.json')
    f_fit = open('../data/fit_' + country + '_data.json')

    # returns JSON object as a dictionary
    data = json.load(f_sim)
    fit_data = json.load(f_fit)

    # Closing file
    f_sim.close()
    f_fit.close()

    # normalized_cases = np.divide(np.array(data["confirmed"]), data["country_population"])
    x = np.linspace(0.0, len(np.array(data["confirmed"])), len(np.array(data["confirmed"])))
    if modelName == "SEIRD":
        S, E, I, R, D = model_we_use(
            x,
            fit_model["r0"],
            fit_model["gamma"],
            fit_model["delta"],
            fit_model["alpha"],
            fit_model["rho"],
            population=data["country_population"],
            fit=False,
            initial_infected=1000  # fit_data["confirmed"][-1]
        )

        # return {"S": S, "E": E, "I": I, "R": R, "D": D}
    elif modelName == "SEIR":
        S, E, I, R = model_we_use(
            x,
            fit_model["r0"],
            fit_model["gamma"],
            fit_model["delta"],
            fit_model["alpha"],
            fit_model["rho"],
            population=data["country_population"],
            fit=False,
            initial_infected=1000
        )

        # return {"S": S, "E": E, "I": I, "R": R}
    else:
        S, I, R = model_we_use(
            x,
            fit_model["r0"],
            fit_model["gamma"],
            fit_model["delta"],
            fit_model["alpha"],
            fit_model["rho"],
            population=data["country_population"],
            fit=False,
            initial_infected=1000

        )
        # return {"S": S, "I": I, "R": R}

    image_label = Label(root, bg='gray')
    image_label.place(x=0, y=0, width=700, height=500)

    fig = plt.figure(figsize=(12, 6))
    plt.plot_date(pd.to_datetime(data["date"]), S, "-")
    if modelName != "SIR":
        plt.plot_date(pd.to_datetime(data["date"]), E, "-")
    plt.plot_date(pd.to_datetime(data["date"]), I, "-")
    plt.plot_date(pd.to_datetime(data["date"]), R, "-")
    if modelName == "SEIRD":
        plt.plot_date(pd.to_datetime(data["date"]), D, "-")
    plt.plot_date(pd.to_datetime(data["date"]), data["confirmed"], "-")

    if modelName == "SIR":
        plt.title("SIR model prediction")
        plt.legend(["Susceptible", "Infected", "Recovered", "Real Confirmed"], loc='upper right')
    elif modelName == "SEIR":
        plt.title("SEIR model prediction")
        plt.legend(["Susceptible", "Exposed", "Infected", "Recovered", "Real Confirmed"], loc='upper right')
    else:
        plt.title("SEIRD model prediction")
        plt.legend(["Susceptible", "Exposed", "Infected", "Recovered", "Death", "Real Confirmed"], loc='upper right')

    plt.xlabel("time")
    plt.ylabel("population")
    canvas = FigureCanvasTkAgg(fig, master=image_label)
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
    canvas.draw()
    # plt.show()
    # plt.close()


# fit
comand1 = tkinter.Button(root, text="Fit", font=('verdana', 10, 'bold'), command=fit_callback, width=10, height=2)
comand1.place(x=740, y=300, width=220, height=40, anchor=NW)

# generate
comand2 = tkinter.Button(root, text="Generate", font=('verdana', 10, 'bold'), command=predict_callback, width=10,
                         height=2)
comand2.place(x=740, y=350, width=220, height=40, anchor=NW)

# exit
exit_ = tkinter.Button(root, text="Exit", font=('verdana', 10, 'bold'), borderwidth=2, command=exit_callback)
exit_.place(x=740, y=400, width=220, height=40)

# start main loop
root.update()
root.mainloop()
