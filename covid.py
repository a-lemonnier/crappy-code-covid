#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 23:19:56 2020

@author: A. Lemonnier
"""
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec  as gs
from matplotlib import use

from scipy.optimize import curve_fit

import requests
import urllib.request

from datetime import datetime, timedelta

import warnings

# -----------------------------------------------------------

def gaussian(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def exponential(x,a,b):
    return a*np.exp(b*x)

def logistic(x, A, B, C):
    return A / (1 + np.exp(-B * (x - C)))
# -----------------------------------------------------------
# METHOD=2: xlsx 
# METHOD=3: csv
METHOD=2

MAXFIT=10
WINDOW=3 # moving avg
COUNTRY="France"
DONT_SHOW=False

# -----------------------------------------------------------

if (len(sys.argv)>1):
    print("disable tk")
    COUNTRY=sys.argv[1]
    DONT_SHOW=True
else:
    import tkinter as tk
    from tkinter import simpledialog
    global win
    win = tk.Tk()
    win.withdraw()
    
    win.option_add('*Font', "arial 12")
    
    answer=simpledialog.askstring("Country", "Enter a country",
                                  parent=win,
                                  initialvalue=COUNTRY)
    
    if (answer!=None):
        COUNTRY=answer
        
    if (answer=="US"):
        COUNTRY="United_States_of_America"
    
    if (answer=="UK"):
        COUNTRY="United_Kingdom"

# -----------------------------------------------------------

warnings.filterwarnings("ignore", category=UserWarning)

VER=matplotlib.__version__
print("matplotlib version:", VER, "( ver < 3.3: ", VER<"3.3", ")")
NEWVER=False
if (VER>="3.3"):
    NEWVER=True

if (not DONT_SHOW):
    # sometimes backend turns things ugly
    backend="qt5agg"
else:
    backend="agg"
    print("config: ", matplotlib.matplotlib_fname())

print("backend:",backend)
use(backend)


data=[]
data_str=[]

# -----------------------------------------------------------

if (METHOD==1):
    # https://data.world/covid-19-data-resource-hub/covid-19-case-counts/workspace/file?filename=COVID-19+Cases.csv
    #
    # SQL Query:
    
    # SELECT cases
    # FROM covid_19_cases
    # WHERE country_region="France" and province_state="N/A" and case_type="Confirmed"
    # ORDER BY date ASC
    
    print("fetching data from data.world...")
    url="https://download.data.world/s/otsn7w47i7tauksqhzmopluudtnezm"
    data_str=(urllib.request.urlopen(url)).read().decode('utf-8').split('\n')
    
    print("extracting data...")
    data_str = [ s.replace('\r','') for s in data_str]
    data = [ float(x) for x in data_str if (x.isdigit() or x.isdecimal()) ]

# -----------------------------------------------------------
    
if (METHOD==2):
    import xlrd
    print("fetching data...")
   
    count=0
    acount=0
    status=0
    while(status!=200):
        
        if (acount > 5):
            simpledialog.messagebox.showerror("Error", "Cannot fetch data")
            print("Cannot fetch data")
            exit(0);
    
        day=(datetime.today() - timedelta(days=count)).strftime("%Y-%m-%d")
        url_ecdc="https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide-" \
            +str(day)+".xlsx" 
        
        try:
            status=requests.get(url_ecdc).status_code
        except:
            acount+=1
            print("Attempts: {0}".format(acount))
            
        count+=1
        
    # get xlsx
    file, header = urllib.request.urlretrieve(url_ecdc)
    
    if (not DONT_SHOW):
        print("Current country:", COUNTRY)
        print("extracting data...")   
    
    # parse xlsx
    wb=xlrd.open_workbook(file)
    sheet=wb.sheet_by_index(0)
    col_loc=sheet.col_values(6)
    col_cases=sheet.col_values(4)
    col_deaths=sheet.col_values(5)
    
    index_l=0
    index_u=0
    count=1
    
    # find rows where the country appears
    for val in col_loc:
        if (val==COUNTRY):
            if (index_l==0):
                index_l=count      
            else:
                index_u=count
        count+=1
    
    if (index_l==index_u):
            if (not DONT_SHOW):
                simpledialog.messagebox.showerror("Error","Country not found: "+ COUNTRY)
            print("country not found:", COUNTRY)
            exit(0);
    
    index_l+=-1
    index_u+=0
    
    if (not DONT_SHOW):
        print("- position: ({0}, {1})".format(index_l, index_u))
        print("- size: {0}".format(index_u-index_l))
        
        print("trimming and reverse arrays...")
    
    col_cases=col_cases[index_l:index_u]
    col_deaths=col_deaths[index_l:index_u]
    
    for val in col_cases:
        data.append(val)
    data.reverse()
    
    col_cases.reverse()
    col_deaths.reverse()

    cases_by_days=[]
    deaths_by_days=[]
    
    for i in range(index_u-index_l):
        cases_by_days.append(sum(col_cases[0:i+1]))
        deaths_by_days.append(sum(col_deaths[0:i+1]))
        
    data_c=cases_by_days
    data_d=deaths_by_days

if (METHOD==3):
    import pandas as pd
    print("fetching data...")
    
    day=datetime.today().strftime("%Y-%m-%d")
    
    acount=0
    status=0
    while(status!=200):
        
        if (acount > 5):
            simpledialog.messagebox.showerror("Error", "Cannot fetch data")
            print("Cannot fetch data")
            exit(0);
    
        url_ecdc="https://opendata.ecdc.europa.eu/covid19/casedistribution/csv" 

        try:
            status=requests.get(url_ecdc).status_code
        except:
            acount+=1
            print("Attempts: {0}".format(acount))
            
    
    if (not DONT_SHOW):
        print("Current country:", COUNTRY)
        print("extracting data...")   
    
    
    raw_data = pd.read_csv(url_ecdc, encoding = "ISO-8859-1", index_col=None, na_values=['NA'],sep=',')
    raw_data.describe()
    
    col_loc=[]
    col_cases=[]
    col_deaths=[]
    
    for val in raw_data["countriesAndTerritories"]:
        col_loc.append(val) 
    
    for val in raw_data["cases"]: 
        col_cases.append(val)
    
    for val in raw_data["deaths"]:
        col_deaths.append(val)
    
    index_l=0
    index_u=0
    count=1
    
    # find rows where the country appears
    for val in col_loc:
        if (val==COUNTRY):
            if (index_l==0):
                index_l=count      
            else:
                index_u=count
        count+=1
    
    if (index_l==index_u):
            simpledialog.messagebox.showerror("Error","Country not found: "+ COUNTRY)
            print("country not found:", COUNTRY)
            exit(0);
    
    index_l+=-1
    index_u+=0
    
    if (not DONT_SHOW):
        print("- position: ({0}, {1})".format(index_l, index_u))
        print("- size: {0}".format(index_u-index_l))
        
        print("trimming and reverse arrays...")
    
    col_cases=col_cases[index_l:index_u]
    col_deaths=col_deaths[index_l:index_u]
    
    for val in col_cases:
        data.append(val)
    
    data.reverse()
    
    col_cases.reverse()
    col_deaths.reverse()

    cases_by_days=[]
    deaths_by_days=[]
    
    for i in range(index_u-index_l):
        cases_by_days.append(sum(col_cases[0:i+1]))
        deaths_by_days.append(sum(col_deaths[0:i+1]))
        
    data_c=cases_by_days
    data_d=deaths_by_days

# -----------------------------------------------------------
# -----------------------------------------------------------
# -----------------------------------------------------------
    
x=[]
y=[]
y_c=[]
y_d=[]

days=0

for row in data:
    x.append(days)
    days+=1    
    
y=data

print("compute moving average: window:",WINDOW)

y_mov_avg=[]
y_mov_avg=np.cumsum(y, dtype=float)
y_mov_avg[WINDOW:] = y_mov_avg[WINDOW:] - y_mov_avg[:-WINDOW]
y_mov_avg=y_mov_avg[WINDOW - 1:] / WINDOW


for row in data_c:
    y_c.append(row)

for row in data_d:
    y_d.append(row)

if (not DONT_SHOW):
    print("- days:", len(x))
    print("fitting curves...")


y_max=1
y_max_c=1
y_max_d=1


center=np.max(x)
stdev=(np.max(x))/5

if (not DONT_SHOW):
    print("- init gaussian parameters:(", y_max,",", center,",", stdev,")")
    print("- init logistic parameters:(", y_max_c,",", 1,",", 1,")")
    print("- init death logistic parameters:(", y_max_d,",", 1,",", 1,")")

p_found=False
p_found_c=False
p_found_d=False

print("- Fitting gaussian")
try:
    popt,pcov = curve_fit(gaussian, x, y, p0=[y_max, center, stdev], maxfev=100000)
    p_found=True
except:
    print("Failed: Retry with dogbox method")
    try:
        popt,pcov = curve_fit(gaussian, x, y, method='dogbox', p0=[y_max, center, stdev], maxfev=100000)
        p_found=True
    except:
        if (not DONT_SHOW):
            simpledialog.messagebox.showerror("Error","Optimal parameters not found")
        print("Optimal parameters not found")
        
if (len(popt)>2 and popt[2]>=0):
    try:
        print("- try to find the best parameters")
        for i in range(MAXFIT):
            popt,pcov = curve_fit(gaussian, x, y, p0=[popt[0], popt[1], popt[2]], maxfev=100000) # try with better params
            print("-> fit",i, ": p0=", popt, " diag(C)=", np.diag(pcov))
        p_found=True
    except:
        print("Optimal parameters not found")
        popt,pcov = curve_fit(gaussian, x, y, p0=[popt[0], popt[1], popt[2]], method='dogbox', maxfev=100000)
        p_found=True
else:
    if (not DONT_SHOW):
        simpledialog.messagebox.showerror("Error","Optimal parameters not found")
    print("Optimal parameters not found")


print("- Fitting logistic")
try:
    popt_c,pcov_c = curve_fit(logistic, x, y_c, p0=[y_max_c, 1, 1], maxfev=100000)
    p_found_c=True
except:
    print("Failed: Retry with dogbox method")
    try:
        popt_c,pcov_c = curve_fit(logistic, x, y_c, method='dogbox', p0=[y_max_c, 1, 1], maxfev=100000)
        p_found_c=True
    except:
        if (not DONT_SHOW):
            simpledialog.messagebox.showerror("Error","Optimal parameters not found")
        print("Optimal parameters not found")

        
if (len(popt_c)>1):
    try:
        print("- try to find the best parameters")
        for i in range(MAXFIT):
            popt_c,pcov_c = curve_fit(logistic, x, y_c, p0=[popt_c[0], popt_c[1], popt_c[2]], maxfev=100000) # try with better params
            print("-> fit",i, ": p0=", popt, " diag(C)=", np.diag(pcov))
        p_found_c=True
    except:
        print("Optimal parameters not found")
        popt_c,pcov_c = curve_fit(logistic, x, y_c, p0=[popt_c[0], popt_c[1], popt_c[2]], method='dogbox', maxfev=100000)
        p_found_c=True
else:
    simpledialog.messagebox.showerror("Error","Optimal parameters not found")
    print("Optimal parameters not found")

if (not(p_found or p_found_c)):
    exit(0)

print("- Fitting death logistic")
try:
    popt_d,pcov_d = curve_fit(logistic, x, y_d, p0=[y_max_d, 1, 1], maxfev=100000)
    p_found_d=True
except:
    print("Failed: Retry with dogbox method")
    try:
        popt_d,pcov_d = curve_fit(logistic, x, y_d, method='dogbox', p0=[y_max_d, 1, 1], maxfev=100000)
        p_found_d=True
    except:
        if (not DONT_SHOW):
            simpledialog.messagebox.showerror("Error","Optimal parameters not found")
        print("Optimal parameters not found")

        
if (len(popt_d)>1):
    try:
        print("- try to find the best parameters")
        for i in range(MAXFIT):
            popt_d,pcov_d = curve_fit(logistic, x, y_d, p0=[popt_d[0], popt_d[1], popt_d[2]], maxfev=100000) # try with better params
            print("-> fit",i, ": p0=", popt, " diag(C)=", np.diag(pcov))
        p_found_d=True
    except:
        print("Optimal parameters not found")
        popt_d,pcov_d = curve_fit(logistic, x, y_d, p0=[popt_d[0], popt_d[1], popt_d[2]], method='dogbox', maxfev=100000)
        p_found_d=True
else:
    if (not DONT_SHOW):
        simpledialog.messagebox.showerror("Error","Optimal parameters not found")
    print("Optimal parameters not found")

if (not(p_found or p_found_c or p_found_d)):
    exit(0)

# Check finitude
p_found&=not np.isnan(np.linalg.det(pcov))
p_found&=not np.isinf(np.linalg.det(pcov))
p_found_c&=not np.isinf(np.linalg.det(pcov_c))
p_found_c&=not np.isnan(np.linalg.det(pcov_c))
p_found_d&=not np.isinf(np.linalg.det(pcov_d))
p_found_d&=not np.isnan(np.linalg.det(pcov_d))

Today=len(y)
N_today=y[len(y)-1]
N_c_today=y_c[len(y_c)-1]
N_d_today=y_d[len(y_d)-1]

# Gaussian --------------------------------------------------
sigma=popt[2]
FWHM=sigma*np.sqrt(2*np.log(2))
if (FWHM<0):
    FWHM=0
center=popt[1]
maximum=popt[0]

max_gauss=gaussian(center,*popt)
max_FWHM=gaussian(center,*popt)/2

# Exp -------------------------------------------------------
A=popt_c[0]
B=popt_c[1]
C=popt_c[1]

A_d=popt_d[0]
B_d=popt_d[1]
C_d=popt_d[1]

if (A>1e9):
    p_found_c=False

if (not DONT_SHOW):
    max_exp_2=logistic(len(y_c)+2,*popt_c)
    max_exp_3=logistic(len(y_c)+3,*popt_c)
    max_exp_5=logistic(len(y_c)+5,*popt_c)
    max_exp_10=logistic(len(y_c)+10,*popt_c)
    max_exp_20=logistic(len(y_c)+20,*popt_c)
    max_exp_30=logistic(len(y_c)+30,*popt_c)
    
    max_d_exp_2=logistic(len(y_d)+2,*popt_d)
    max_d_exp_3=logistic(len(y_d)+3,*popt_d)
    max_d_exp_5=logistic(len(y_d)+5,*popt_d)
    max_d_exp_10=logistic(len(y_d)+10,*popt_d)
    max_d_exp_20=logistic(len(y_d)+20,*popt_d)
    max_d_exp_30=logistic(len(y_d)+30,*popt_d)
    
    print("Confirmed cases:")
    print("- center:", int(center), "days")
    print("- max:", int(maximum), "cases")
    print("- FWHM:", int(FWHM), "days")
    
    print("Cumulative confirmed cases:")
    print("- A=", A, "cases")
    print("- B=", B, "/day")
    print("- C=", C, "day")
    
    print("- N(t+2)={0} cases".format(int(max_exp_2)))
    print("- N(t+3)={0} cases".format(int(max_exp_3)))
    print("- N(t+5)={0} cases".format(int(max_exp_5)))
    print("- N(t+10)={0} cases".format(int(max_exp_10)))
    print("- N(t+20)={0} cases".format(int(max_exp_20)))
    print("- N(t+30)={0} cases".format(int(max_exp_30)))
    
    print("Total deaths:")
    print("- A_d=", A_d, "cases")
    print("- B_d=", B_d, "/day")
    print("- C_d=", C_d, "day")
    
    print("- N(t+2)={0} cases".format(int(max_d_exp_2)))
    print("- N(t+3)={0} cases".format(int(max_d_exp_3)))
    print("- N(t+5)={0} cases".format(int(max_d_exp_5)))
    print("- N(t+10)={0} cases".format(int(max_d_exp_10)))
    print("- N(t+20)={0} cases".format(int(max_d_exp_20)))
    print("- N(t+30)={0} cases".format(int(max_d_exp_30)))

# -----------------------------------------------------------
# -----------------------------------------------------------
# -----------------------------------------------------------

x_f=np.linspace(np.min(x), len(y)+10, 1000)
x_f_c=np.linspace(np.min(x), len(y_c)+30, 1000)
x_f_d=np.linspace(np.min(x), len(y_c)+30, 1000)

fig=plt.figure(figsize=(19,12))


if (p_found and p_found_c and not p_found_d):
    
    TITLE_SIZE=12
    TEXT_SIZE=12
    
    print("plotting 2 curves")
    
    grid = gs.GridSpec(nrows=1, ncols=2)
    
    ax0=fig.add_subplot(grid[0,0])
    ax1=fig.add_subplot(grid[0,1])
    
    ax0.get_xaxis().get_major_formatter().set_useOffset(False)
    
    ax0.set_title(COUNTRY+" - Confirmed cases $N(t)$", size=TITLE_SIZE, fontweight='bold')
    ax0.set_xlabel("$t$ (days)",size=TEXT_SIZE)
    ax0.set_ylabel("$N$ (cases)",size=TEXT_SIZE)
    
    if (not NEWVER):
        ax0.stem(x,y,'bx',label='Confirmed cases',basefmt=" ",linefmt='lightblue', markerfmt='.') #use_line_collection=True
    else:
        ax0.stem(x,y,'bx',label='Confirmed cases',basefmt=" ",linefmt='lightblue', markerfmt='.', use_line_collection=True)
        
    ax0.plot(x[:-(WINDOW-1)],y_mov_avg,'-',color="black",label="Moving average: "+str(WINDOW)+" days", zorder=3, linewidth=1)
    ax0.plot(x_f,gaussian(x_f,*popt),'-r',label="Gaussian fit.: $\\sigma$="+str(int(sigma))+" days", zorder=3)

    ax0.plot(x_f,gaussian(x_f,*popt+np.sqrt(np.diag(pcov))),'--', color='cornflowerblue', zorder=1)
    if ((popt-np.sqrt(np.diag(pcov)))[0]>0):
        ax0.plot(x_f,gaussian(x_f,*popt-np.sqrt(np.diag(pcov))),':', color='cornflowerblue', zorder=1)

    ax0.plot([np.max(x),np.max(x)], [0, N_today],  color='b', linestyle=':', zorder=2, linewidth=1)
    ax0.annotate("$N_{0}$="+str(int(N_today))+" cases\n$t_0$="+str(int(np.max(x)))+" days", 
                xy=(np.max(x), N_today),  
                xytext=(.05, .35), textcoords='axes fraction', color = "blue",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.15", color='b'), 
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='white', lw=2),
                zorder=8, size=TEXT_SIZE)
    
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
    
    ax0.annotate("Date: "+str(day)+
            "\nPeak at $\mathbf{"+str(int(center))+"}$ days, in $\mathbf{"+str((int(center-np.max(x))))+"}$ days"+
            "\nDaily increase: {0} cases".format(int(N_today-y[len(y)-2])) +

            "\n\n$\\sqrt{C_{ii}}=$"+"{0}".format(np.sqrt(np.diag(pcov))),
            xy=(.03, .80), xycoords='axes fraction',
            bbox=dict(facecolor='lightblue'), 
            zorder=15, size=TEXT_SIZE)            
        
    if (center<Today+10):
        ax0.plot([center,center], [0, max_gauss],  color='r', linestyle=':', zorder=10)
        ax0.annotate("$N_{max}$="+str(int(max_gauss))+" cases\n$t_{max}$="+str(int(center))+" days", 
                    xy=(center, max_gauss), 
                    xytext=(.04, .55), textcoords='axes fraction', color = "red",
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15", color='r'), 
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='white', lw=2),
                    zorder=8, size=TEXT_SIZE)
    
        ax0.plot([center-FWHM,center], [max_FWHM, max_FWHM],  color='r', linestyle='-.', zorder=2)
        
        ax0.annotate("$\\frac{FWHM}{2}=$"+str(int(FWHM))+" days", 
                    xy=(center-abs(FWHM)/2, max_gauss/2), 
                    xytext=(0.9*(Today+10), max_FWHM*0.5), color = "red",
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.35"), 
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='white', lw=2),
                    zorder=8, size=TEXT_SIZE)

    ax0.legend(loc='best')

# -----
    
    ax1.set_title(COUNTRY+" - Total confirmed cases $N(t)$",size=TITLE_SIZE)
    ax1.set_xlabel("$t$ (days)",size=TEXT_SIZE)
    ax1.set_ylabel("$N$ (total cases)",size=TEXT_SIZE)
    
    ax1.plot(x,y_c,'bx',label='Total cases', marker='+', zorder=15)
    ax1.plot(x_f_c,logistic(x_f_c,*popt_c),'-r',
              label="Logistic fit.: $N(t)=\\frac{A}{1+e^{-B \cdot (t-C) }}$", zorder=15)
    
    ax1.plot(x_f_c,logistic(x_f_c,*popt_c+np.sqrt(np.diag(pcov_c))),'--', color='cornflowerblue', zorder=1)
    if ((popt_c-np.sqrt(np.diag(pcov_c)))[0]>0):
        ax1.plot(x_f_c,logistic(x_f_c,*popt_c-np.sqrt(np.diag(pcov_c))),':', color='cornflowerblue', zorder=1)
    
    ax1.plot([np.max(x),np.max(x)], [0, N_c_today],  color='b', linestyle=':', zorder=2, linewidth=1)
    ax1.annotate("$N_{0}$="+str(int(N_c_today))+" cases\n$t_0$="+str(int(np.max(x)))+" days", 
                xy=(np.max(x), N_c_today), 
                xytext=(.60, .97), textcoords='axes fraction',  
                zorder=8, color = "blue",
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='white', lw=2),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='b'),size=TEXT_SIZE)
    
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
        
    ax1.plot([len(y_c)+1, len(y_c)+1], [0, logistic(len(y_c)+1 ,*popt_c)],  color='r', linestyle=':', zorder=2, linewidth=1)
    ax1.plot([len(y_c)+2, len(y_c)+2], [0, logistic(len(y_c)+2 ,*popt_c)],  color='r', linestyle=':', zorder=2, linewidth=1)
    ax1.plot([len(y_c)+5, len(y_c)+5], [0, logistic(len(y_c)+5 ,*popt_c)],  color='r', linestyle=':', zorder=2, linewidth=1)
    ax1.plot([len(y_c)+10, len(y_c)+10], [0, logistic(len(y_c)+10 ,*popt_c)],  color='r', linestyle=':', zorder=2, linewidth=1)

    
    ax1.annotate("+1 days\n"+str(int(logistic(len(y_c)+1 ,*popt_c)))+" cases", 
                xy=(len(y_c)+1, logistic(len(y_c)+1 ,*popt_c)), 
                xytext=(0.02,0.60),
                textcoords='axes fraction', color = "tomato",
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='white', lw=2),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.0", color='r'), zorder=8,
                size=TEXT_SIZE)
    
    ax1.annotate("+5 days\n"+str(int(logistic(len(y_c)+5 ,*popt_c)))+" cases", 
                xy=(len(y_c)+5, logistic(len(y_c)+5 ,*popt_c)), 
                xytext=(0.02,0.70),
                textcoords='axes fraction', color = "red",
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='white', lw=2),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.0", color='r'), zorder=8,
                size=TEXT_SIZE)
    
    ax1.annotate("+10 days\n"+str(int(logistic(len(y_c)+10 ,*popt_c)))+" cases", 
                xy=(len(y_c)+10, logistic(len(y_c)+10 ,*popt_c)), 
                xytext=(0.02,0.80),
                textcoords='axes fraction', color = "crimson",
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='white', lw=2),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.0", color='r'), zorder=8,
                size=TEXT_SIZE)
    
    ax1.annotate("$\mathbf{"+str(int(center))+"}$ days\n"+str(int(logistic(center ,*popt_c)))+" cases", 
                xy=(center, logistic(center,*popt_c)), 
                # xytext=((len(y_c)+10)*.9, logistic(np.max(y_c) ,*popt_c)),
                xytext=(.87, .97), textcoords='axes fraction', color = "green",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.35", color='g'), zorder=8,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='white', lw=2),
                size=TEXT_SIZE)
    
    np.set_printoptions(precision=2)       
    
    ax1.text(0.03,0.1,        
    # "Date: "+str(day)+
    "\nMaximum: $\mathbf{"+str(int(A))+"}$ cases"+
    "\nDaily increase: {0} cases".format(int(N_c_today-y_c[len(y)-2])) +
    "\n$A={0:.2e}$".format(A)+
    "\n$B={0:.2e}$".format(B)+
    "\n$C={0:.2e}$".format(C)+
      "\n\n$\\sqrt{C_{ii}}=$"+"{0}".format(np.sqrt(np.diag(pcov_c))) , 
    transform = ax1.transAxes,
    bbox=dict(facecolor='lightblue'),clip_on=False, zorder=100,
    fontsize=TEXT_SIZE)
    
    ax1.legend(loc='best')

# -----------------------------------------------------------
# -----------------------------------------------------------
# -----------------------------------------------------------

if (p_found and p_found_c and p_found_d):
    
    TITLE_SIZE=11
    TEXT_SIZE=10
    
    print("plotting 3 curves")
    
    grid = gs.GridSpec(nrows=1, ncols=3)
    
    fig.set_dpi(300)
    fig.set_rasterized(False)
    
    ax0=fig.add_subplot(grid[0,0])
    ax1=fig.add_subplot(grid[0,1])
    ax2=fig.add_subplot(grid[0,2])
    
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
   
    ax0.get_xaxis().get_major_formatter().set_useOffset(False)
    
    ax0.set_title(COUNTRY+" - Confirmed cases $N(t)$", size=TITLE_SIZE, fontweight='bold')
    ax0.set_xlabel("$t$ (days)",size=TEXT_SIZE)
    ax0.set_ylabel("$N$ (cases)",size=TEXT_SIZE)
    
    if (not NEWVER):
        ax0.stem(x,y,'bx',label='Confirmed cases',basefmt=" ",linefmt='lightblue', markerfmt='.') #use_line_collection=True
    else:
        ax0.stem(x,y,'bx',label='Confirmed cases',basefmt=" ",linefmt='lightblue', markerfmt='.', use_line_collection=True)
        
    ax0.plot(x[:-(WINDOW-1)],y_mov_avg,'-',color="black",label="Moving average: "+str(WINDOW)+" days", zorder=3, linewidth=1)
    ax0.plot(x_f,gaussian(x_f,*popt),'-r',label="Gaussian fit.: $\\sigma$="+str(int(sigma))+" days", zorder=3)

    ax0.plot(x_f,gaussian(x_f,*popt+np.sqrt(np.diag(pcov))),'--', color='cornflowerblue', zorder=1)
    if ((popt-np.sqrt(np.diag(pcov)))[0]>0):
        ax0.plot(x_f,gaussian(x_f,*popt-np.sqrt(np.diag(pcov))),':', color='cornflowerblue', zorder=1)

    ax0.plot([np.max(x),np.max(x)], [0, N_today],  color='b', linestyle=':', zorder=2, linewidth=1)
    ax0.annotate("$N_{0}$="+str(int(N_today))+" cases\n$t_0$="+str(int(np.max(x)))+" days", 
                xy=(np.max(x), N_today),  
                xytext=(.05, .35), textcoords='axes fraction', color = "blue",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.15", color='b'), 
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='white', lw=2),
                zorder=8, size=TEXT_SIZE)
    
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
    
    ax0.annotate("Date: "+str(day)+
            "\nPeak at $\mathbf{"+str(int(center))+"}$ days, in $\mathbf{"+str((int(center-np.max(x))))+"}$ days"+
            "\nDaily increase: {0} cases".format(int(N_today-y[len(y)-2])) +

            "\n\n$\\sqrt{C_{ii}}=$"+"{0}".format(np.sqrt(np.diag(pcov))),
            xy=(.03, .80), xycoords='axes fraction',
            bbox=dict(facecolor='lightblue'), 
            zorder=15, size=TEXT_SIZE)            
        
    if (center<Today+10):
        ax0.plot([center,center], [0, max_gauss],  color='r', linestyle=':', zorder=10)
        ax0.annotate("$N_{max}$="+str(int(max_gauss))+" cases\n$t_{max}$="+str(int(center))+" days", 
                    xy=(center, max_gauss), 
                    xytext=(.04, .55), textcoords='axes fraction', color = "red",
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15", color='r'), 
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='white', lw=2),
                    zorder=8, size=TEXT_SIZE)
    
        ax0.plot([center-FWHM,center], [max_FWHM, max_FWHM],  color='r', linestyle='-.', zorder=2)
        
        ax0.annotate("$\\frac{FWHM}{2}=$"+str(int(FWHM))+" days", 
                    xy=(center-abs(FWHM)/2, max_gauss/2), 
                    xytext=(0.9*(Today+10), max_FWHM*0.5), color = "red",
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.35"), 
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='white', lw=2),
                    zorder=8, size=TEXT_SIZE)

    ax0.legend(shadow=True,loc='best', prop={'size': TEXT_SIZE}).set_zorder(5)

# -----
    
    ax1.set_title(COUNTRY+" - Total confirmed cases $N(t)$",size=TITLE_SIZE)
    ax1.set_xlabel("$t$ (days)",size=TEXT_SIZE)
    ax1.set_ylabel("$N$ (total cases)",size=TEXT_SIZE)
    
    ax1.plot(x,y_c,'bx',label='Total cases', marker='+', zorder=15)
    ax1.plot(x_f_c,logistic(x_f_c,*popt_c),'-r',
              label="Logistic fit.: $N(t)=\\frac{A}{1+e^{-B \cdot (t-C) }}$", zorder=15)
    
    ax1.plot(x_f_c,logistic(x_f_c,*popt_c+np.sqrt(np.diag(pcov_c))),'--', color='cornflowerblue', zorder=1)
    if ((popt_c-np.sqrt(np.diag(pcov_c)))[0]>0):
        ax1.plot(x_f_c,logistic(x_f_c,*popt_c-np.sqrt(np.diag(pcov_c))),':', color='cornflowerblue', zorder=1)
    
    ax1.plot([np.max(x),np.max(x)], [0, N_c_today],  color='b', linestyle=':', zorder=2, linewidth=1)
    ax1.annotate("$N_{0}$="+str(int(N_c_today))+" cases\n$t_0$="+str(int(np.max(x)))+" days", 
                xy=(np.max(x), N_c_today), 
                xytext=(.60, .97), textcoords='axes fraction',  
                zorder=8, color = "blue",
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='white', lw=2),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='b'),size=TEXT_SIZE)
    
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
        
    ax1.plot([len(y_c)+1, len(y_c)+1], [0, logistic(len(y_c)+1 ,*popt_c)],  color='r', linestyle=':', zorder=2, linewidth=1)
    ax1.plot([len(y_c)+2, len(y_c)+2], [0, logistic(len(y_c)+2 ,*popt_c)],  color='r', linestyle=':', zorder=2, linewidth=1)
    ax1.plot([len(y_c)+5, len(y_c)+5], [0, logistic(len(y_c)+5 ,*popt_c)],  color='r', linestyle=':', zorder=2, linewidth=1)
    ax1.plot([len(y_c)+10, len(y_c)+10], [0, logistic(len(y_c)+10 ,*popt_c)],  color='r', linestyle=':', zorder=2, linewidth=1)

    
    ax1.annotate("+1 days\n"+str(int(logistic(len(y_c)+1 ,*popt_c)))+" cases", 
                xy=(len(y_c)+1, logistic(len(y_c)+1 ,*popt_c)), 
                xytext=(0.02,0.60),
                textcoords='axes fraction', color = "tomato",
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='white', lw=2),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.0", color='r'), zorder=8,
                size=TEXT_SIZE)
    
    ax1.annotate("+5 days\n"+str(int(logistic(len(y_c)+5 ,*popt_c)))+" cases", 
                xy=(len(y_c)+5, logistic(len(y_c)+5 ,*popt_c)), 
                xytext=(0.02,0.70),
                textcoords='axes fraction', color = "red",
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='white', lw=2),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.0", color='r'), zorder=8,
                size=TEXT_SIZE)
    
    ax1.annotate("+10 days\n"+str(int(logistic(len(y_c)+10 ,*popt_c)))+" cases", 
                xy=(len(y_c)+10, logistic(len(y_c)+10 ,*popt_c)), 
                xytext=(0.02,0.80),
                textcoords='axes fraction', color = "crimson",
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='white', lw=2),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.0", color='r'), zorder=8,
                size=TEXT_SIZE)
    
    ax1.annotate("$\mathbf{"+str(int(center))+"}$ days\n"+str(int(logistic(center ,*popt_c)))+" cases", 
                xy=(center, logistic(center,*popt_c)), 
                # xytext=((len(y_c)+10)*.9, logistic(np.max(y_c) ,*popt_c)),
                xytext=(.87, .97), textcoords='axes fraction', color = "green",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.35", color='g'), zorder=8,
                size=TEXT_SIZE)
    
    np.set_printoptions(precision=2)       
    
    ax1.text(0.03,0.1,        
    # "Date: "+str(day)+
    "\nMaximum: $\mathbf{"+str(int(A))+"}$ cases"+
    "\nDaily increase: {0} cases".format(int(N_c_today-y_c[len(y)-2])) +
    "\n$A={0:.2e}$".format(A)+
    "\n$B={0:.2e}$".format(B)+
    "\n$C={0:.2e}$".format(C)+
      "\n\n$\\sqrt{C_{ii}}=$"+"{0}".format(np.sqrt(np.diag(pcov_c))) , 
    transform = ax1.transAxes,
    bbox=dict(facecolor='lightblue'),clip_on=False, zorder=100,
    fontsize=TEXT_SIZE)
    
    ax1.legend(shadow=True,loc='upper left', prop={'size': TEXT_SIZE}).set_zorder(5)
    
    # -----
    
    ax2.set_title(COUNTRY+" - Total deaths $N(t)$",size=TITLE_SIZE)
    ax2.set_xlabel("$t$ (days)",size=TEXT_SIZE)
    ax2.set_ylabel("$N$ (total deaths)",size=TEXT_SIZE)
    
    ax2.plot(x,y_d,'bx',label='Deaths', marker='+', zorder=3)
    ax2.plot(x_f_d,logistic(x_f_d,*popt_d),'-r',
             label="Logistic fit.: $N(t)=\\frac{A}{1+e^{-B \cdot (t-C) }}$", zorder=3)
    
    ax2.plot(x_f_d,logistic(x_f_d,*popt_d+np.sqrt(np.diag(pcov_d))),'--', color='cornflowerblue', zorder=1)
    if ((popt_d-np.sqrt(np.diag(pcov_d)))[0]>0):
        ax2.plot(x_f_d,logistic(x_f_d,*popt_d-np.sqrt(np.diag(pcov_d))),':', color='cornflowerblue', zorder=1)
    
    ax2.plot([np.max(x),np.max(x)], [0, N_d_today],  color='b', linestyle=':', zorder=2, linewidth=1)
    ax2.annotate("$N_{0}$="+str(int(N_d_today))+" Total deaths\n$t_0$="+str(int(np.max(x)))+" days", 
                xy=(np.max(x), N_d_today), 
                # xytext=(0, N_d_today),  
                xytext=(0.02,0.50),
                textcoords='axes fraction',
                zorder=8, color = "blue",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='b'),size=TEXT_SIZE)
    
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
        
    ax2.plot([len(y_d)+1, len(y_d)+1], [0, logistic(len(y_d)+1 ,*popt_d)],  color='r', linestyle=':', zorder=2, linewidth=1)
    ax2.plot([len(y_d)+2, len(y_d)+2], [0, logistic(len(y_d)+2 ,*popt_d)],  color='r', linestyle=':', zorder=2, linewidth=1)
    ax2.plot([len(y_d)+5, len(y_d)+5], [0, logistic(len(y_d)+5 ,*popt_d)],  color='r', linestyle=':', zorder=2, linewidth=1)
    ax2.plot([len(y_d)+10, len(y_d)+10], [0, logistic(len(y_d)+10 ,*popt_d)],  color='r', linestyle=':', zorder=2, linewidth=1)
    
    ax2.annotate("+10 days\n"+str(int(logistic(len(y_d)+10 ,*popt_d)))+" cases", 
                xy=(len(y_d)+10, logistic(len(y_d)+10 ,*popt_d)), 
                xytext=(.60, .97), textcoords='axes fraction',  color = "red",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='r'), zorder=8,
                size=TEXT_SIZE)
    
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)

    ax2.text(0.03,.1,
                # "Date: "+str(day)+
                "\nMaximum: $\mathbf{"+str(int(A_d))+"}$ Total deaths"+
                #"\nGaussian peak: {0} Total deaths".format((int(logistic(center ,*popt_d))))+
                "\nDaily increase: {0} Total deaths".format(int(N_d_today-y_d[len(y_d)-2])) +
                "\n$A_d={0:.2e}$".format(A_d)+
                "\n$B_d={0:.2e}$".format(B_d)+
                "\n$C_d={0:.2e}$".format(C_d)+
                  # "\n\nCovariance matrix $C(A,B,C)=C(N_{max}, \\tau, t_s)$:\n"+str(pcov_d) +
                  "\n\n$\\sqrt{C_{ii}}=$"+"{0}".format(np.sqrt(np.diag(pcov_d))) ,
                transform = ax2.transAxes,
                bbox=dict(facecolor='lightblue'), zorder=15,
                size=TEXT_SIZE)

    ax2.legend(shadow=True,loc='upper left', prop={'size': TEXT_SIZE}).set_zorder(5)
    
# -----------------------------------------------------------
# -----------------------------------------------------------
# -----------------------------------------------------------

if (p_found and not p_found_c):
   
    TITLE_SIZE=12
    TEXT_SIZE=12
    
    grid = gs.GridSpec(nrows=1, ncols=1)

    ax0=fig.add_subplot(grid[0,0])
    
    ax0.set_title(COUNTRY+" - Confirmed cases $N(t)$", size=TITLE_SIZE, fontweight='bold')
    ax0.set_xlabel("$t$ (days)",size=TEXT_SIZE)
    ax0.set_ylabel("$N$ (cases)",size=TEXT_SIZE)
    
    if (not NEWVER):
        ax0.stem(x,y,'bx',label='Confirmed cases',basefmt=" ",linefmt='lightblue', markerfmt='.') #use_line_collection=True
    else:
        ax0.stem(x,y,'bx',label='Confirmed cases',basefmt=" ",linefmt='lightblue', markerfmt='.', use_line_collection=True)
        
    ax0.plot(x[:-(WINDOW-1)],y_mov_avg,'-',color="black",label="Moving average: "+str(WINDOW)+" days", zorder=3, linewidth=1)
    ax0.plot(x_f,gaussian(x_f,*popt),'-r',label="Gaussian fit.: $\\sigma$="+str(int(sigma))+" days", zorder=3)

    ax0.plot(x_f,gaussian(x_f,*popt+np.sqrt(np.diag(pcov))),'--', color='cornflowerblue', zorder=1)
    if ((popt-np.sqrt(np.diag(pcov)))[0]>0):
        ax0.plot(x_f,gaussian(x_f,*popt-np.sqrt(np.diag(pcov))),':', color='cornflowerblue', zorder=1)

    ax0.plot([np.max(x),np.max(x)], [0, N_today],  color='b', linestyle=':', zorder=2, linewidth=1)
    ax0.annotate("$N_{0}$="+str(int(N_today))+" cases\n$t_0$="+str(int(np.max(x)))+" days", 
                xy=(np.max(x), N_today),  
                xytext=(.05, .35), textcoords='axes fraction', color = "blue",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.15", color='b'), 
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='white', lw=2),
                zorder=8, size=TEXT_SIZE)
    
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
    
    ax0.annotate("Date: "+str(day)+
            "\nPeak at $\mathbf{"+str(int(center))+"}$ days, in $\mathbf{"+str((int(center-np.max(x))))+"}$ days"+
            "\nDaily increase: {0} cases".format(int(N_today-y[len(y)-2])) +

            "\n\n$\\sqrt{C_{ii}}=$"+"{0}".format(np.sqrt(np.diag(pcov))),
            xy=(.03, .80), xycoords='axes fraction',
            bbox=dict(facecolor='lightblue'), 
            zorder=15, size=TEXT_SIZE)            
        
    if (center<Today+10):
        ax0.plot([center,center], [0, max_gauss],  color='r', linestyle=':', zorder=10)
        ax0.annotate("$N_{max}$="+str(int(max_gauss))+" cases\n$t_{max}$="+str(int(center))+" days", 
                    xy=(center, max_gauss), 
                    xytext=(.04, .55), textcoords='axes fraction', color = "red",
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15", color='r'), 
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='white', lw=2),
                    zorder=8, size=TEXT_SIZE)
    
        ax0.plot([center-FWHM,center], [max_FWHM, max_FWHM],  color='r', linestyle='-.', zorder=2)
        
        ax0.annotate("$\\frac{1}{2}\ FWHM=$"+str(int(FWHM))+" days", 
                    xy=(center-abs(FWHM)/2, max_gauss/2), 
                    xytext=(0.9*(Today+10), max_FWHM*0.5), color = "red",
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.35"), 
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='white', lw=2),
                    zorder=8, size=TEXT_SIZE)

    ax0.legend(loc='best')

# -----------------------------------------------------------
# -----------------------------------------------------------
# -----------------------------------------------------------
        
if (not p_found and p_found_c):
    
    TITLE_SIZE=12
    TEXT_SIZE=12
    
    grid = gs.GridSpec(nrows=1, ncols=1)
    
    ax1=fig.add_subplot(grid[0,0])
    
    ax1.set_title(COUNTRY+" - Total confirmed cases $N(t)$",size=TITLE_SIZE)
    ax1.set_xlabel("$t$ (days)",size=TEXT_SIZE)
    ax1.set_ylabel("$N$ (total cases)",size=TEXT_SIZE)
    
    ax1.plot(x,y_c,'bx',label='Total cases', marker='+', zorder=15)
    ax1.plot(x_f_c,logistic(x_f_c,*popt_c),'-r',
              label="Logistic fit.: $N(t)=\\frac{A}{1+e^{-B \cdot (t-C) }}$", zorder=15)
    
    ax1.plot(x_f_c,logistic(x_f_c,*popt_c+np.sqrt(np.diag(pcov_c))),'--', color='cornflowerblue', zorder=1)
    if ((popt_c-np.sqrt(np.diag(pcov_c)))[0]>0):
        ax1.plot(x_f_c,logistic(x_f_c,*popt_c-np.sqrt(np.diag(pcov_c))),':', color='cornflowerblue', zorder=1)
    
    ax1.plot([np.max(x),np.max(x)], [0, N_c_today],  color='b', linestyle=':', zorder=2, linewidth=1)
    ax1.annotate("$N_{0}$="+str(int(N_c_today))+" cases\n$t_0$="+str(int(np.max(x)))+" days", 
                xy=(np.max(x), N_c_today), 
                xytext=(.60, .97), textcoords='axes fraction',  
                zorder=8, color = "blue",
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='white', lw=2),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='b'),size=TEXT_SIZE)
    
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
        
    ax1.plot([len(y_c)+1, len(y_c)+1], [0, logistic(len(y_c)+1 ,*popt_c)],  color='r', linestyle=':', zorder=2, linewidth=1)
    ax1.plot([len(y_c)+2, len(y_c)+2], [0, logistic(len(y_c)+2 ,*popt_c)],  color='r', linestyle=':', zorder=2, linewidth=1)
    ax1.plot([len(y_c)+5, len(y_c)+5], [0, logistic(len(y_c)+5 ,*popt_c)],  color='r', linestyle=':', zorder=2, linewidth=1)
    ax1.plot([len(y_c)+10, len(y_c)+10], [0, logistic(len(y_c)+10 ,*popt_c)],  color='r', linestyle=':', zorder=2, linewidth=1)

    
    ax1.annotate("+1 days\n"+str(int(logistic(len(y_c)+1 ,*popt_c)))+" cases", 
                xy=(len(y_c)+1, logistic(len(y_c)+1 ,*popt_c)), 
                xytext=(0.02,0.60),
                textcoords='axes fraction', color = "tomato",
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='white', lw=2),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.0", color='r'), zorder=8,
                size=TEXT_SIZE)
    
    ax1.annotate("+5 days\n"+str(int(logistic(len(y_c)+5 ,*popt_c)))+" cases", 
                xy=(len(y_c)+5, logistic(len(y_c)+5 ,*popt_c)), 
                xytext=(0.02,0.70),
                textcoords='axes fraction', color = "red",
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='white', lw=2),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.0", color='r'), zorder=8,
                size=TEXT_SIZE)
    
    ax1.annotate("+10 days\n"+str(int(logistic(len(y_c)+10 ,*popt_c)))+" cases", 
                xy=(len(y_c)+10, logistic(len(y_c)+10 ,*popt_c)), 
                xytext=(0.02,0.80),
                textcoords='axes fraction', color = "crimson",
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='white', lw=2),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.0", color='r'), zorder=8,
                size=TEXT_SIZE)
    
    ax1.annotate("$\mathbf{"+str(int(center))+"}$ days\n"+str(int(logistic(center ,*popt_c)))+" cases", 
                xy=(center, logistic(center,*popt_c)), 
                # xytext=((len(y_c)+10)*.9, logistic(np.max(y_c) ,*popt_c)),
                xytext=(.87, .97), textcoords='axes fraction', color = "green",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.35", color='g'), zorder=8,
                size=TEXT_SIZE)
    
    np.set_printoptions(precision=2)       
    
    ax1.text(0.03,0.1,        
    # "Date: "+str(day)+
    "\nMaximum: $\mathbf{"+str(int(A))+"}$ cases"+
    "\nDaily increase: {0} cases".format(int(N_c_today-y_c[len(y)-2])) +
    "\n$A={0:.2e}$".format(A)+
    "\n$B={0:.2e}$".format(B)+
    "\n$C={0:.2e}$".format(C)+
      "\n\n$\\sqrt{C_{ii}}=$"+"{0}".format(np.sqrt(np.diag(pcov_c))) , 
    transform = ax1.transAxes,
    bbox=dict(facecolor='lightblue'),clip_on=False, zorder=100,
    fontsize=TEXT_SIZE)
    
    ax1.legend(loc='best')

# -----------------------------------------------------------
# -----------------------------------------------------------
# -----------------------------------------------------------


fig.subplots_adjust(left=0.06, bottom=0.08, right=0.98, top=0.95, wspace=0.16)

fig.set_dpi(300)
fig.savefig("covid_"+COUNTRY+"_"+str(day)+".png", dpi=300)
fig.savefig("covid_"+COUNTRY+"_"+str(day)+".pdf", dpi=300)

if (not DONT_SHOW):
    plt.show()


# -----------------------------------------------------------
# -----------------------------------------------------------
# -----------------------------------------------------------

