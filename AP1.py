
import math
import numpy as np
import matplotlib.pylab as plt


def Mittelwert(array):
    """
    berechnete den mittelwert einer liste
    :param array: liste, deren mittelwert berechnet werden soll [list]
    :return: mittelwert [float]
    """
    return sum(array) / len(array)


def Varianz(array):
    """
    berechnet die Varianz einer gegebenen liste
    :param array: liste mit werten [list]
    :return: varianz der liste [float]
    """
    Var=(1/(len(array)-1))*(sum((array - Mittelwert(array))**2))
    return Var

def Standardabweichung(array):
    #berechnet die Standardabweichung einer gegebenen Liste
    return math.sqrt(Varianz(array))


def StandardabweichungMittelwert(array):
     #berechnet die Standardabweichung des Mittelwerts einer gegebenen Liste
    return Standardabweichung(array)/math.sqrt(len(array))


def TTest(x,y,dx,dy=0):
    #überprüft zwei Werte x und y mit Unsicherheiten dx und dy auf Verträglichkeit
    return abs(x-y)/(math.sqrt(dx**2 + dy**2))


def LineareRegression(x,y):
    #berechnet die für die lineare regression relevanten konstanten sowie deren unsicherheiten
    a_0 = (sum(x**2)*sum(y)-sum(x)*sum(x * y))/(len(x)*sum(x**2)-(sum(x)**2))
    #a_0 ist achsenabschnittskonstante (bei berechnung hier nicht einheiten vergessen)
    b_0 = (   len(x)*sum(x*y) - sum(x) * sum (y)  )/ (   len(x) *sum(x**2) - (sum(x))**2)
    #b_0 ist die steigungskonstante (auch hier einheiten beachten)
    s = math.sqrt(   1/(len(x)-2)*sum((y - (a_0 + b_0 *x))**2))
    #s ist streuung der messwerte und wird zur fehlerrechnung verwendet.
    da = s * math.sqrt((  sum(x**2)) / ( len(x) * sum(x**2) - (sum(x)**2)))
    #da ist standardunsicherheit der konstante a_0
    db = s * math.sqrt( (len(x)/(len(x)*sum(x**2)-(sum(x)**2))))
    #db gibt die standardunsicherheit der konstante b_0 an.
    return a_0 , b_0 , da , db


def GewichteterMittelwert(liste,fehler):
    delta = [1/fehler[i]**2 for i in range(len(fehler))]
    return sum(delta *liste)/sum(delta)





def Mittelwertfehler(fehler):
    fehler2 = [1/fehler[i]**2 for i in range(len(fehler))]
    delta = math.sqrt(sum(fehler2))
    return 1/delta

def GewichteteRegression(x,y,dx):
    #Eine gewichteste Regression macht dann sinn, wenn die messerte
    # unterschiedliche ungenauigkeiten haben.

    w  = np.divide(np.ones(len(x)), (dx**2))
    #w gibt die sogenannten gewichtsfaktoren an.

    aw_0 = (sum(w*x**2) * sum(w*y) - sum(w*x) *sum(w* x *y)) / (sum(w) *sum(w*x**2) - (sum(w *x )**2))
    # gewichteter achsenabschnitt
    bw_0 = (sum(w) * sum (w * x * y) - sum(w*x) * sum(w*y)) / (sum(w)* sum(w*x ** 2)-(sum(w * x)**2))
    #gewichtete steigung
    daw = math.sqrt(  (sum(w* x**2)) / (sum(w) * sum(w * x**2) - (sum(w*x)**2)))
    #unsicherheit des achsenabschnitts
    dbw = math.sqrt( ( sum(w)) / (sum(w)* sum(w*x**2) - (sum(w * x )**2)))
    #unsicherheit der steigung bw_0

    return aw_0, bw_0 , daw , dbw

def linear(x,a,b):
    #definiert unsere lineare regressionsfunktion
    #x ist beliebiger x wert/werte a ist achsenabschnitt b ist steigung
    return a + b * x


def ChiTest(x,y,dx,a,b):
    #Chi Quadrat test funktion um zu überprüfen ob die bestimmte lineare
    # Regression Verträglichkeit
    #mit den Messwerten, und deren Fehlern ist.
    #hier nur für fehler der x messung
    #benötigt x= Messwerte , y = Messwerte , dx = Unischerheiten
    # der x- Messwerte , a = Achesnabscnitt der Regression,
    # b= steigung der regression
    chi2 = np.sum([((y[i] - linear(x[i], a, b)) / dx[i]) ** 2
                   for i in range(len(x))])
    #den hierbei erhaltenen chi quadrat wert muss man nun noch teilen:
    ChiRed = chi2 / (len(x) - 2)
    #wenn ChiRed <1 ist , sind die messunggenauigkeiten zu klein abgeschätzt
    #wenn ChiRed >>3 ist,
    # ist die Regression nicht mit den Messpunkten
    # und ungenauigkeiten verträglich

    return ChiRed


"""
#hier noch eine grobe Code Vorlage für  einen schönen  plot :

    #plot der Messwerte mit Fehlerbalken
    plt.errorbar(x, y, y_error, x_error, fmt='x',label=r'einzelne messungen')
    #plot der linearen Regression
    #plt.plot(x, line(x, *popt),label = `hier das label schreiben` )
    #beschriftung der Achsen  ( unter umständen wie hier mit latex
      ansonsten einfach schrift )
    plt.xlabel(r'$a^{2}$ [cm$^{2}$]')
    plt.ylabel(r'$T^{2}$ [s$^{2}$]')

    #If you want customized axis, use plt.xlim(xmin, xmax)
     and plt.ylim(ymin, ymax).


    #Plots the legend.
    plt.legend()

    #Do not forget to give your graph a title! E. g.:
    plt.title(r"$T^2$ über $a^2$ zur Ermittlung der
     Richtkonstante", fontsize=20)

    #A grid in the graph is nice.
    plt.grid()

    #Save your graph in the folder your in. You can use different formats.
     E. g.:
    plt.savefig("Richtkonstante.pdf")
    plt.show()
"""


def runden(zahl, stellen=1):
    """
    Funktion rundet angegeben zahl auf anzahl der angegebenen (signifikanten)
    stellen.
    :param zahl: Zu rundende Zahl [float]
    :param stellen: anzahl der signifikanten Stellen auf die gerundet werden
    soll [int]
    :return: gerundete zahl [float]
    """
    logzahl = np.log10(abs(zahl))
    if logzahl >= 0:
        aufgerundet = math.floor(logzahl) - stellen + 1
        zahl = np.round(zahl, (-1)*aufgerundet)
        return zahl
    elif logzahl <= 0:
        logzahl = abs(logzahl)
        zahl = np.round(zahl, math.floor(logzahl)+stellen)
    return zahl


def listerunden(list, stellen=1):
    """
    Funktion die angegebene liste auf beliebig viele signifikante stellen
    rundet
    :param list: liste mit Werten, die gerundet werden sollen [list]
    :param stellen: anzahl der signifikanten stellen [int]
    :return: liste mit gerundeten werten [list]
    """
    for i, j in enumerate(list):
        list[i] = runden(j, stellen)
    return list


def anpassen(zahl, ungenauigkeit, stellen=1):
    """
    Funktion die die erste zahl auf die Anzahl der Nach-
    kommastellen der zweiten zahl anpasst. Hierbei kann angegeben werden,
    auf die wievielte signifikante stelle der zweiten zahl angepasst werden soll

    :param zahl: anzupassende zahl [float]
    :param ungenauigkeit: zahl an die 1. zahl angepasst werden soll [float]
    :param stellen: anzahl der signifikanten stellen der 2. Zahl [int]
    :return: angepasste zahl [float]
    """
    logzahl = np.log10(abs(ungenauigkeit))
    if logzahl >= 0:
        aufgerundet = math.floor(logzahl) - stellen + 1
        zahl = np.round(zahl, (-1) * aufgerundet)
        return zahl
    elif logzahl <= 0:
        logzahl = abs(logzahl)
        zahl = np.round(zahl, math.floor(logzahl) + stellen)
    return zahl


def listeanpassen(liste, ungenauigkeit, stellen=1):
    """
        Funktion die die erste liste von Werten auf die Anzahl der Nach-
        kommastellen der zweiten liste anpasst. Hierbei kann angegeben werden,
        auf die wievielte signifikante stelle der zweiten liste angepasst
        werden soll.

        :param liste: liste von Werten, welche angepasst werden sollen [list]
        :param ungenauigkeit: liste von ungenauigkeiten nach denen
        liste 1 angepasst werden soll [list]
        :param stellen: anzahl der signifikanten stellen von liste 2, auf die
        liste 1 angepasst werden soll [int]
        :return: eine Liste mit den angepassten werten [list]
        """
    for i, j in enumerate(liste):
        liste[i] = anpassen(j, ungenauigkeit[i], stellen)
    return liste


def tabelle(listen, spaltenname=None, spaltweite=12):
    """
    gibt den latex code einer tabelle aus, welche die in listen
    (listen muss liste aus listen sein) angegebenen listen (alle listen
    muessen die gleiche laenge haben) als spalten hat.
    Optional kann eine liste aus strings zur spaltenbeschriftung angegeben
    werden.
    :param listen: eine Liste aus Listen, deren Listen zu spalten der Tabelle
    werden sollen. bsp. [[1,2,3],[3,4,5],[6,7,8]] (alle listen muessen gleiche
    laenge haben.
    :param spaltenname: Liste mit strings. Strings werden als
     spaltenueberschrift
    verwendet (muss gleiche laenge wie listen haben
    :param spaltweite: Weite der der spalten in latex code
    (gut um uebersichtlichkeit zu bewahren) [int]
    :return:
    """
    print('\\toprule')
    if spaltenname is None:
        pass
    else:
        ausgabe = ''
        for i in range(len(spaltenname)):
            if i == len(listen)-1:
                ausgabe += "{:^{}}".format(spaltenname[i], spaltweite)\
                           + '  \\\\'
            else:
                ausgabe += "{:^{}}".format(spaltenname[i], spaltweite) + ' & '
        print(ausgabe)
        print('\midrule')

    for j in range(len(listen[0])):
        ausgabe = ''
        for i in range(len(listen)):
            if i == len(listen)-1:
                ausgabe += "{:<{}}".format(listen[i][j], spaltweite) + '  \\\\'
            else:
                ausgabe += "{:<{}}".format(listen[i][j], spaltweite) + ' & '
        print(ausgabe)
    print('\\bottomrule')
