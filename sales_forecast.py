# -----------------------
# Time series Forecasting
# -----------------------

# librerie
import warnings
warnings.filterwarnings('ignore')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
import statsmodels.api as sm
import datetime

color = sns.color_palette()
sns.set_style('darkgrid')

# carico dati
train = pd.read_csv('Machine_Learning_Tutorial/Dati/sales2/train.csv')
test = pd.read_csv('Machine_Learning_Tutorial/Dati/sales2/test.csv')

# formato data
train.date=train.date.apply(lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'))

# tengo solo le prime 1000 per semplicità computazionale
# train = train.loc[1:1000,]
# test = test.loc[1:300,]

# sguardo ai dati raw
train.head()
# ho la data (date) il negozio (store) l'oggetto venduto (item) ed il numero di vendite (sales)

# grandezza datasets
print("Le dimensioni di train e test sono {} e {}".format(train.shape, test.shape))

# --------------------------------------------------------------------------------------------------
# ---- Fase Uno -----
# ---- Forecast modeling - ARIMA ----

# per 1 store, 1 item
train_df = train[train['store']==1]
train_df = train_df[train['item']==1]
# train_df = train_df.set_index('date')
train_df['year'] = train['date'].dt.year
train_df['month'] = train['date'].dt.month
train_df['day'] = train['date'].dt.dayofyear
train_df['weekday'] = train['date'].dt.weekday

train_df.head()

# --------------------------------------------------------------------------------------------------
# ---- Scomposizione Time Series
# per cominciare bisogna scomporre la serie per studiare:
# -- la stagionalità
# -- il trend
# -- i residui
# dato che abbiamo 5 anni di dati ci aspettiamo una stagionalità annuale o settimanale

sns.lineplot(x="date", y="sales",legend = 'full' , data=train_df)
# dal grafico sopra possiamo vedere un trend crescente ed una stagionalità annuale
# sembra che nei mesi centrali ci sia un picco nelle vendite

sns.lineplot(x="date", y="sales",legend = 'full' , data=train_df[:28])
# dal grafico sopra non vediamo chiaramente la presenza di una stagionalità settimanale

sns.boxplot(x="weekday", y="sales", data=train_df)
# da questo grafico (sopra) possiamo vedere come in media ci sia un lieve trend crescente
# nelle vendite all'interno della settimana
# Lunedì = 0 ---- Domenica = 6
# inoltre possiamo vedere che nei giorni infra-settimanali le vendite siano minori che nel weekend

# creo train_df
train_df = train_df.set_index('date')
train_df['sales'] = train_df['sales'].astype(float)
train_df.head()

# scomposizione stagionalità
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(train_df['sales'], model='additive', freq=365)

# mandare tutto insieme
fig = plt.figure()
fig = result.plot()
fig.set_size_inches(15, 12)
#
# vediamo che è presente un trend crescente e che è evidente un pattern annuale
#  ! deduciamo che chiaramente i dati non sono stazionari

# --------------------------------------------------------------------------------------------------
# Rendere Stazionari i dati

# Il prossimo step è quello di rendere stazionari i dati, cosa vuol dire stazionario?
# 1.
# vuol dire che la media della serie non deve essere funzione del tempo
# 2.
# la varianza della serie non deve essere funzione del tempo (omoschedasticità)
# 3.
# la covarianza dell'i-esimo termine e del (i+m)-esimo termine non devono essere funzione del tempo

# Perché la stazionarietà è importante?
# perchè nonostante sappiamo che il fenomeno è tempo-dipendente, trasformarlo stazionario ci offre
# delle proprietà utili

# Ci sono due metodi per controllare la stazionarietà della serie:
# - uno studio grafico
# - Dickey-Fuller test

# ------ vediamoli entrambi:
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries, window = 12, cutoff = 0.01):

    # calcolo delle medie mobili
    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()

    # grafico delle medie mobili
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # calcolo Dickey-Fuller test:
    print('risultato Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC', maxlag = 20 )
    dfoutput = pd.Series(dftest[0:4], index=['statistica test','p-value','#Lags usati','numero di osservazioni usate'])
    for key,value in dftest[4].items():
        dfoutput['valore critico (%s)'%key] = value
    pvalue = dftest[1]
    if pvalue < cutoff:
        print('p-value = %.4f. la serie è stazionaria' % pvalue)
    else:
        print('p-value = %.4f. la serie è non-stazionaria' % pvalue)

    print(dfoutput)

test_stationarity(train_df['sales'])
# nonostante si possa vedere che la serie sarebbe stazionaria con un p value al 5% dal grafico
# vediamo che chiaramente non lo è, quindi utilizziamo una soglia più rigida per avere più fiducia
# nel nostro risultato

# --------------------
# adesso per rendere il processo stazionario possiamo usare diverse tecniche,
# come: logaritmo, differenze, ecc.
first_diff = train_df.sales - train_df.sales.shift(1)
first_diff = first_diff.dropna(inplace = False)
test_stationarity(first_diff, window = 12)
# dopo aver utilizato la tecnica delle differenze otteniamo un p-value estremamente basso e
# un riscontro grafico sulla stazionarietà della serie

# --------------------------------------------------------------------------------------------------
# ACF and PACF
# l'auto-correlazione parziale a lag k è la correlazione che rimane togliendo gli effetti
# delle altre correlazioni a lag minori

# ----- Considerazioni sull'Autocorrelazione
# consideriamo una serie storica generata da un processo auto-regressivo AR a lag k:
# sappiamo che ACF descrive l'autocorrelazione tra una osservazione ed un altra osservazione ad un
# dato istante di tempo che include informazioni di dipendenza dirette ed indirette.
# -- Ci aspettiamo che la ACF per AR(k) sia più forte ad un lag k e che l'inerzia di questa relazione
# sia presente anche a lag differenti finchè non sia troppo debole
# -- PACF descrive la relazione diretta tra un'osservazione e se stessa ad un altro lag, questo
# suggerisce che non ci dovrebbe essere correlazione per lag a valori oltre k

# ----- Considerazioni sulla media mobile
# consideriamo una serie storica generata da un processo a media mobile (MA) con lag k
# -- Ci aspettiamo che ACF di MA(k) mostri una forte correlazione con valori recenti fino al lag k,
# poi un discesa fino alla assenza di correlazione
# -- Per PACF ci aspettiamo che mostri una forte relazione con il lag e la fine della correlazione
# dal ritardo in poi

# grafici della serie storia non stazionaria
import statsmodels.api as sm
# eseguire tutto insieme
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(train_df.sales, lags=40, ax=ax1) #
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(train_df.sales, lags=40, ax=ax2)# , lags=40
# da questi grafici vediamo che c'è la necessità di una media mobile

# grafici serie storia con media mobile
# eseguire tutto insieme
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(first_diff, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(first_diff, lags=40, ax=ax2)
# vediamo che sia ACF che PACF hanno pattern ricorrenti ogni 7 giorni
# ogni volta che viene incontrato un pattern come questo bisogna sospettare una stagionalità
# quindi bisognerebbe pensare ad utilizzare un modello SARIMA che tiene conto della stagionalità

# --------------------------------------------------------------------------------------------------
# Costruzione del modello

# --- Come determinare p, d , q
# Nel nostro caso è stato semplice trovare I, poiché con I = 1 abbiamo trovato una serie stazionaria

# i modelli AR andrebbero investigati inizialmente con il lag scelto dal PACF, nel nostro caso è
# chiaro che entro 6 lag AR è significativo, quindi possiamo usare AR = 6

# La cosa interessante è che quando il modello AR è propriamente specificato, i residui del modello
# possono essere utilizzati per osservare direttamente gli errori incorrelati.
# Questi residui possono essere anche utilizzati per investigare ulteriori specificazioni dei
# modelli MA e ARMA

# Assumento un modello AR(s), suggerirei che il prossimo passo nell'identificazione sia la stima
# di un modello MA con lag s-1 sugli errori incorrelati derivati dalla regressione

arima_mod6 = sm.tsa.ARIMA(train_df.sales, (6,1,0)).fit(disp=False)
print(arima_mod6.summary())

# Analisi dei risulatati
# per vedere come performa il primo modello possiamo fare il grafico della distribuzione dei residui
# osservando se sono distribuiti normalmente. Possiamo inoltre guardare ACF e PACF
# Per un buon modello ci aspettiamo i residui normali e ACF e PACF con nessun termine significativo

from scipy import stats
from scipy.stats import normaltest

resid = arima_mod6.resid
print(normaltest(resid))
# restituisce la statistica chi-quadro ed il pvalue
# il pvalue è molto piccolo quindi i residui non hanno distribuzione normale

fig = plt.figure(figsize=(12,8))
ax0 = fig.add_subplot(111)

sns.distplot(resid ,fit = stats.norm, ax = ax0)

# estraggo i paramentri usati dalla funzione
(mu, sigma) = stats.norm.fit(resid)

# guardiamo il grafico della distribuzione
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('Residual distribution')


# ACF e PACF
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(arima_mod6.resid, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(arima_mod6.resid, lags=40, ax=ax2)

# nonostante il grafico sembri normale ha fallito il test
# inoltre vediamo una correlazione ricorrente sia nel ACF che nel PACF, quindi dobbiamo lavorare
# con la stagionalità

# --------------------------------------------------------------------------------------------------
# considerando la stagionalità con un modello SARIMA

sarima_mod6 = sm.tsa.statespace.SARIMAX(train_df.sales, trend='n', order=(6,1,0)).fit()
print(sarima_mod6.summary())

resid = sarima_mod6.resid
print(normaltest(resid))

fig = plt.figure(figsize=(12,8))
ax0 = fig.add_subplot(111)
sns.distplot(resid ,fit = stats.norm, ax = ax0)

# estraggo i paramentri usati dalla funzione
(mu, sigma) = stats.norm.fit(resid)

# grafico della distribuzione
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('Residual distribution')


# ACF e PACF
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(sarima_mod6.resid, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(sarima_mod6.resid, lags=40, ax=ax2)

# --------------------------------------------------------------------------------------------------
# previsioni e valutazione
# uso gli ultimi 30 giorni del training come test

start_index = 1730
end_index = 1826
train_df['forecast'] = sarima_mod6.predict(start = start_index, end= end_index, dynamic= True)
train_df[start_index:end_index][['sales', 'forecast']].plot(figsize=(12, 8))

def smape_kun(y_true, y_pred):
    mape = np.mean(abs((y_true-y_pred)/y_true))*100
    smape = np.mean((np.abs(y_pred - y_true) * 200/ (np.abs(y_pred) + np.abs(y_true))).fillna(0))
    print('MAPE: %.2f %% \nSMAPE: %.2f'% (mape,smape), "%")

smape_kun(train_df[1730:1825]['sales'],train_df[1730:1825]['forecast'])
