# Mitschrift Zeitreihenanalyse

## Grundbegriffe

- **Zeitreihen:** Messungen über die Zeit.
- **Prozesse:** Mathematische Modelle der Zeitreihen.
- **Ziel:** Erstellung einer Funktion, um zukünftige Werte vorherzusagen.
- **Rauschen entfernen:** Zufällige Schwankungen herausfiltern, um den tatsächlichen Effekt sichtbar zu machen.

## Methoden der Zeitreihenanalyse

### Moving Average (Rolling Mean)

- Beispielwerte: 251.2; 251.4; 251.3; 251.5; 252.6
- Anstatt einzelne Werte, mehrere (z.B. 3) Werte zusammenfassen für robustere Ergebnisse.
- Weniger Daten, aber deutlicher sichtbare Trends.
- Je mehr gruppiert, desto länger der Nachlaufeffekt.
- Große Sprünge besser sichtbar, "Zittern" ausgeblendet.

### Autokorrelation

- Bestimmung, wie weit man in die Vergangenheit schauen muss für eine präzise Vorhersage.
- Annahme langfristig linearer Abhängigkeiten.
- Formel:
$y(t) = f(y_{t-1}, y_{t-2}, y_{t-3}, \dots, y_2, y_1)$
- Schnelles Abfallen der Korrelation ist erwünscht (weniger Variablen).
- Periodizität sichtbar machen (z.B. wöchentliche Zyklen).

### Partielle Autokorrelation

- Restkorrelation nach erstem Autokorrelationsschritt.
- Erstes Lag immer Korrelation = 1.
- Korrelation sinkt schneller als Autokorrelation.
- Autokorrelation zeigt maximale relevante Datenanzahl, partielle Autokorrelation minimale notwendige Datenanzahl.

## Autoregressive Modelle (AR)

- Formel:
$y(t) = \alpha + b_1 y_{t-1} + b_2 y_{t-2} + \dots + \epsilon$
- Markov-Prozess/Markov-Kette: Zusammenhang exakt einen Schritt in Vergangenheit.
- Weißes Rauschen: Schwankungen um Null ohne Zeitabhängigkeit.

Beispiele:
- AR(1): Ein Schritt in Vergangenheit (einfachster Fall)
- AR(2): Zwei Schritte (z.B. Fibonacci)
- AR(7): Wochenprozesse

## ARMA-Modelle (Autoregressive Moving Average)

- Nicht Median, da zu robust für Trendverfolgung.

## ARIMA-Modelle (Autoregressive Integrated Moving Average)

- Erweiterte Version der ARMA-Modelle.
- Modelliert differenzierte Zeitreihen (Differenzenbildung zur Stabilisierung).

## Stochastic Volatility

- Große Schwankungen (hohe Volatilität) nach Ausreißern.
- Modell:
$e_t \sim N(0, \sigma_t)$

- Erweiterung der Modelle zur Berücksichtigung der Volatilität:
$\bar{y}_t = \alpha + \beta \bar{y}_{t-1} + \beta \bar{y}_{t-2} + \dots + \epsilon_t + \gamma \epsilon_{t-1} \dots$

