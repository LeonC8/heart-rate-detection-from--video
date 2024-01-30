# Detekcija lica i analiza srčane frekvencije
Ova skripta omogućava detekciju lica i analizu prosječnih zelenih vrijednosti u video zapisima kako bi se procijenila frekvencija kucanja srca pomoću Eulerove magnifikacije videa.

# Instalacija
Prije pokretanja skripta, potrebno je instalirati potrebne biblioteke pomoću pip:

```bash
pip install -r requirements.txt
```

# Korištenje
Skript se pokreće iz komandne linije. Možete odabrati hoće li se izvršiti detekcija lica u videu.
```bash
python detect.py --detect_face "putanja/do/videa.mp4"
```

Ako ne želite da se izvrši detekcija lica, putanju videa jednostavno navedite bez dodatnih opcija:
```bash
python detect.py "putanja/do/videa.mp4"
```

Također, ako želite da se prikaže frekvencijski spektar, može se postaviti zastavica --plot
```bash
python detect.py --plot "putanja/do/videa.mp4"
```

# Funkcionalnosti
1. face_detection_and_crop: Detektira lica u slikama i obrezuje ih.
2. loadVideo: Učitava video zapis i ekstrahira slike.
3. detect_face: Učitava video zapis, detektira lica i obrezuje slike lica.
4. main: Glavna funkcija za izvršavanje skripta s argumentima komandne linije.

# Argumenti komandne linije
1. video_path: Putanja do video datoteke koja se obrađuje.
2. --detect_face: Opcionalna zastavica koja omogućava detekciju lica u videu.
3. --plot: Opcionalna zastavica koja omogućava prikaz grafa frekvencijskog spektra.

Skripta će ispisati procijenjeni broj otkucaja srca u otkucajima u minuti (bpm).