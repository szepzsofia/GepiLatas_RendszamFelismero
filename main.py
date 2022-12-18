import cv2
import imutils
import pytesseract
import messagebox

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract'

kep = cv2.imread('test1.jpg')
kep = imutils.resize(kep, width=300) #Kép átméretezése
cv2.imshow("Alap kep", kep) #Megjelenik az alap kép átméretezve

szurke_kep = cv2.cvtColor(kep, cv2.COLOR_BGR2GRAY) #Kép színének szürkeárnyalatossá tétele
#cv2.imshow("Szurke arnyalatos kep", szurke_kep) # Szürkeárnyalatos kép megjelenítése

szurke_kep = cv2.bilateralFilter(szurke_kep, 11, 17, 17)
#cv2.imshow("Simitott kep", szurke_kep)

elek = cv2.Canny(szurke_kep, 30, 200)
#cv2.imshow("El detektalas", elek)

kontur,new = cv2.findContours(elek.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
kepmas = kep.copy()
cv2.drawContours(kepmas,kontur,-1,(0,255,0),3)
#cv2.imshow("Konturok",kepmas)

kontur = sorted(kontur, key = cv2.contourArea, reverse = True) [:10]
korvonal = None
kepmas2 = kep.copy()
cv2.drawContours(kepmas2,kontur,-1,(0,255,0),3)
cv2.imshow("Top 10 kontur",kepmas)

i=7
for c in kontur:
        kerulet = cv2.arcLength(c, True)
        kb = cv2.approxPolyDP(c, 0.018 * kerulet, True)
        if len(kb) == 4:
                korvonal = kb

        x, y, w, h = cv2.boundingRect(c)
        uj_kep = kep[y:y + h, x:x + w]
        cv2.imwrite('./' + str(i) + '.png', uj_kep)
        i += 1
        break

cv2.drawContours(kep, [korvonal], -1, (0, 255, 0), 3)
cv2.imshow("Rendszam tabla", kep)

kivagott = './7.png'
cv2.imshow("Korbevagott", cv2.imread(kivagott))

kiv = cv2.imread(kivagott)


# Szürkeárnyalatossá konvertálás és elhomályosítás
szurkearnyalat = cv2.cvtColor(kiv, cv2.COLOR_BGR2GRAY)
homalyos = cv2.GaussianBlur(szurkearnyalat, (7, 7), 0)

cv2.imshow("Szurkearnyalatos kep", szurkearnyalat)
cv2.imshow("Elhomalyositott", homalyos)

# Az inverz bináris küszöbérték használata
binaris = cv2.threshold(szurkearnyalat, 255, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cv2.imshow("Binaris", binaris)

def rovid_elek(kontur, reverse=False):
        i = 0
        doboz = [cv2.boundingRect(c) for c in kontur]
        (kontur, doboz) = zip(*sorted(zip(kontur, doboz),
                        key=lambda b: b[1][i], reverse=reverse))
        return kontur

kont, _ = cv2.findContours(binaris, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Másolat létrehozása a keret rajzolásához
masolat = kiv.copy()

# Inicializált lista létrehozása
kar_vagas = []

# Karakter szélességének és magasságának meghatározása
szeles, magas = 30, 60

for c in rovid_elek(kont):
        (x, y, w, h) = cv2.boundingRect(c)
        arany = h / w
        if 1 <= arany <= 3.5:  # Kontúr meghatározása
                if h / kiv.shape[0] >= 0.5:  # Válassza ki azt a kontúrt, ami nagyobb a kivágott rész 50%-át
                        # Karakterek köré berajzolja a keretet
                        cv2.rectangle(masolat, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Szétválasztja a karaktereket
                        jel_szam = szurkearnyalat[y:y + h, x:x + w]
                        jel_szam = cv2.resize(jel_szam, dsize=(szeles, magas))
                        _, jel_szam = cv2.threshold(jel_szam, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        kar_vagas.append(jel_szam)

for i in range(len(kar_vagas)):
        cv2.imshow("Kivagott karakterek", kar_vagas[i])
        cv2.waitKey(0)

rendszam = pytesseract.image_to_string(binaris, lang='eng')

messagebox.showinfo(title="Erzekelt Rendszam", message=rendszam)
cv2.waitKey(0)
cv2.destroyAllWindows()


