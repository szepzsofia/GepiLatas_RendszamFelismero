import cv2
import imutils
import pytesseract
import messagebox

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract'

kep = cv2.imread('test.jpg')
kep = imutils.resize(kep, width=300) #Kép átméretezése
cv2.imshow("Alap kep", kep) #Megjelenik az alap kép átméretezve

szurke_kep = cv2.cvtColor(kep, cv2.COLOR_BGR2GRAY) #Kép színének szürkeárnyalatossá tétele
cv2.imshow("Szurke arnyalatos kep", szurke_kep) #Szürkeárnyalatos kép megjelenítése

szurke_kep = cv2.bilateralFilter(szurke_kep, 11, 17, 17) #Kép simítása
cv2.imshow("Simitott kep", szurke_kep) #Simított kép megjelenítése

szel = cv2.Canny(szurke_kep, 30, 200) #Kép éleinek detektálása
cv2.imshow("El detektalas", szel)

kontur,new = cv2.findContours(szel.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #Összes kontur keresése
kepmas=kep.copy()
cv2.drawContours(kepmas,kontur,-1,(0,255,0),3) #Konturok kirajzolása
cv2.imshow("Konturok",kepmas)

kontur = sorted(kontur, key = cv2.contourArea, reverse = True) [:10] #Konturok kikeresése, legjobb 10 kiválasztása
screenCnt = None
kepmas2 = kep.copy()
cv2.drawContours(kepmas2,kontur,-1,(0,255,0),3)
cv2.imshow("Top 10 kontur",kepmas2)


kivagott = './7.png'
cv2.imshow("Korbevagott", cv2.imread(kivagott))

i=7
for c in kontur:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
        if len(approx) == 4:
                screenCnt = approx

        x, y, w, h = cv2.boundingRect(c)
        new_img = kep[y:y + h, x:x + w]
        cv2.imwrite('./' + str(i) + '.png', new_img)
        i += 1
        break

cv2.drawContours(kep, [screenCnt], -1, (0, 255, 0), 3)
cv2.imshow("Rendszam tabla", kep)


rendszam = pytesseract.image_to_string(kivagott, lang='eng')

messagebox.showinfo(title="Erzekelt Rendszam", message=rendszam)
cv2.waitKey(0)
cv2.destroyAllWindows()
