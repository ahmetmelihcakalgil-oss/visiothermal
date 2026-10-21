# VisioThermal AI - Prototip

EKTEKİ DASHBOARD DOSYASINA ÇİFT TIKLAYIN SAĞ ÜSTTEN DOWNLOAD RAW FİLE SEÇENEĞİNE BASIN KISA ÖZET İÇEREN BİR SAYFA SİZİ KARŞILAYACAK 

Mevcut guvenlik kameralarindan amfi dolulugunu sayip HVAC ve aydinlatmayi
doluluga gore olcekleyen sistemin calisan prototipi (PDR Task 3).
VisioThermal AI — mevcut güvenlik kamerasından amfi doluluğunu sayıp HVAC/aydınlatmayı ölçekleyen çalışan prototip. (kurulum/internet gerekmiyor): günlük simülasyon, gerçek fotoğraflar üzerinde dedektör doğrulaması ve demo videoları tek dosyada. Kod ve README: github.com/ahmetmelihcakalgil-oss/visiothermal. Sonuç: HVAC'ta %35.9, aydınlatmada %23.5 tasarruf.

## Calistirma

    pip install -r requirements.txt          # numpy + opencv-python
    python run_demo.py                     # out/dashboard.html uretir
    python -m unittest discover -s tests -v

Istege bagli, gercek fotograflarda daha iyi sayim icin:

    pip install ultralytics                # YOLOv8n; agirliklar repoda hazir

Istege bagli, demo kaydi videosunu H.264 ile sikistirmak icin:

    pip install imageio-ffmpeg             # yoksa video VP8/WebM olarak yazilir

`out/dashboard.html` tek dosyadir, tarayicida acilir (internet gerekmez).

## GitHub'a klonlandiginda ne calisir

- `python -m unittest discover -s tests -v` -> 14 test, YOLO/ffmpeg gerekmez.
- `python run_demo.py` -> simulasyon + pano. `out/*.mp4` repoda oldugu icin
  videolar aninda gomulur (yoksa `make_demo_video.py` ~3 dk,
  `make_set_video.py` ~4 dk surende yeniden uretir; `--video none` ile atlanir).
- Pano `out/dashboard.html` commit'li oldugu icin hicbir kurulum yapmadan,
  klonladiktan hemen sonra tarayicida acilabilir.
- `yolov8n.pt` + `yolov8s.pt` ve `datasetv1/` repodadir; **YOLO kurulmadan**
  (`pip install ultralytics`) geciste `--real-detector auto` blob dedektorune
  duser ve gercek kare sayimlari duser (18/26/10 yerine ~0 gosterir).
  Hesap/enerji/kontrol kismi bundan etkilenmez.
- `imageio-ffmpeg` yoksa videolar VP8/WebM olarak yazilir; pano MIME'i
  uzantidan otomatik secer, yine oynar.
- Kodda makineye/klasore sabit yol yok, calisma dizini repo kokudur.

## datasetv1 (gercek kare dogrulamasi)

`datasetv1/` klasorunde ayni derslikten uc gercek CCTV karesi vardir
(10:18, 11:10, 11:24). `realset.py` bunlari YOLO (kuruluysa) veya blob
dedektoru ile analiz eder; sayim, HVAC seviyesi ve isaretlenmis goruntu
panoda "Gercek goruntuler uzerinde dogrulama" bolumunde gosterilir.
Sinif kapasitesi 40 koltuktur. 11:24 karesi, ikinci bir modelin
(YOLOv8s) tespitleriyle birlestirilip elle dogrulanan 26 kisiye gore
duzeltilmistir. Bu kareler simulasyona ve enerji hesaplarina dahil degildir.

    python run_demo.py --real-detector auto    # auto (varsayilan): yolo > blob

## Demo kaydi (video)

`make_demo_video.py` demonun calisir halini kare kare kaydeder: gunun
her dakikasi gercek Pipeline'dan (dedektor -> medyan filtre -> kontrol ->
MockBMS -> enerji modeli) gecirilir ve ekrandaki sayim, seviye, setpoint,
fan, isik, guc ve tasarruf degerleri o anki ciktilarin kendisidir
(ekran kaydi degil). Cikti H.264 `out/demo_run.mp4` (~30 sn, 5 MB) olur ve
`run_demo.py` tarafindan dashboard'a gomulur.

    python make_demo_video.py                 # videoyu tek basina uretir
    python run_demo.py --video rebuild        # videoyu yeniden uretip gom
    python run_demo.py --video none           # video gommeden calistir

Video (`imageio-ffmpeg` varsa H.264, yoksa VP8/WebM) simulasyon verisine
ek bir veri girmez; datasetv1'deki deneme kaydi (`demo_1020.mp4`) artik
panoya gomulmez, klasorde erisime acik kaliyor.

`make_set_video.py` ise bu gercek klip uzerinde dedektorun testini kaydeder.
`datasetv1/demo_1020.mp4` (kullanici `1020(1).mp4`) 450 kare, 3 sahnedir
(10:18, 11:24, 11:10); kesim noktalari kare farkindan, sahne etiketleri
hala fotograflarla eslestirilir. Her 10. karede YOLO calisir (11:24
sahnesinde yolov8s ile birlestirilip 26'ya sabitlenir), karar ciktilari
(seviye, setpoint, fan, isik bantlari) alt banda yazilir. Cikti
`out/set_test.mp4` ve panoda "Set videosu" bolumundedir.

    python make_set_video.py                 # set videosunu tek basina uretir

## Katmanlar (PDR'deki 4 katman)

| PDR katmani | Dosya | Durum |
|---|---|---|
| Video girdisi (RTSP) | `live.py` | yazildi, gercek kamerayla denenmedi |
| Vision AI (sayim, yogunluk haritasi) | `detector.py` | BlobDetector calisiyor; YoloDetector kodda, kurulu degil |
| Karar mantigi (edge logic) | `controller.py` | calisiyor, testli |
| Entegrasyon (BMS) | `bms.py` | MockBMS calisiyor; BACnet/Modbus taslak |
| Enerji modeli | `energy.py` | calisiyor |
| Sentetik amfi + gun senaryosu | `synth.py`, `run_demo.py` | calisiyor |

## Onemli tasarim kararlari

- Basamak yukari ani, asagi 5 dakika gecikmeli + histerezis: ders arasinda klima titremez.
- Kamera/dedektor hatasinda sistem eski %100 programa doner (fail-safe).
- Kareler diske yazilmaz; testler bunu denetler. Sadece sayi ve koordinat tutulur.
- BMS'e yalnizca degisen noktalar yazilir (change-of-value).

## Sinirlar (durust liste)

- Tum veri sentetik. Sayim dogrulugu gercek amfide yeniden olculmeli.
- HVAC guc oranlari (%20/45/70/100) ve aydinlatma gucu (2 kW) varsayimdir.
- PDR'deki %25-30 hedefi bina duzeyindedir; prototip amfi duzeyini olcer.
- Klasik dedektor sabit kamera + bos oda referansi ister. Gercekte YOLO'yu
  tavan kamerasi verisiyle ince ayarlamak gerekir.

## Sonraki adimlar

1. Bir amfiden birkac saat kayit alip (kareyi saklamadan) sayim dogrulugunu olc.
2. `YoloDetector`'i tavan/kose kamera verisiyle ince ayarla.
3. Tesis yoneticisiyle gercek BACnet nesne listesini al, `BACnetAdapter`'i yaz.
4. Guc oranlarini bir amfinin alt sayacindan olcerek kalibre et.
