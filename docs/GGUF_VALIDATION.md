# DirectAMD GGUF dönüşümü ve gerçek model doğrulaması

18 Kasım 2026. Yalnız yerel geliştirme ve test yapıldı. GitHub'a push, model
indirme/eğitme veya GPU ayarı değişikliği yapılmadı. Hesaplama AMDGPU/DRM
üzerinden mevcut DirectAMD çekirdekleriyle yürütüldü.

## Sonuç

Qwen2.5-1.5B-Instruct modelinin yerel Q4_K_M GGUF dosyası FP16 SafeTensors'a
dönüştürüldü. RX 570 ve RX 580, gerçek eğitilmiş ve kuantizasyondan çözülmüş
bu ağırlıklarla aynı Türkçe yanıtı üretti.

Soru `Türkiye'nin başkenti neresidir?`

Ham tamamlanmış yanıt

> Türkiye'nin başkenti Ankara'dır.

8 token sınırındaki ham çıktı `Türkiye'nin başkenti Ankara'd` oldu. 64 token
üst sınırıyla yapılan tur her kartta EOS dahil 11 tokenla normal sonlandı.
Bunlar 64 token üretilmiş testler değildir. Bu görevde gerçek modelle tam
64 token uzunluğunda bir devam doğrulanmadı. Sentetik 64 token testi ayrıca geçti.

## Kaynak ve dosya yapısı

Kaynak `/home/cgu/Belgeler/qwen2.5-1.5b-instruct-q4_k_m.gguf`.
GGUF sürümü 3, kuantizasyon sürümü 2, veri başlangıcı 5.950.496 bayt.
Kaynak boyutu 1.117.320.736 bayt. Kaynak salt okunur açıldı; ikinci tam dönüşüm
öncesi/sonrası SHA-256 aynı kaldı.

`6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`

339 tensorun 169'u Q4_K, 29'u Q6_K, 141'i F32. Dosyada desteklenmeyen tensor
türü yok. Dönüştürücü ayrıca F16'yı destekler; diğer türleri açık hatayla reddeder.
Q4_K_M bir model kuantizasyon önayar adıdır, tek tensor veri türü değildir.

Qwen2 yapılandırması 28 katman, gizli boyut 1536, MLP boyutu 8960, 12 Q başlığı,
2 KV başlığı, başlık boyutu 128, sözlük 151936, bağlam üst sınırı 32768,
RoPE tabanı 1000000 ve epsilon yaklaşık 1e-6. Embedding ve çıktı matrisi
GGUF içinde farklı kuantizasyonlarla ayrı tutulur ve dönüştürücü ikisini de korur.
Depolanan parametre sayısı 1.777.088.000; ağırlıklar yanlış biçimde paylaşılmadı.

BPE sözlüğü, 151387 merge, özel tokenlar ve chat template GGUF metadata'dan
çıkarıldı. 271 UNUSED sözlük satırının indeksleri korundu. Türkçe soru,
Türkçe büyük/küçük harfler, rakamlar, Çince, emoji, aksan birleşimleri,
sekme ve satır sonları encode/decode roundtrip testlerini geçti.
Tokenizer başka bir uygulamayla token-token karşılaştırılmadı. Tool-call
çalıştırma ve genel Jinja chat template yorumlama mevcut CLI kapsamında değil.

## Dönüşüm ve bellek

Son çıktı `/home/cgu/work/models/qwen2.5-1.5b-directamd-fp16`.
Dosyalar `model.safetensors`, `config.json`, `tokenizer.json`,
`tokenizer_config.json`, `generation_config.json`, `conversion-report.json`.

Son tam dönüşüm 28.292 saniye sürdü; süreç tepe RSS'i 76,820 KiB oldu.

FP16 ağırlık verisi 3.554.176.000 bayt; SafeTensors başlığı dahil dosya
3.554.214.488 bayt (yaklaşık 3,31 GiB). GPU FP32 ham ağırlıkları 7.108.352.000
bayt (yaklaşık 6,62 GiB). Koruma ve sayfa hizalamasıyla yalnız ağırlık BO'ları
7.109.509.120 bayt. Test bağlamı 160 token; KV cache BO'ları 9.633.792 bayt.
Ağırlık ve cache toplam VRAM tahsisi 7.119.142.912 bayt; aktivasyonlarla tepe
izlenen tensor tahsisi 7.120.412.672 bayt. Bu sayaç küçük program/IB BO'larını içermez.

Model FP32 olarak her iki 8 GiB karta sığdı. CPU-görünür VRAM penceresi 256 MiB
olduğundan LLM ağırlıkları CPU'ya eşlenmeden ayrılıyor. FP16 dönüşümü C++ içinde
262144 elemanlık tamponlarla yapılıp GTT üzerinden mevcut GPU add-zero yoluyla
VRAM'e aktarılıyor. Giriş/çıkış korumaları da GPU üzerinden okunuyor. Genel
BufferObject çağrılarının önceki host-access varsayılanı korunuyor.

Gerçek üretim süreçlerinde tepe RSS RX 570 için 94.948 KiB, RX 580 için
120.940 KiB idi. Ayrı, mmap kullanan çift duyarlıklı CPU/GPU doğrulama süreci
500.972 KiB'ye ulaştı. Örneklenen sistem kullanılabilir RAM'i en az
3.734.068 KiB idi; mevcut swap kullanımı bu süreçlere özel tüketim değildir.
Global sysfs VRAM örneklerinde RX 570 tepe 7.182.843.904, RX 580 tepe
8.257.302.528 bayt görüldü. Bu global değerler masaüstü ve diğer süreçleri de içerir.

Model yükleme ana testte RX 570 için 23,50 saniye, RX 580 için 18,62 saniye sürdü.
Yükleme süresi aşağıdaki ilk token sürelerine dahil değildir. Her GPU için
7400 MiB tensor bütçesi kullanıldı. CLI varsayılanı 0 ile otomatik bütçe olup
fiziksel VRAM'den 256 MiB düşer; model + KV + scratch kontrolü yine yapılır.

## Gerçek GPU üretim ölçümleri

| Kart / node | İstenen üst sınır | Üretilen token | İlk token | Decode token/s | Toplam üretim |
|---|---:|---:|---:|---:|---:|
| RX 570 / D129 | 8 | 8 | 116.283 s | 0.3374 | 137.032 s |
| RX 570 / D129 | 64 | 11 | 117.267 s | 0.3365 | 146.985 s |
| RX 580 / D128 | 8 | 8 | 159.100 s | 0.2439 | 187.796 s |
| RX 580 / D128 | 64 | 11 | 139.385 s | 0.2856 | 174.394 s |

Her tur aynı model nesnesinde cache konumu sıfırlanarak bağımsız istek olarak
başlatıldı; ağırlıklar yeniden yüklenmedi. 64 sınırındaki 11 tokenın sonuncusu
EOS 151645'tir. İki kartın token ID dizileri de eşleşti. İlk token süresi 40
prompt tokenının seri prefill'ini içerir. İki kartın çalışmalarının bir bölümü
aynı anda yürüdü ve RX 580 masaüstünü de çalıştırdı. Bunlar izole performans
benchmark'ları değildir; frekans/voltaj ayarı değiştirilmedi.

## CPU referansı ve kuantizasyon doğrulaması

Model genelinden seçilmiş 6048 blok / 912063 eleman, mevcut bağımsız GGML CPU
`dequantize_row_q4_K` / `dequantize_row_q6_K` fonksiyonlarıyla karşılaştırıldı.
F32 için Python struct referansı kullanıldı. FP16 yuvarlaması sonrası sıfır bit
uyuşmazlığı bulundu. Aynı örneklerin yazılmış SafeTensors dosyasındaki baytları
da referansla eşleşti. Tüm bloklar bağımsız çözücüyle karşılaştırıldı iddiası yoktur.

İki tam dönüşümün bütün SafeTensors dosyalarının SHA-256 değerleri aynı oldu
`37318cf4d1c0fcb066d1f61baf5e24f226a0383baf877e81752443c9c638e9f6`.
Kuantizasyonun kaybettiği bilgi geri kazanılmaz; çıktı orijinal eğitimin
kuantize edilmemiş FP16 checkpoint'i değildir.

Her kartta ilk iki gerçek prompt tokenı için tüm 28 katmanın kritik ara
çıktıları ve logits, ayrı çift duyarlıklı CPU Transformer referansıyla
karşılaştırıldı. Kart başına 286 tensor ve 668416 skaler karşılaştırma geçti;
her iki adımda CPU/GPU argmax tokenları aynıydı. En büyük mutlak fark
0,0155813453; katman 26 çıktısında GPU 103,517578125, CPU 103,501996780.
Kullanılan tolerans `abs(error) <= 0.002 + 0.002*abs(reference)`.
`abs(reference)>1e-4` için en büyük bağıl hata 1,24202212 oldu; sıfıra yakın
iptallerde bağıl hata yüksek olduğundan birleşik tolerans kullanıldı.
Bu doğrulama üretilen bütün tokenların çift duyarlıklı referansla
karşılaştırıldığı anlamına gelmez.

## Regresyonlar, gönderimler ve fence

CTest 7/7 hedef geçti. Yeni dönüştürücü hedefi 8 unittest içeriyor; Q4 ölçek
paketleme, Q6 işaretli ölçek/sıra, FP16 uç değerler, nonfinite/overflow,
geçersiz tür ve blok uzunluğu, ad eşleme, bozuk metadata, tensor sınırları
ve tam küçük GGUF dönüşümü kontrol edildi. Yarım çıktı yüklenmesi reddedildi,
kaynak hash'i değişmedi. Var olan host LLM testleri de geçti.

Sekiz GPU işlemi her kartta 103/103 vaka, toplam 206/206 geçti. Kart başına
vector-add 2, vector-mul 15, GEMV 9, GEMM 10, Softmax 16, RMSNorm 16,
RoPE 20 ve SiLU 15. Güncellenen VRAM yolu üzerinden sentetik Transformer
self-testleri her iki kartta geçti; 64 token CPU/GPU ID eşleşmesi, EOS,
cache replay ve sınır/koruma testleri dahil. Sentetik sayısal hata ayrı
raporlandı; gerçek model ölçümleriyle karıştırılmadı.

Ana gerçek model testinde RX 570 714096 başarılı gönderim/fence, RX 580
728350 başarılı gönderim/fence bildirdi. RX 580 sayısı iki tokenlık CPU
karşılaştırmasını da içerir. RX 570'te iki ayrı ek referans doğrulamasının
her biri yükleme dahil 37031 gönderim/fence bildirdi. Tamamlanan gerçek
model doğrulama süreçleri toplamı 1516508 gönderim ve aynı sayıda fence.
İki sentetik self-test 53304 ve regresyonlar 206 ek gönderim/fence ile
sayılmış tamamlanan test toplamı 1570018'dir. Yavaş ilk yükleme denemesi
kontrollü kesildi; bu toplamda yer almaz.

Bu sayılar başarılı CS ve WAIT dönüşlerinden sonra artan uygulama sayaçlarıdır;
bu görevde milyonlarca ioctl bağımsız strace ile tekrar sayılmadı. Sayılar
bellek aktarımı ve guard okumalarını da içerir. Her gönderim fence bekledi,
koruma bozulması, fence timeout'u veya hesaplama hatası görülmedi.

## Sınırlar ve başarısız ilk denemeler

İlk dönüşüm UNUSED token türü 5'i açık hatayla reddetti; destek eklenip
metadata indeksleri korunarak tam dönüşüm yeniden yapıldı. Yavaş Python FP16
çözme yolundaki ilk model yüklemesi kontrollü durduruldu ve C++ parça dönüşümü
ile yeniden çalıştırıldı. Bunlar başarılı test olarak sayılmadı. Geçici hatalı
çıktı temizlendi; kaynak GGUF ve önceden çalışan GPU işlemleri korunuyor.

Tam 64 token gerçek devam, tam 32768 bağlam, çok kullanıcılı eşzamanlı istekler,
farklı GGUF mimarileri, diğer kuantizasyon türleri ve tool-call çalıştırma
bu görevde doğrulanmadı. Model yanıtı doğru; mevcut senkron skaler GEMV yolu
üretim kullanımı için yavaştır. Hiçbir performans optimizasyonu iddiası yoktur.

## Yeniden çalıştırma

```bash
/home/cgu/work/DirectAMD/build-llm/directamd-llm \
  --model /home/cgu/work/models/qwen2.5-1.5b-directamd-fp16 \
  --device /dev/dri/renderD128 \
  --prompt "Türkiye'nin başkenti neresidir?" \
  --context-length 160 --max-tokens 64 --memory-mib 7400
```

Dönüştürücü ve tekrarlanabilir CPU/GPU doğrulama komutları proje içindeki
`docs/GGUF_CONVERSION.md` belgesindedir. GPU kullanılan testler D128/D129
cihaz erişimi gerektirir. Depodaki önceki yerel değişiklikler korunmuş,
commit ve uzak depo değişikliği yapılmamıştır.
