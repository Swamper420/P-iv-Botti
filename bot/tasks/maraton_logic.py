from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
import html
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from bot.config import MaratonConfig
from bot.rendering import BadgeColor, Card
from bot.storage import load_json_data, save_json_data

MARATON_STATE_FILE = "maraton_state.json"
CAMPAIGN_WINDOW_DAYS = 365

WEEKDAY_NAMES_FI: tuple[str, ...] = (
    "Maanantai",
    "Tiistai",
    "Keskiviikko",
    "Torstai",
    "Perjantai",
    "Lauantai",
    "Sunnuntai",
)


@dataclass(frozen=True)
class RelayLeg:
    number: int
    distance_km: str
    loops: str
    role: str
    danger_note: str
    tactical_tip: str


RELAY_LEGS: tuple[RelayLeg, ...] = (
    RelayLeg(
        number=1,
        distance_km="11,2 km",
        loops="Siirtymä + 1 kierros",
        role="Avaaja",
        danger_note="Lähtöhuuma huijaa vauhdin 30 s/km liian kovaksi",
        tactical_tip="Pidä ensimmäiset 2 km kurissa; nosta vauhtia vasta reitille päästyäsi.",
    ),
    RelayLeg(
        number=2,
        distance_km="7,0 km",
        loops="1 kierros",
        role="Tykkimies",
        danger_note="Lyhin osuus — ei armoa, keuhkot huutavat kolmannesta kilsasta alkaen",
        tactical_tip="Lämmittele kunnolla ennen vaihtoa ja juokse tasaisen kovaa VK2/MK-rajalla.",
    ),
    RelayLeg(
        number=3,
        distance_km="14,1 km",
        loops="2 kierrosta",
        role="Raskassarjalainen",
        danger_note="Viestin hirviöosuus: toinen kierros syö heikon peruskunnon elävältä",
        tactical_tip="Juokse eka kierros maltilla, ota juomaa pisteiltä ja raasta toinen kierros.",
    ),
    RelayLeg(
        number=4,
        distance_km="9,9 km",
        loops="Maaliosuus Ratinaan",
        role="Ankkuri",
        danger_note="Koko joukkueen kunnia ja Ratinan loppusuora lepäävät harteillasi",
        tactical_tip="Säästä yksi vaihde viimeiselle kahdelle kilometrille ja kiristä stadionille.",
    ),
)


@dataclass(frozen=True)
class PhaseDefinition:
    key: str
    name: str
    threat_label: str
    badge_color: BadgeColor
    accent_color: str
    dread_percent: int
    headlines: tuple[str, ...]
    dispatches: tuple[str, ...]
    coach_tips: tuple[str, ...]
    weekly_workouts: tuple[str, ...]  # Indexed 0 (Mon) .. 6 (Sun)


PHASES: tuple[PhaseDefinition, ...] = (
    PhaseDefinition(
        key="shadow",
        name="Peruskunto • Varjo horisontissa",
        threat_label="TASO 1: VARJO",
        badge_color=BadgeColor.BLUE,
        accent_color="#38bdf8",
        dread_percent=18,
        headlines=(
            "Kello käy jo, vaikka sohva väittää toista.",
            "Tampere Maraton 2027 odottaa pimeässä.",
            "Syyskuu 2027 tuntuu kaukaiselta — se on ansa.",
            "Tänään rakennetaan pohja, joka pelastaa sinut Ratinassa.",
        ),
        dispatches=(
            "Vielä nyt naurat, että aikaa on kuukausia. Juuri niin ajattelevat nekin, jotka elokuussa googlaavat 'voiko 14 km juosta nollatreenillä'. Käy ulkona tänään.",
            "Kolme joukkuekaveria luottaa siihen, ettet saavu viivalle rapakunnossa. Peruskunto ei kasva viikossa — se taotaan nyt, kun ketään ei vielä pelota.",
            "Asfaltti Tampereella ei kysy tekosyitä. Joka kerta kun siirrät lenkin 'huomiseen', lainaat velkaa, jonka korko maksetaan maitohappona syyskuussa 2027.",
            "Maratonviesti ei ole lenkkikerho — se on 42,195 kilometriä totuutta. Jos jalat eivät kestä helppoa hölkkää nyt, ne eivät kestä kisavauhtia silloinkaan.",
            "Talven ja kevään laiskuus näkyy armottomasti Ratinan loppusuoralla. Yksi rauhallinen lenkki tänään tekee sinusta vaarallisemman huomenna.",
        ),
        coach_tips=(
            "Pidä 80 % lenkeistä aidosti kevyenä (PK1/PK2): sinun pitää pystyä puhumaan kokonaisia lauseita.",
            "Tarkista juoksukengät nyt: kuluneet tossut tuovat penikkataudin ja rasitusvammat ennen kesää.",
            "Nosta viikkokilometrejä enintään ~10 % viikossa. Tasaisuus voittaa yksittäiset sankarilenkit.",
            "Lisää 2 x viikossa 15 min jalkaterä-, pohje- ja pakaratreeniä — jänteet vahvistuvat hitaammin kuin keuhkot.",
            "Opettele juoksemaan hitaasti: jos syke karkaa joka lenkillä, kone käy tyhjänä eikä kehity.",
        ),
        weekly_workouts=(
            "Lepo tai 25 min liikkuvuus + keskivartalo (lankut, lantionnostot, pohjenousut). Huolla kroppa uuteen viikkoon.",
            "40-45 min kevyt PK-lenkki + loppuun 4 x 15 s rennot kiihdytykset (rullaukset) hyvällä ryhdillä.",
            "35-45 min rauhallinen PK1-peruskestävyyslenkki. Syke matalalla, hengitys nenän kautta hallinnassa.",
            "30-40 min lihaskunto (kyykyt, askelkyykyt, pohkeet, pakarat, core) tai reipas kävely/sauvakävely.",
            "Täyslepo tai 25 min erittäin kevyt palauttava hölkkä. Uni on ilmainen doping.",
            "Viikon pitkä lenkki: 60-75 min tasainen PK-hölkkä (tarvittaessa kävelypätkillä). Rakenna iskunkestävyyttä.",
            "40 min kevyt palauttava lenkki tai kevyt pyöräily/uinti + 15 min venyttely ja rullaus.",
        ),
    ),
    PhaseDefinition(
        key="winter_grind",
        name="Pohjatyö • Kukaan ei tule pelastamaan",
        threat_label="TASO 2: VAROITUS",
        badge_color=BadgeColor.YELLOW,
        accent_color="#eab308",
        dread_percent=34,
        headlines=(
            "Sää on huono? Niin on myös kunto, jos jäät sisälle.",
            "Kolmososuuden 14,1 km tarkkailee sinua hiljaa.",
            "Tekosyiden varasto alkaa tyhjentyä.",
            "Kunto tehdään silloin, kun kukaan ei katso.",
        ),
        dispatches=(
            "Pimeää, märkää tai tuulista? Täydellistä. Tampereella ei jaeta sympatiapisteitä eikä kukaan kanna sinua vaihtopuomille. Lenkkarit jalkaan ja ovesta ulos.",
            "Kuvittele hetki, kun joukkuekaveri ojentaa viestin ja sykkeesi hakkaa 185:ssä jo kahden kilometrin jälkeen. Vielä ehdit estää sen katastrofin.",
            "Aika hupenee huomaamatta. Jokainen väliin jätetty viikko on suoraa tasoitusta krampeille ja pistokselle.",
            "Tänään joko rakennat aerobista moottoria tai selittelet syyskuussa, miksi 'pohje vähän kiristi'. Valinta on sinun.",
            "Jalat eivät totu 10-14 kilometrin asfaltti-iskutukseen yhdessä kuukaudessa. Nyt tehty työ on puhdasta henkivakuutusta.",
        ),
        coach_tips=(
            "Älä skippaa lämmittelyä: 5-10 min rauhallista hölkkää ja nilkkojen pyörittelyä säästää viikkojen telakalta.",
            "Testaa kadenssia (~170-180 askelta/min): lyhyempi ja rullaavampi askel säästää polvia asfaltilla.",
            "Juokse kerran viikossa mäkisellä reitillä — mäkivoima tuo ilmaista vauhtia tasaiselle.",
            "Syö kunnon hiilihydraatteja ennen yli tunnin lenkkejä ja proteiinia heti lenkin jälkeen palautumiseen.",
            "Pidä joka neljäs viikko hieman kevyempänä (-20-30 % määrää), jotta keho ehtii superkompensoitua.",
        ),
        weekly_workouts=(
            "Lepo + 20 min liikkuvuus, pohkeiden rullaus ja lonkankoukistajien avaus.",
            "45-50 min PK-lenkki, jossa mukana 6 x 20 s reippaat mäkivedot (hölkkäpalautus alas). Voimaa askeleeseen!",
            "45 min puhdas PK1/PK2-lenkki tasaisella alustalla. Keskity rentoihin hartioihin ja tasaiseen hengitykseen.",
            "35 min juoksijan voimatreeni (yhden jalan maastaveto, bulgarialainen kyykky, pohkeet, keskivartalo).",
            "Lepo tai 30 min kevyt palauttava hölkkä. Tarkista nesteytys ja yöunet.",
            "Pitkä PK-lenkki: 75-90 min rauhallisella vauhdilla. Ota mukaan vettä, jos kesto ylittää 75 min.",
            "40-45 min kevyt palauttava lenkki tai korvaava aerobinen (hiihto/pyörä/uinti) matalalla sykkeellä.",
        ),
    ),
    PhaseDefinition(
        key="engine_up",
        name="Määrä ja vauhti • Kone käyntiin",
        threat_label="TASO 3: KOHOAVA UHKA",
        badge_color=BadgeColor.YELLOW,
        accent_color="#f59e0b",
        dread_percent=50,
        headlines=(
            "Puolet tekosyistä on jo vanhentunut.",
            "Et juokse vain itsellesi — joukkue katsoo selkääsi.",
            "Kevyt hölkkä ei enää yksin riitä. Nyt nostetaan kierroksia.",
            "Ratinan asfaltti kuulee jo askeltesi puutteen.",
        ),
        dispatches=(
            "Viestijoukkue on tasan yhtä vahva kuin sen heikoin osuus. Haluatko olla se, jonka osuudella muut katsovat kelloa hiljaa ja huolestuneina?",
            "Peruskunto ei enää yksin pelasta — nyt pitää opettaa keho sietämään vauhtia, joka tuntuu epämukavalta. Epämukavuus treenissä on halpaa; romahdus kisassa on kallista.",
            "Alle puoli vuotta jäljellä. Kalenteri ei hidasta, vaikka sinä hidastaisit. Ota tämä viikko tosissasi.",
            "7 km täysillä vai 14,1 km raastoa? Molemmat rankaisevat armotta, jos VK-alue on jäänyt harjoittelematta.",
            "Jokainen kova lenkki, jonka saat tehtyä nyt, on yksi paniikkikohtaus vähemmän lähtökarsinassa.",
        ),
        coach_tips=(
            "VK1/VK2-vauhtilenkki tuntuu 'mukavan kovalta' — pystyt sanomaan vain muutaman sanan kerrallaan.",
            "Älä juokse kevyitä päiviä liian kovaa, muuten et jaksa juosta kovia päiviä oikeasti kovaa.",
            "Aloita osuuksien alustava miettiminen joukkueessa: kuka kestää 14,1 km ja kuka repii 7,0 km?",
            "Opettele juomaan vauhdista mukista tai pullosta — kisassa nestehukka iskee salakavalasti.",
            "Tarkkaile leposykettä ja jalkojen tuntemuksia: terävä kipu on stop-merkki, yleinen väsymys kuuluu asiaan.",
        ),
        weekly_workouts=(
            "Lepo + 20 min lihashuolto (pohkeet, takareidet, pakarat). Valmistele jalat huomisen vauhtitreeniin.",
            "Vauhtikestävyys: 15 min verryttely + 20 min tasavauhtinen VK1/VK2-osuus + 10 min loppuverryttely.",
            "50 min kevyt PK-lenkki. Pidä vauhti kurissa — tämän päivän tarkoitus on palauttaa ja kerätä kilometrejä.",
            "35 min kevyt lenkki + 6 x 80 m rennot rullaukset + 20 min juoksuvoima (pohkeet, pakarat, core).",
            "Lepo tai 30 min palauttava hölkkä. Syö ja nuku kunnolla huomista pitkää varten.",
            "Pitkä lenkki: 12-15 km rauhallisella PK-vauhdilla. Totuta kroppa yli 14,1 km kolmososuuden rasitukseen!",
            "45 min kevyt palauttava hölkkä sunnuntain kisarytmiin totutellen + venyttely.",
        ),
    ),
    PhaseDefinition(
        key="lactate_truth",
        name="Tehovaihe • Maitohappo odottaa",
        threat_label="TASO 4: KORKEA HÄLYTYS",
        badge_color=BadgeColor.RED,
        accent_color="#f97316",
        dread_percent=68,
        headlines=(
            "Maitohappo ei neuvottele. Se vain ottaa jalkasi.",
            "Laskuri kutistuu nopeammin kuin kunto nousee.",
            "Jos treeni tuntuu pahalta nyt, hyvä — kisassa se on tuplasti pahempaa.",
            "Ratina häämöttää. Oletko valmis vai toivotko ihmettä?",
        ),
        dispatches=(
            "Alle neljä kuukautta. Tässä vaiheessa toiveajattelu muuttuu kylmäksi todellisuudeksi: jos kynnysvauhtisi ei nouse nyt, viimeiset kilometrit Tampereella ovat pelkkää selviytymistaistelua.",
            "Kuvittele kova vastatuuli Näsijärveltä, 3 km maaliin ja reidet betonia. Ainoa keino selvitä siitä kunnialla on harjoitella sitä oloa jo tänään.",
            "Säästätkö itseäsi vieläkin? Kisapäivänä kello ei pysähdy odottamaan, kun haet henkeä. Rakenna kestävyys, jolla painat ohi muiden.",
            "Kolme muuta ihmistä seisoo vaihtoalueella odottamassa sinua. Älä anna heidän odottaa sekuntiakaan turhaan.",
            "Kuumuus, syke, asfaltti ja yleisön katseet. Nyt on aika tehdä tästä projektista totta eikä vitsiä.",
        ),
        coach_tips=(
            "Kynnysvedot (esim. 4 x 6 min VK2, 2 min hölkkäpalautus) ovat tehokkain ase 7-14 km viestiosuuksille.",
            "Harjoittele kisapäivän aamupalaa (2,5-3 h ennen lenkkiä) ja nesteytystä lämpimässä säässä.",
            "Juokse vähintään yksi lenkki viikossa samoissa kengissä ja sukissa, joilla aiot juosta maratonviestin.",
            "Opettele negatiivinen splitti: aloita rauhallisemmin ja juokse jälkimmäinen puolisko kovempaa.",
            "Kovalla treenikaudella 8 tunnin yöuni on tärkeämpi kuin yksikään lisäravinne.",
        ),
        weekly_workouts=(
            "Täyslepo tai 20 min kevyt liikkuvuus ja jalkojen huolto. Keho kehittyy levossa.",
            "Kynnysvedot: 15 min verryttely + 4 x 6 min VK2-vauhdilla (2 min kevyt hölkkäpalautus) + 10 min verryttely.",
            "50-60 min rauhallinen PK-lenkki. Helppo vauhti huuhtoo jalat ja rakentaa hiussuonistoa.",
            "45 min kevyt lenkki + 6 x 100 m kisavauhtiset rullaukset + 15 min keskivartalo ja pohkeet.",
            "Lepo tai 25-30 min erittäin kevyt palauttava hölkkä.",
            "Pitkä lenkki: 14-16 km PK-vauhdilla (viimeiset 3 km reippaammin kohti tavoitevauhtia). Ota nestettä mukaan!",
            "45 min kevyt palauttava lenkki klo 11 tienoilla (kisalähdön aikaan!) + huolellinen venyttely.",
        ),
    ),
    PhaseDefinition(
        key="critical_sharpen",
        name="Kilpailuvauhti • Tuho tai kunnia",
        threat_label="TASO 5: KRIITTINEN",
        badge_color=BadgeColor.RED,
        accent_color="#ef4444",
        dread_percent=82,
        headlines=(
            "Alle 60 päivää. Nyt ei enää haaveilla — nyt teroitetaan terä.",
            "Sydän, keuhkot, jalat — kohta mitataan kaikki.",
            "Osuusjaot lukkoon ja vauhti selkärankaan.",
            "Virheiden aika on ohi. Jokainen treeni ratkaisee.",
        ),
        dispatches=(
            "Nyt näet peilistä totuuden: kaikki aiemmat lenkit ovat pankissa, ja seuraavat viikot päättävät, juoksetko Ratinassa voimalla vai raahaudutko maaliin.",
            "Klassisin maratonviestivirhe: adrenaliini saa sinut juoksemaan ekan kilsan 25 sekuntia liian kovaa, ja puolivälissä sielu poistuu kehosta. Harjoittele tavoitevauhtia NÄILLÄ lenkeillä!",
            "Kello tikittää armottomasti. Älä riko itseäsi yliyrittämisellä, mutta älä myöskään luovuta nyt kun maali jo häämöttää.",
            "Jos sinulla on 14,1 km kolmososuus, pitkä kestävyys ja tasainen vauhti ovat elinehto. Jos sinulla on 7,0 km kakkososuus, opettele sietämään veren makua suussa.",
            "Tampereen yleisö, stadionin kaarre ja joukkueen huudot — ansaitse oikeus nauttia siitä hetkestä tekemällä työ nyt.",
        ),
        coach_tips=(
            "Lukitkaa joukkueen osuusjärjestys (11,2 / 7,0 / 14,1 / 9,9 km) ja testaa oman osuutesi tavoitevauhtia!",
            "Testaa kisavaatteet, sukat, kengät ja mahdollinen geeli/urheilujuoma — ei mitään uutta kisapäivänä.",
            "Vältä äkillisiä uusia saliliikkeitä tai maksimaalisia riuhtaisuja: vamma nyt pilaa koko kisan.",
            "Harjoittele 15 min kisalämmittelyrutiini (hölkkä + avaukset + 3 kiihdytystä), jotta et lähde kylmänä vaihtoon.",
            "Tarkista Tampere Maratonin reittikartta ja vaihtopaikat ajoissa, ettei vaihdossa tule paniikkia.",
        ),
        weekly_workouts=(
            "Lepo + 15 min kevyt venyttely ja nilkka-/pohjehuolto. Pidä jalat tuoreina.",
            "Kisavauhtiset vedot: 15 min verryttely + 5 x 1000 m oman osuutesi tavoitevauhdilla (2 min hölkkäpalautus) + 10 min verryttely.",
            "45-50 min kevyt PK-lenkki. Älä ahnehdi vauhtia välipäivänä.",
            "Kisavauhtinen tempo: 10 min verryttely + 20-25 min tasainen VK2-vauhti + 10 min verryttely.",
            "Täyslepo. Nesteytys, elektrolyytit ja aikaisin nukkumaan.",
            "Osuussimulaatio / pitkä lenkki: 12-15 km, josta lopussa 4 km oman viestiosuuden tavoitevauhdilla!",
            "35-45 min kevyt palauttava lenkki + huolellinen lihashuolto.",
        ),
    ),
    PhaseDefinition(
        key="pre_storm",
        name="Loppusilaus • Hätätila päällä",
        threat_label="TASO 6: HÄTÄTILA",
        badge_color=BadgeColor.RED,
        accent_color="#dc2626",
        dread_percent=91,
        headlines=(
            "Alle kuukausi tuomiopäivään. Kunto on melkein valmis — tai sitten ei.",
            "Älä nyt riko jalkojasi, äläkä nuku pommiin.",
            "Lähtölaukaus kuuluu jo horisontista.",
            "Hätätila: jokainen lepo, ateria ja veto lasketaan.",
        ),
        dispatches=(
            "Alle 30 päivää jäljellä! Tässä vaiheessa et voi enää taikoa uutta peruskuntoa tyhjästä, mutta voit teroittaa huippukunnon — tai pilata kaiken vammoilla ja univelalla.",
            "Jännittääkö jo? Hyvä. Pelko terävöittää aistit. Kanavoi se kuriin, uneen ja täsmällisiin treeneihin.",
            "Kohta seisot Tampereella numerolappu rinnassa ja sydän takoo rintalastaa vasten. Luota tehtyyn työhön ja pidä pää kylmänä.",
            "Viestinvaihto, kuljetukset, verryttely, varusteet — amatöörit miettivät niitä kisa-aamuna, voittajat sopivat ne nyt.",
            "Nyt karsitaan ylimääräinen kuorma ja haetaan terävä, jousimainen askel. Keho kuntoon, pää teräväksi.",
        ),
        coach_tips=(
            "Aloita määrän maltillinen kevennys (-20 %): pidä vauhtiterävyys mukana, mutta lyhennä kestoja.",
            "Sovi joukkueen kanssa tarkka aikataulu: milloin kukin saapuu vaihtopaikalle ja kuka seuraa väliaikoja.",
            "Vältä flunssia ja turhia riskejä — käsihygienia, riittävä uni ja lämmin vaatetus treenin jälkeen!",
            "Suunnittele oman osuutesi kilometrivauhdit paperille: eka kilsa 5 s/km tavoitetta rauhallisemmin.",
            "Juominen ja suola/elektrolyytit kuntoon jo päivinä ennen pitkiä tai lämpimiä lenkkejä.",
        ),
        weekly_workouts=(
            "Täyslepo + kevyt liikkuvuus. Anna mikrovaurioiden parantua.",
            "Terävöitysvedot: 15 min verryttely + 4 x 1000 m tavoitevauhtia (2,5 min kevyt palautus) + 10 min verryttely.",
            "40 min helppo PK-lenkki. Tuntuma kevyenä ja rullaavana.",
            "30 min kevyt lenkki + 6 x 15 s terävät rullaukset kisavauhdilla. Ei uuvuttavaa salitreeniä!",
            "Täyslepo. Tankkaa unta ja vettä.",
            "10-12 km kevyt lenkki, sis. 3 km tasaisella kisavauhdilla keskivaiheilla. Testaa täysi kisavarustus!",
            "30-35 min palauttava hölkkä + joukkueen taktiikkapalaveri (osuudet, vaihdot, kyydit).",
        ),
    ),
    PhaseDefinition(
        key="final_countdown",
        name="Punainen hälytys • Viimeiset päivät",
        threat_label="TASO 7: PUNAINEN HÄLYTYS",
        badge_color=BadgeColor.RED,
        accent_color="#b91c1c",
        dread_percent=97,
        headlines=(
            "PUNAINEN HÄLYTYS: PAKO EI OLE ENÄÄ MAHDOLLINEN!",
            "Nyt kevennetään — myrsky iskee Ratinaan hetkenä minä hyvänsä.",
            "Kuuntele hiljaisuutta. Se on oma sykkeesi lähtöviivalla.",
            "Viimeiset päivät. Kaikki työ on tehty — nyt ladataan ruuti kuivaksi.",
        ),
        dispatches=(
            "Kaksi viikkoa tai vähemmän! Paniikkitreenaaminen nyt ei nosta kuntoa grammaakaan — se vain syö jaloistasi paukut kisapäivältä. Luota kevennykseen!",
            "Jalat saattavat tuntua keventelyssä oudoilta ja raskailta — se on täysin normaalia ('taper madness'). Kisapäivänä ne räjähtävät eloon.",
            "Tarkista kengät, sukat, numerolapun kiinnitys, varavaatteet ja vaihtopaikka. Nolla säätöä kisa-aamuna!",
            "Syö tuttuja, hiilihydraattipitoisia aterioita, juo säännöllisesti ja mene ajoissa nukkumaan. Kisan voittaa se, joka saapuu viivalle tuoreena ja nälkäisenä.",
            "Kohta ei ole enää yhtään 'huomista'. Kohta olet sinä, kolme joukkuekaveria, 42,195 km ja Tampere.",
        ),
        coach_tips=(
            "Kevennä harjoittelun määrää 40-60 % viimeisellä viikolla — vain lyhyitä, virkistäviä pyrähdyksiä.",
            "Pakkauslista valmiiksi ajoissa: kengät, sukat, kisapaita/shortsit, hakaneulat, juomapullo, kuiva vaihtopaita.",
            "Vältä kaikkea outoa ruokaa, raskasta kuitupommia ja alkoholia 48 h ennen lähtöä.",
            "Tärkein yöuni on KAKSI yötä ennen kisaa — kisaedeltävän yön jännitys ei haittaa, jos pohjaunet ovat kunnossa.",
            "Lämmittele 15-20 min ennen omaa vaihtoasi, jotta saat koneen käyntiin heti vaihdosta.",
        ),
        weekly_workouts=(
            "Täyslepo. Kevyt kävely ja venyttely sallittu — kerää energiaa.",
            "Herättely: 12 min kevyt hölkkä + 4 x 400 m rennosti kisavauhdilla (2 min kävely/hölkkä) + 8 min verryttely.",
            "30 min erittäin kevyt hölkkä. Jos jalkoja väsyttää, lyhennä 20 minuuttiin.",
            "20-25 min kevyt hölkkä + 4 x 15 s terävät kiihdytykset. Jalat hereille ilman väsymystä.",
            "Täyslepo. Pakkaa kisakassi, tarkista vaihtoaikataulu ja tankkaa hiilihydraatteja sekä nestettä.",
            "15-20 min erittäin kevyt verryttelyhölkkä + 3 x 10 s rento kiihdytys. Varusteet ja mieli valmiiksi!",
            "KISAPÄIVÄN RYTMI: kevyt aamupala 2,5-3 h ennen lähtöä, kunnon verryttely, maltillinen alku, kaikki peliin!",
        ),
    ),
)


MILESTONE_BANNERS: dict[int, str] = {
    365: "MERKKIPAALU: TÄSMÄLLEEN VUOSI LÄHTÖLAUKAUKSEEN! Tästä alkaa matka Ratinaan.",
    300: "MERKKIPAALU: 300 PÄIVÄÄ JÄLJELLÄ! Kolmesataa mahdollisuutta voittaa sohva.",
    250: "MERKKIPAALU: 250 PÄIVÄÄ! Neljännesvuosituhatta päiviä — ja aika kiihtyy.",
    200: "MERKKIPAALU: 200 PÄIVÄÄ! Kakkonen edessä ei kestä kauaa.",
    180: "MERKKIPAALU: PUOLI VUOTTA JÄLJELLÄ! Talven selitykset päättyvät tähän.",
    150: "MERKKIPAALU: 150 PÄIVÄÄ! Peruskunnon on parempi olla jo rakenteilla.",
    100: "MERKKIPAALU: SATA PÄIVÄÄ! Kolminumeroinen luku näkyy VIIMEISEN KERRAN huomenna.",
    90: "MERKKIPAALU: 90 PÄIVÄÄ / 3 KUUKAUTTA! Tehokausi on virallisesti täällä.",
    60: "MERKKIPAALU: 60 PÄIVÄÄ! Kaksi kuukautta — ei armoa, ei oikoteitä.",
    50: "MERKKIPAALU: 50 PÄIVÄÄ! Viisikymmentä aamua ennen Ratinan asfalttia.",
    42: "MERKKIPAALU: 42 PÄIVÄÄ — YKSI PÄIVÄ PER MARATONKILOMETRI (42,195 km)!",
    30: "MERKKIPAALU: 30 PÄIVÄÄ! Yksi kuukausi. Hätätila julistettu.",
    21: "MERKKIPAALU: 21 PÄIVÄÄ — KOLME VIIKKOA! Puolimaratonin verran päiviä jäljellä.",
    14: "MERKKIPAALU: 14 PÄIVÄÄ — KAKSI VIIKKOA! Punainen hälytys, kevennys lähestyy.",
    10: "MERKKIPAALU: 10 PÄIVÄÄ! Kymmenen laskenta on alkanut.",
    7: "MERKKIPAALU: TÄSMÄLLEEN VIIKKO KISAAN! Kisa-aamu häämöttää jo!",
    5: "MERKKIPAALU: 5 PÄIVÄÄ! Pakkaa kamat, syö hyvin, suojele terveyttäsi.",
    3: "MERKKIPAALU: 3 PÄIVÄÄ! Hiilihydraatit tankkiin ja turha höntyily seis.",
    2: "MERKKIPAALU: 2 PÄIVÄÄ! TÄNÄ YÖNÄ NUKUTAAN TÄRKEIMMÄT UNET.",
    1: "MERKKIPAALU: HUOMENNA JUOSTAAN! Kengät valmiiksi, pää kylmäksi, sydän kuumaksi!",
    0: "TÄNÄÄN SE TAPAHTUU: TAMPERE MARATON 2027 MARATONVIESTI!",
}

FINAL_DAYS_OVERRIDES: dict[int, tuple[str, str]] = {
    3: (
        "3 PÄIVÄÄ JÄLJELLÄ. HIILIHYDRAATIT TANKKIIN JA HERMOT KURIIN.",
        "Kolme yötä jäljellä. Älä testaa enää mitään hullua treeniä — keskity nesteytykseen, hiilihydraatteihin ja varmista, että koko joukkue tietää vaihtopaikat.",
    ),
    2: (
        "2 PÄIVÄÄ JÄLJELLÄ. TÄMÄN ILLAN UNI ON KULLANARVOISTA.",
        "Ylihuomenna Tampereella paukkuu lähtölaukaus. Tarkista varusteet tänään, juo tasaisesti vettä ja mene ajoissa nukkumaan — huomisen jännitys ei haittaa, jos nukut tänään hyvin.",
    ),
    1: (
        "HUOMENNA JUOSTAAN. EI ENÄÄ PALUUTA.",
        "Huomenna seisot Ratinassa ja katsot totuutta silmiin. Pakkaa kisakassi valmiiksi, kiinnitä hakaneulat, syö tuttu illallinen ja laita herätys ajoissa. Huomenna jätetään kaikki Tampereen asfaltille!",
    ),
}


@dataclass(frozen=True)
class MaratonBriefing:
    event_date: date
    today: date
    days_remaining: int
    weeks_remaining: int
    weekday_name: str
    phase_key: str
    phase_name: str
    threat_label: str
    badge_text: str
    badge_color: BadgeColor
    accent_color: str
    progress_percent: float
    dread_percent: int
    milestone_banner: str | None
    headline: str
    dispatch: str
    workout_title: str
    workout_detail: str
    coach_tip: str
    relay_spotlight: RelayLeg
    is_race_day: bool = False
    is_post_race: bool = False


def parse_event_date(date_str: str) -> date:
    return date.fromisoformat(date_str.strip())


def parse_send_time(time_str: str) -> time:
    hour_str, minute_str = time_str.strip().split(":", 1)
    return time(hour=int(hour_str), minute=int(minute_str))


def get_local_now(timezone_name: str, now: datetime | None = None) -> datetime:
    tz = ZoneInfo(timezone_name)
    if now is None:
        return datetime.now(tz)
    if now.tzinfo is None:
        return now.replace(tzinfo=tz)
    return now.astimezone(tz)


def _state_path(storage_dir: Path) -> Path:
    return storage_dir / MARATON_STATE_FILE


def load_maraton_state(storage_dir: Path) -> dict[str, Any]:
    data = load_json_data(_state_path(storage_dir), default_factory=dict)
    if isinstance(data, dict):
        return data
    return {}


def save_maraton_state(storage_dir: Path, state: dict[str, Any]) -> None:
    save_json_data(_state_path(storage_dir), state)


def should_send_daily_reminder(
    config: MaratonConfig,
    storage_dir: Path,
    now: datetime | None = None,
) -> bool:
    if not config.enabled:
        return False

    local_now = get_local_now(config.timezone, now=now)
    event_date = parse_event_date(config.event_date)
    days_remaining = (event_date - local_now.date()).days

    # Stop daily reminders after race day has passed.
    if days_remaining < 0:
        return False

    send_at = parse_send_time(config.send_time)
    if local_now.time() < send_at:
        return False

    state = load_maraton_state(storage_dir)
    last_sent_date = state.get("last_sent_date")
    return last_sent_date != local_now.date().isoformat()


def mark_reminder_sent(
    storage_dir: Path,
    sent_date: date,
    now: datetime | None = None,
) -> None:
    state = load_maraton_state(storage_dir)
    state["last_sent_date"] = sent_date.isoformat()
    if now is not None:
        state["last_sent_at"] = now.isoformat()
    save_maraton_state(storage_dir, state)


def select_phase(days_remaining: int) -> PhaseDefinition:
    if days_remaining > 240:
        return PHASES[0]
    if days_remaining > 180:
        return PHASES[1]
    if days_remaining > 120:
        return PHASES[2]
    if days_remaining > 60:
        return PHASES[3]
    if days_remaining > 30:
        return PHASES[4]
    if days_remaining > 14:
        return PHASES[5]
    return PHASES[6]


def format_finnish_date(d: date) -> str:
    return f"{d.day}.{d.month}.{d.year}"


def build_maraton_briefing(
    config: MaratonConfig,
    today: date | None = None,
) -> MaratonBriefing:
    event_date = parse_event_date(config.event_date)
    if today is None:
        today = get_local_now(config.timezone).date()

    days_remaining = (event_date - today).days
    weekday_idx = today.weekday()
    weekday_name = WEEKDAY_NAMES_FI[weekday_idx]
    ordinal = today.toordinal()
    spotlight = RELAY_LEGS[ordinal % len(RELAY_LEGS)]

    if days_remaining == 0:
        return MaratonBriefing(
            event_date=event_date,
            today=today,
            days_remaining=0,
            weeks_remaining=0,
            weekday_name=weekday_name,
            phase_key="race_day",
            phase_name="KISAPÄIVÄ • Ratina kutsuu",
            threat_label="TUOMIOPÄIVÄ",
            badge_text="KISAPÄIVÄ",
            badge_color=BadgeColor.RED,
            accent_color="#ef4444",
            progress_percent=100.0,
            dread_percent=100,
            milestone_banner=MILESTONE_BANNERS[0],
            headline="TÄNÄÄN SE TAPAHTUU — RATINAN ASFALTTI ODOTTAA!",
            dispatch=(
                "Kaikki selitykset, epäilyt ja treenikilometrit päättyvät tänään. "
                "Neljä juoksijaa, 42,195 kilometriä ja yksi tavoite: vie kapula maaliin "
                "äläkä jätä yhtään voimanrippeitä reitin varteen. Nyt mennään!"
            ),
            workout_title=f"Kisapäivän protokolla ({weekday_name})",
            workout_detail=(
                "Syö tuttu aamupala 2,5-3 h ennen lähtöä, verryttele 15 min ennen omaa "
                "vaihtoasi, malta ensimmäinen kilometri ja paina maaliin asti täydellä sydämellä!"
            ),
            coach_tip=(
                "Älä lähde ekalla kilometrillä 30 s/km liian kovaa. Jätä yksi vaihde "
                "viimeiselle kahdelle kilometrille ja nauti Ratinan maalikaarteesta!"
            ),
            relay_spotlight=RELAY_LEGS[0],
            is_race_day=True,
            is_post_race=False,
        )

    if days_remaining < 0:
        days_since = abs(days_remaining)
        return MaratonBriefing(
            event_date=event_date,
            today=today,
            days_remaining=days_remaining,
            weeks_remaining=0,
            weekday_name=weekday_name,
            phase_key="post_race",
            phase_name="Jälkipyykki • Maali ylitetty",
            threat_label="SUORITETTU",
            badge_text="OHI ON",
            badge_color=BadgeColor.GREEN,
            accent_color="#22c55e",
            progress_percent=100.0,
            dread_percent=0,
            milestone_banner=None,
            headline=f"Tampere Maraton 2027 juostiin {days_since} pv sitten!",
            dispatch=(
                "Selvisitkö hengissä Ratinasta vai jäikö sielu kolmososuuden toiselle "
                "kierrokselle? Nyt palautellaan kunnolla — ja kohta aletaan miettiä "
                "ensi vuoden ennätysaikaa."
            ),
            workout_title=f"Palautuminen ({weekday_name})",
            workout_detail=(
                "Kevyt kävely, venyttely, reilu uni ja hyvä ruoka. Älä kiirehdi takaisin "
                "koville lenkeille ennen kuin jalat tuntuvat taas tuoreilta."
            ),
            coach_tip=(
                "Kirjaa ylös mikä toimi ja mikä petti — seuraavassa viestissä olette "
                "vielä kovempia."
            ),
            relay_spotlight=spotlight,
            is_race_day=False,
            is_post_race=True,
        )

    phase = select_phase(days_remaining)
    weeks_remaining = (days_remaining + 6) // 7

    elapsed = max(0, min(CAMPAIGN_WINDOW_DAYS, CAMPAIGN_WINDOW_DAYS - days_remaining))
    progress_percent = round((elapsed / CAMPAIGN_WINDOW_DAYS) * 100.0, 1)

    if days_remaining in FINAL_DAYS_OVERRIDES:
        headline, dispatch = FINAL_DAYS_OVERRIDES[days_remaining]
    else:
        headline = phase.headlines[ordinal % len(phase.headlines)]
        dispatch = phase.dispatches[ordinal % len(phase.dispatches)]

    workout_detail = phase.weekly_workouts[weekday_idx]
    coach_tip = phase.coach_tips[(ordinal // 2 + weekday_idx) % len(phase.coach_tips)]
    milestone_banner = MILESTONE_BANNERS.get(days_remaining)

    badge_text = (
        f"1 PV JÄLJELLÄ"
        if days_remaining == 1
        else f"{days_remaining} PV JÄLJELLÄ"
    )

    return MaratonBriefing(
        event_date=event_date,
        today=today,
        days_remaining=days_remaining,
        weeks_remaining=weeks_remaining,
        weekday_name=weekday_name,
        phase_key=phase.key,
        phase_name=phase.name,
        threat_label=phase.threat_label,
        badge_text=badge_text,
        badge_color=phase.badge_color,
        accent_color=phase.accent_color,
        progress_percent=progress_percent,
        dread_percent=phase.dread_percent,
        milestone_banner=milestone_banner,
        headline=headline,
        dispatch=dispatch,
        workout_title=f"Päivän treeniohje ({weekday_name})",
        workout_detail=workout_detail,
        coach_tip=coach_tip,
        relay_spotlight=spotlight,
    )


def format_maraton_caption(briefing: MaratonBriefing) -> str:
    """Short HTML caption (<1024 chars) to accompany the rendered countdown card."""
    date_str = format_finnish_date(briefing.event_date)
    if briefing.is_race_day:
        return (
            f"🚨 <b>TAMPERE MARATON 2027 — KISAPÄIVÄ ({date_str})!</b>\n"
            f"🔥 {html.escape(briefing.headline)}\n"
            f"🏃 {html.escape(briefing.workout_detail)}"
        )
    if briefing.is_post_race:
        return (
            f"🏁 <b>TAMPERE MARATON 2027 ({date_str})</b>\n"
            f"{html.escape(briefing.headline)}"
        )

    icon = "🚨" if briefing.days_remaining <= 30 else ("⚠️" if briefing.days_remaining <= 120 else "⏳")
    lines = [
        f"{icon} <b>TAMPERE MARATON 2027 — MARATONVIESTI ({briefing.days_remaining} pv jäljellä)</b>",
    ]
    if briefing.milestone_banner:
        lines.append(f"🔥 <b>{html.escape(briefing.milestone_banner)}</b>")
    lines.append(f"💀 <i>{html.escape(briefing.headline)}</i>")
    lines.append(f"🏃 <b>{html.escape(briefing.weekday_name)}:</b> {html.escape(briefing.workout_detail)}")
    return "\n".join(lines)


def format_maraton_text(briefing: MaratonBriefing) -> str:
    """Full HTML-formatted Telegram message (used directly or as card fallback)."""
    date_str = format_finnish_date(briefing.event_date)

    if briefing.is_post_race:
        return "\n".join(
            [
                f"🏁 <b>TAMPERE MARATON 2027 — MARATONVIESTI ({date_str})</b>",
                "",
                f"<b>{html.escape(briefing.headline)}</b>",
                html.escape(briefing.dispatch),
                "",
                f"🧘 <b>{html.escape(briefing.workout_title)}:</b> {html.escape(briefing.workout_detail)}",
                f"💡 <b>Vinkki:</b> {html.escape(briefing.coach_tip)}",
            ]
        )

    header_icon = "🚨" if briefing.days_remaining <= 30 else ("🔥" if briefing.days_remaining <= 120 else "⏳")
    countdown_line = (
        "🚨 <b>KISAPÄIVÄ ON TÄÄLLÄ!</b>"
        if briefing.is_race_day
        else f"⏱ <b>{briefing.days_remaining} päivää</b> ({briefing.weeks_remaining} vko) lähtölaukaukseen • {html.escape(briefing.threat_label)}"
    )

    lines = [
        f"{header_icon} <b>TAMPERE MARATON 2027 — MARATONVIESTI</b> ({date_str})",
        countdown_line,
        f"📍 Vaihe: <b>{html.escape(briefing.phase_name)}</b>",
        "",
    ]

    if briefing.milestone_banner:
        lines.append(f"⚡ <b>{html.escape(briefing.milestone_banner)}</b>")
        lines.append("")

    lines.extend(
        [
            f"💀 <b>{html.escape(briefing.headline)}</b>",
            html.escape(briefing.dispatch),
            "",
            f"🏃 <b>{html.escape(briefing.workout_title)}:</b>",
            html.escape(briefing.workout_detail),
            "",
            f"🧠 <b>Valmentajan vinkki:</b> {html.escape(briefing.coach_tip)}",
            "",
            (
                f"🎯 <b>Osuusnosto — {briefing.relay_spotlight.number}. osuus "
                f"({briefing.relay_spotlight.distance_km}, {html.escape(briefing.relay_spotlight.role)}):</b> "
                f"{html.escape(briefing.relay_spotlight.danger_note)}. "
                f"{html.escape(briefing.relay_spotlight.tactical_tip)}"
            ),
        ]
    )
    return "\n".join(lines)


def build_maraton_card(briefing: MaratonBriefing) -> Card:
    """Builds a high-contrast dark-mode sharp-cornered Card for the daily briefing."""
    date_str = format_finnish_date(briefing.event_date)
    card = Card(
        title="TAMPERE MARATON 2027",
        subtitle=f"MARATONVIESTI • {date_str} • RATINAN STADION",
        accent_color=briefing.accent_color,
        footer="42,195 km • 4 osuutta (11,2 + 7,0 + 14,1 + 9,9 km) • !maraton",
    ).set_badge(briefing.badge_text, briefing.badge_color)

    if briefing.milestone_banner:
        card.add_text(
            briefing.milestone_banner,
            bold=True,
            color=briefing.accent_color,
        )

    card.add_text(briefing.headline, bold=True, color="#f4f4f5")
    card.add_text(briefing.dispatch)

    if not briefing.is_post_race:
        days_display = "TÄNÄÄN!" if briefing.is_race_day else f"{briefing.days_remaining} pv"
        weeks_display = "0 vko" if briefing.is_race_day else f"{briefing.weeks_remaining} vko"
        card.add_key_values(
            [
                ("Lähtöön aikaa", days_display),
                ("Viikkoja jäljellä", weeks_display),
                ("Vaihe", briefing.phase_name),
                ("Hälytystaso", briefing.threat_label),
            ],
            columns=2,
        )
        card.add_progress_bar(
            label="Lähtölaukaus lähestyy (365 pv ikkuna)",
            value=briefing.progress_percent,
            max_value=100.0,
            unit="%",
            color=briefing.accent_color,
        )
        card.add_progress_bar(
            label="Paine- ja intensiteettitaso",
            value=float(briefing.dread_percent),
            max_value=100.0,
            unit="%",
            color=briefing.accent_color,
        )

    card.add_divider()
    card.add_text(briefing.workout_title.upper(), bold=True, color=briefing.accent_color)
    card.add_text(briefing.workout_detail)
    card.add_text(f"Valmentajan vinkki: {briefing.coach_tip}", muted=True)

    if not briefing.is_post_race:
        leg = briefing.relay_spotlight
        card.add_divider()
        card.add_text(
            f"OSUUSNOSTO • {leg.number}. OSUUS ({leg.distance_km} • {leg.role.upper()})",
            bold=True,
            color="#e4e4e7",
        )
        card.add_text(f"{leg.danger_note}. {leg.tactical_tip}", muted=True)

    return card
