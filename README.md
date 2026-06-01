# Audio Vorm Analyse

Vorm-analyse van audio: ontleed een signaal in **stemmen (streams)** en lees de *vorm* ervan — contrapunt, fuga-operaties (retrograde, inversie), fase, en informatie-indices — culminerend in **W**, een **waarachtigheidsindex**.

**Author:** Marinus Jacobus Hogerheijde · MarinUS / soniclab · 2026 · **Docs: CC BY 4.0 · Code: MIT**

## Wat het doet

- **`analyse_way.py`** — batch-engine (`MultitimbralAnalysisSystem`): framet een WAV, splitst het spectrum in N stemmen, per stem formanten/LPC/energie/residu, tussen-stem fase + **contrapunt** + **fuga** (retrograde / inversie / parallel), en indices: comprimeerbaarheid **C**, entropie **H**, fase **P**, onafhankelijkheid **I**, en **W = (H·I)/C**. Plot de vijf curven.
- **`waarachtig/waarachtig.py`** — realtime: live microfoon → C/H/P/W op een rollende grafiek.
- **`waarachtig/waarachtigrecplotswich.py`** — realtime + opname (wav/csv/png per sessie), routing microfoon / BlackHole / simulatie.

## De W-index — eerlijke kanttekening

W is een **operationele definitie** van "waarachtigheid van vorm": vorm scoort hoog als ze *rijk* is (hoge entropie H), *echt meerstemmig* (hoge onafhankelijkheid I), en *niet tot een formule te persen* (lage comprimeerbaarheid C). Het is een **heuristiek / onderzoeksmaat — geen gevalideerde, objectieve waarheidsmeter. Geen wetenschappelijke claim.**

Let op: H en C zijn anti-gecorreleerd (entropische signalen comprimeren slecht), dus `W = (H·I)/C` versterkt de "levende" richting sterk en kan uitschieten. Een begrenzing of log-normalisatie op W is aan te raden voor stabiel gebruik.

## Installeren & draaien

```
pip install numpy matplotlib soundfile sounddevice scipy
python analyse_way.py                      # batch (WAV)
python waarachtig/waarachtig.py            # realtime (microfoon)
```

## Licentie

Documentatie/inhoud: **CC BY 4.0** (`LICENSE`). Code: **MIT** (`LICENSE-MIT`).
Naamsvermelding: *"Marinus Jacobus Hogerheijde — Audio Vorm Analyse (2026), CC BY 4.0 / MIT."*

VSM-Cantus toegepast op geluid: de fuga meetbaar gemaakt. Van het luisterend oor.
