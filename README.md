# Noch mal! RL

Reinforcement-Learning-Agent für das Würfelspiel **Noch mal!** (Schmidt Spiele, Inka & Markus Brand)
mit GPU-Training, GPU-Inferenz, Browser-Oberfläche zum Spielen gegen den Agenten bzw. zum
Simulieren echter Partien, und einem Trainings-Dashboard. Alles läuft in Docker.

```
┌──────────────┐   JSON    ┌──────────────────┐  Tensoren  ┌───────────────────────┐
│ boards/*.json│ ───────▶  │ nochmal/board.py │ ─────────▶ │ nochmal/env.py (GPU)   │
│ (beliebiges  │           │ Regionen, Züge   │            │ G Spiele × P Spieler   │
│  Spielblatt) │           └──────────────────┘            └──────────┬────────────┘
└──────────────┘                                                      │ PPO
                     ┌──────────────────┐   Checkpoints   ┌───────────▼───────────┐
   Browser  ◀──────▶ │ nochmal/server.py│ ◀────────────── │ nochmal/train.py      │
   (web/)            │ FastAPI          │                 │ Self-Play, log.jsonl  │
                     └──────────────────┘                 └───────────────────────┘
```

## Schnellstart (Docker, GPU)

Voraussetzungen: Docker + Docker Compose, NVIDIA-Treiber und das
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
(`sudo apt install nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`).

```bash
docker compose up --build web          # Web-UI auf http://localhost:8000
```

Training (eigener Container, schreibt Checkpoints nach `./checkpoints/<run>/`):

```bash
docker compose run --rm train --players 2 --games 2048 --iters 2000 --out /app/checkpoints/duo
# oder kurz:  scripts/train.sh duo --iters 2000
```

Das Training kann auch direkt aus dem Web-Dashboard (Tab **Training → Neues Training**) gestartet
und gestoppt werden; es läuft dann als Prozess im `web`-Container auf derselben GPU.

Ohne NVIDIA-Toolkit: `docker compose --profile cpu up --build web-cpu` (alles auf der CPU, deutlich langsamer).

## Ohne Docker

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu128 torch
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m nochmal.train --players 2 --out checkpoints/duo --iters 500
.venv/bin/python -m nochmal.server            # http://localhost:8000
.venv/bin/python tests/test_rules.py          # Regel-Tests
```

## Web-Oberfläche

**Spiel**
- Beliebige Sitzordnung aus Menschen und Agenten (1–6 Spieler). 1 Spieler = offizielle Solo-Regeln
  (2 Farb- + 2 Zahlenwürfel, 30 Würfe).
- Würfel automatisch oder **manuell eingeben** – so lässt sich eine echte Partie am Tisch nachspielen:
  Würfel eintippen, eigene Züge per Klick eintragen, Gegner als Agent oder ebenfalls als Mensch führen.
- Für menschliche Sitze: markierbare Felder werden hervorgehoben, die Auswahl wird live auf Gültigkeit geprüft
  (Farbe, Anzahl, Zusammenhang, Anschluss an Spalte H / eigene Kreuze, Joker-Kosten).
- **Agent fragen** zeigt die Zugverteilung des Agenten (Top-Züge mit Wahrscheinlichkeit, Pass-Wahrscheinlichkeit,
  Value-Schätzung); ein Klick auf einen Vorschlag übernimmt ihn. **Agent zieht für mich** spielt den Zug direkt.
- Rückgängig-Funktion für jeden Schritt, Verlauf, Spaltenwertung mit erster/zweiter Punktzahl, Joker, Farbbonus.
- Für die Bedienung am Tisch (auch am Handy): Würfel und Status bleiben oben fest, das Spielfeld skaliert auf die
  Bildschirmbreite, die wichtigsten Aktionen liegen in einer festen Leiste unten, Würfel werden per Tipp auf
  Farb-/Zahlchips eingegeben, ein Tipp auf einen Würfel filtert die markierbaren Felder nach Farbe/Zahl, optional
  wird ein Zug automatisch bestätigt, sobald die Auswahl gültig ist; Mitspieler erscheinen als kompakte Punkte-
  Leiste (antippen zeigt deren Blatt).

**Training**
- Dashboard im Stil moderner Broker-Apps: KPI-Kacheln (Ø Score, bester Eval-Score, Siegquote gegen die
  Heuristik, Spiellänge, Samples/s, Entropie …), großer Score-Chart mit Trainings-/Eval-Kurve und den
  Baselines Heuristik/Zufall, Detail-Charts (Value-Loss, Policy-Loss, Entropie, KL, Clip-Anteil …),
  Live-Konsole. Aktualisiert sich alle 3 s.
- Runs starten/stoppen/löschen, fortsetzen (`Fortsetzen von: duo/latest.pt`), Run-Auswahl, Vergleichschart.

## Agent & Training

- **Umgebung** (`nochmal/env.py`): vollständig vektorisiert in PyTorch. G Spiele × P Spieler werden als Tensoren
  gehalten; Würfeln, Legalitätsmasken, Zug-Anwendung und Wertung sind Matrix-Operationen. Rundenablauf:
  aktiver Spieler (Batch G) → alle passiven Spieler (Batch G·(P−1)) → Rundenwertung. Dadurch
  passen Tausende Spiele parallel auf die GPU (auf einer RTX 3080 mehrere 100k Entscheidungen/s).
- **Züge**: Alle zusammenhängenden Feldgruppen (1–5 Felder) innerhalb eines Farbblocks werden beim Laden
  des Spielblatts einmal aufgezählt (klassisches Blatt: 321). Die Aktion ist ein Index in diese Liste (+ Passen);
  die Würfelwahl folgt daraus (regulärer Würfel bevorzugt, sonst Joker).
- **Netz** (`nochmal/model.py`): spielfeld-agnostisch. Ein kleines Residual-CNN erzeugt eine Logit-Karte pro Feld;
  der Logit eines Zugs ist die Summe seiner Feld-Logits plus zustandsabhängig gewichtete Zug-Features
  (schließt Spalte/Farbe ab, Sterne, Joker-Kosten, Spielende, neue Anschlussfelder …) plus Größen-Bias.
  Globale Infos (verfügbare Würfel, Runde, Joker, Punktestand, Gegner) gehen als Broadcast-Kanäle ein.
  Dasselbe Netz läuft auf jedem Spielblatt (bis 8 Farben, beliebige Größe).
- **Algorithmus**: PPO mit Self-Play (alle Sitze dasselbe Netz), GAE, Reward = Score-Delta
  (Spalten-/Farbpunkte bei Rundenwertung, Sterne/Joker sofort), optional Margin-Term gegen die Mitspieler
  (`--margin-weight`). Eval alle n Iterationen mit greedy Policy; `best.pt` = bester Eval-Score,
  `latest.pt` = letzter Stand. Baselines: Heuristik (gewichtete Zug-Features) und Zufall.

Wichtige Parameter von `python -m nochmal.train`:

| Parameter | Bedeutung |
|---|---|
| `--board classic` | Spielblatt aus `boards/` |
| `--players 2` | Sitze pro Spiel (1 = Solo-Regeln) |
| `--games 2048 --rounds 32` | parallele Spiele, Runden pro Iteration |
| `--iters`, `--lr`, `--ent`, `--clip`, `--epochs`, `--minibatch` | PPO-Hyperparameter |
| `--margin-weight 0.3` | belohnt Vorsprung gegenüber den Gegnern |
| `--resume checkpoints/duo/latest.pt` | fortsetzen |
| `--no-amp`, `--no-compile` | bfloat16-Autocast bzw. torch.compile abschalten (beides standardmäßig an bei CUDA) |

## Arena: Agenten und Algorithmen gegeneinander

```bash
python -m nochmal.arena --agents heuristic duo/best.pt duo-blind/best.pt a2c/best.pt --players 1 2 3 4 --games 240
scripts/run_experiments.sh 150      # trainiert alle Varianten nacheinander und startet danach die Arena
```

Jede Kombination aus P Agenten spielt `--games` Partien (Sitzplätze rotieren). Ausgabe: Siegquote (Unentschieden
geteilt, Tie-Break Joker), Ø Score, Ø Rang je Spieleranzahl, bei 2 Spielern zusätzlich die Direktvergleichs-Matrix.
Im Web-UI unter **Arena** starten und ansehen; Ergebnisse liegen als JSON in `checkpoints/arena/`.

Algorithmen (`--algo`): `ppo` (Clipped Surrogate, 2 Epochen), `a2c` (GAE-Vorteil, 1 Epoche, ohne Clipping),
`vpg` (REINFORCE mit Value-Baseline, λ = 1), `dqn` (Double DQN mit Dueling-Kopf, Replay-Puffer auf der GPU,
ε-greedy über legale Züge, weiches Target-Update; Parameter `--buffer --dqn-steps --dqn-batch --eps-* --tau`,
Implementierung in `nochmal/dqn.py`). DPO/DDPO sind bewusst nicht enthalten: DPO braucht Präferenzpaare und
eine Referenz-Policy (LLM-Alignment), DDPO ist RL für Diffusionsmodelle; beides passt nicht zu einem
diskreten Würfelspiel. Dazu Varianten: `--opponent-view none` (blind), `--channels 48
--blocks 3` (kleineres Netz), `--players 3|4` (Training mit mehr Sitzen). Jeder Checkpoint speichert Algorithmus,
Regelvariante und Sichtmodus; die Agentenliste im UI zeigt sie an.

### Finetuning und Champion-Pipeline

Finetuning setzt auf einem Checkpoint auf und folgt den üblichen Best Practices für RL-Feintuning:

| Maßnahme | Flag | Zweck |
|---|---|---|
| Gewichte laden, Optimizer neu | `--finetune-from duo/best.pt` | sauberer Start, kein Adam-Momentum aus dem alten Lauf |
| kleine Lernrate mit Warmup | `--lr 1e-4 --warmup-iters 10` | keine großen ersten Schritte |
| zuerst nur die Köpfe | `--freeze-trunk-iters 5` | Repräsentation bleibt, Köpfe passen sich an |
| KL zur eingefrorenen Referenz-Policy (PPO/A2C/VPG) | `--kl-ref 0.02` | verhindert Vergessen/Kollaps (wie bei RLHF) |
| L2-SP zur Referenz (DQN) | `--l2-ref 1e-3`, Start-ε 0,2 | gleiche Idee für Q-Netze |
| Gegner-Pool früherer Checkpoints | `--opponent-pool duo/best.pt,a2c/best.pt` | gegen Self-Play-Zyklen, Lerner sitzt auf Platz 0 |
| Auswahl nach Eval | `best.pt` | nur bessere Zwischenstände werden behalten |

`--resume` setzt dagegen denselben Lauf exakt fort (inkl. Optimizer-Zustand, der jetzt in `latest.pt` liegt).

Im Dashboard: **Finetuning** → Basis-Checkpoints auswählen → „Gewählte parallel finetunen“ startet je einen Lauf
(bis `NOCHMAL_MAX_TRAININGS`, Standard 2, gleichzeitig). **Pipeline** finetuned alle Modelle, startet danach ein
Turnier und schreibt den Sieger nach `checkpoints/champion.json`; dieser Champion wird im Spiel-Setup als
Standardagent vorgeschlagen (`/api/champion`). Mehrere Läufe parallel teilen sich die GPU-Rechenzeit (die
Auslastung liegt bereits bei 99 %), der Gewinn ist also begrenzt; nützlich ist es vor allem, um Varianten
gleichzeitig laufen zu lassen.

### Blatt-Pools: jedes Spiel auf einem eigenen Blatt

`--boards random-1,random-2` und/oder `--random-boards 64 --board-seed 0` geben dem Training einen Pool von
Spielblättern. Die Umgebung hält alle Blätter als gestapelte Tensoren; jedes der parallelen Spiele trägt eine
Blatt-ID, die beim Reset neu gezogen wird, Zugtabellen und Features werden per Gather über diese ID indiziert.
Die **Eval bleibt davon unberührt** (`--eval-board`, Standard: das klassische Blatt, gleiche Spielzahl, gleicher
Referenzgegner), damit Modelle über Läufe hinweg vergleichbar bleiben. Training-Formular: „Weitere
Trainingsblätter“, „Zufallsblätter im Pool“, „Seed“.

### Eval-Tab: unabhängige Evaluation

Modelle lassen sich jederzeit erneut evaluieren (`python -m nochmal.evalrun`, Tab **Eval**): Agenten wählen,
Datensätze (klassisches Blatt, weitere Blätter, `random:N:seed` = N generierte Blätter, jedes Spiel auf einem
eigenen), Spielerzahl, Spiele, Würfel-Seed (identisch für alle Agenten), Regelvariante und **Referenzgegner**
(Heuristik, Zufall oder ein Checkpoint; die Heuristik ist für echtes Spielerniveau zu schwach, ein starker
Checkpoint oder der Champion ist der bessere Maßstab). Ergebnisse liegen als JSON in `checkpoints/evals/`.
Training ist reines Self-Play; die Heuristik dient nur als Baseline.

### Ensembles

Mehrere Checkpoints lassen sich zu einem Agenten kombinieren (Arena-Tab → Agenten markieren → „Ensemble anlegen“,
oder als Spec `ensemble:mean:duo/best.pt+duo-blind/best.pt`):

- `mean` (Soft-Voting): Mittel der Zugverteilungen, Greedy auf dem Mittel.
- `vote` (Hard-Voting): jedes Mitglied stimmt für seinen Greedy-Zug, Mehrheit gewinnt, Gleichstand nach mittlerer Wahrscheinlichkeit.

Ensembles sind in Arena, Pipeline-Champion und Spiel-Setup wie normale Agenten nutzbar (gespeichert in
`checkpoints/ensembles.json`). Inferenzkosten steigen linear mit der Mitgliederzahl.

### Geschwindigkeit

Eine Iteration = 2048 Spiele × 2 Spieler × 32 Runden = 131 072 Entscheidungen (je ein CNN-Vorwärtslauf) plus
2 PPO-Epochen darüber. Auf einer RTX 3080: ca. 3,2 s pro Iteration (41 000 Samples/s), davon ~0,5 s Rollout
(kompilierte Inferenz) und ~2,7 s Update. Hebel: `--epochs 1`, kleineres Netz (`--channels 48 --blocks 3`:
+50 % Update-Durchsatz), weniger `--games`. Die Umgebung selbst kostet nur ~3 ms pro Runde; die Zeit geht fast
vollständig ins Netz.

## Die sieben offiziellen Spielblätter

`boards/` enthält alle sieben offiziellen Blätter des Spiels (Hintergrundfarbe Orange, Grün, Rot, Schwarz = das
Blatt aus der Anleitung (`classic`), Lila, Blau, Gelb), aus Fotos pixelbasiert digitalisiert und gegen die
Designregeln geprüft: 21 Felder je Farbe, 15 Sterne mit genau einem Stern je Spalte, 30 Farbblöcke von maximal
6 Feldern. Das schwarze Blatt stimmt exakt mit dem Blatt in der Anleitung überein, was die Erkennung validiert.

- **Spiel**: Blatt im Setup wählen (offizielle Blätter stehen vorn, generierte Blätter sind mit ⚙ markiert).
- **Training auf einem Blatt**: `--board offiziell-blau` bzw. Auswahl im Formular.
- **Rotation über alle Blätter**: `--boards official` (Formular: „alle offiziellen Blätter rotierend“). Jedes der
  parallelen Spiele spielt dann auf einem eigenen Blatt, beim Reset wird neu gezogen. Die Eval bleibt auf dem
  gewählten `--board` (Standard klassisch/schwarz), damit die Läufe vergleichbar bleiben.
- **Eval-Tab**: Datensatz „alle offiziellen Blätter (rotierend)“ oder einzelne Blätter.
- Generierte Zufallsblätter (`--random-boards N`, `scripts/make_board.py`) bleiben als Zusatz verfügbar, sind aber
  keine offiziellen Layouts.

## Eigene Spielblätter

Ein Spielblatt ist eine JSON-Datei in `boards/` (siehe `boards/classic.json`): Zeilen als Farbstrings,
Sterne als `*`-Masken, Spaltenwerte, Startspalte, Farben mit Namen/Hex, Würfelseiten, Joker-Anzahl,
Spielende-Bedingung. Breite, Höhe, Farbanzahl und Regionen sind frei. Der Generator
`scripts/make_board.py --out boards/random-2.json --seed 2` erzeugt zufällige Blätter zum Testen.
Spielfelder erscheinen automatisch in UI und Training.

Das klassische Blatt wurde aus der offiziellen Anleitung (Seite 1) pixelgenau übernommen
(je 21 Felder pro Farbe, 15 Sterne, 8 Joker, Spaltenwerte 5/3 3/2 3/2 3/2 2/1 2/1 2/1 1/0 …).

## Umgesetzte Regeln

Offizielle Anleitung Schmidt Spiele 49327: Startspalte H für den allerersten Zug; Kreuze zusammenhängend in einem
Farbblock, waagerecht/senkrecht an eigene Kreuze anschließend; genau so viele Felder wie der Zahlenwürfel; nie mehr
als 5; Joker kosten je 1 Ausrufezeichen (8 Stück), 2 Joker in einem Wurf erlaubt; Passen erlaubt (aktiver Spieler
lässt dann alle 6 Würfel frei); der aktive Spieler wählt aus allen Würfeln und nimmt seine beiden weg, alle anderen
wählen nur aus den verbleibenden 4; Spaltenwertung oberer Wert für den Ersten (bei Gleichzeitigkeit alle), danach
unterer Wert; Farbbonus 5/3 analog; Spielende sofort nach dem Wurf mit dem zweiten vollen Farbbonus (Mitspieler
ziehen in diesem Wurf noch); Wertung Spalten + Farben − 2 je offenem Stern + 1 je Joker; Gleichstand: mehr übrige
Joker gewinnt. Solo: 2 + 2 Würfel, 30 Würfe.

**Zwei Regelstellen sind schaltbar** (Spiel-Setup → „Regelvarianten“, Training → `--free-rounds`,
`--start-column-always`); Standard ist die strenge Tischregel:

| Regel | Standard (streng) | Variante (Wortlaut Anleitung) |
|---|---|---|
| Spalte H | nur der allererste Zug; danach nur angrenzend an eigene Kreuze | „… oder in der Startspalte beginnend“: H darf jederzeit eine neue Gruppe starten (`--start-column-always`) |
| Würfelwahl | aktiver Spieler alle Würfel, alle anderen immer nur die 4 verbleibenden | „In den ersten 3 Würfen darf sich jeder Spieler aus allen geworfenen Würfeln … auswählen“ (`--free-rounds 3`) |

Ein Checkpoint speichert, mit welcher Variante er trainiert wurde (`rules` im Checkpoint, `run.json`).

### Gegnerblätter: sichtbar oder blind

Zweite schaltbare Variante (`--opponent-view full|none`, Training-Formular „Gegner-Spielfeld“):

- **full**: Der Agent sieht die Vereinigung aller Gegnerkreuze, das komplette Blatt des punktbesten Gegners,
  Gegner-Punktestand und Farbfortschritt.
- **none (blind)**: nur das eigene Blatt und die öffentlichen Ansagen (abgeschlossene Spalten/Farben).

Im Spiel-Setup legt „Gegnerblätter mitführen“ fest, ob Agenten die anderen Blätter sehen dürfen; ein blind
trainierter Checkpoint wird dann bevorzugt vorgeschlagen (Kennzeichnung „· blind“).

### Echte Partie ohne Eingabe der Gegnerblätter

Sitztyp **Extern** = realer Mitspieler, dessen Blatt nicht geführt wird. Für ihn trägst du pro Runde nur das
Öffentliche ein: welche beiden Würfel er als aktiver Spieler genommen hat (damit die anderen die richtigen
Restwürfel bekommen), welche Spalten/Farben er als vollständig angesagt hat (damit die Erst-/Zweitwerte stimmen)
und ob er seine zweite Farbe voll hat (Spielende). Sobald ein externer Sitz existiert, spielen alle Agenten
blind. Am Ende werden nur die mitgeführten Blätter gewertet; die Punkte der externen Spieler liest du vom
echten Blatt ab.

Vereinfachung: Der Agent wählt bei vorhandener passender Farbe/Zahl immer den regulären Würfel statt eines Jokers
(ein Joker würde nur Punkte kosten). Das absichtliche Wegnehmen „guter“ Würfel vor den Mitspielern lernt der
Agent nur insoweit, wie es sich aus seiner eigenen Zugwahl ergibt.

## Projektstruktur

```
boards/           Spielblätter (JSON)
nochmal/board.py  Laden, Farbblöcke, Zug-Aufzählung
nochmal/features.py  Board-Tensoren, Legalität, Zug-Features, Netz-Eingaben
nochmal/env.py    vektorisierte Umgebung (GPU)
nochmal/model.py  Policy/Value-Netz
nochmal/agent.py  Policies (NN, Heuristik, Zufall), Checkpoints
nochmal/train.py  PPO-Training, log.jsonl
nochmal/game.py   interaktive Einzelpartie (Undo, manuelle Würfel, Vorschläge)
nochmal/server.py FastAPI (Spiel-API, Trainings-API, statische UI)
web/              Browser-UI (ohne Build-Schritt)
tests/            Regel-Tests
```
