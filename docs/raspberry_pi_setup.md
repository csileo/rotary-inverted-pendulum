# Installation d'un Raspberry Pi 3B+ pour le démo autonome

Guide pas à pas pour transformer un Raspberry Pi 3B+ tout neuf en la
« brique » qui flashe le Nano, attend que le 12V et l'USB soient branchés
(dans n'importe quel ordre) et lance la policy de balance — sans écran ni
clavier une fois que c'est fait. C'est le rôle de
[`tools/pi_demo/`](../tools/pi_demo/README.md) ; ce document couvre tout ce
qu'il y a *avant* ça : préparer la carte SD, installer les dépendances sur
le Pi, et brancher le rig physiquement.

## 1. Matériel nécessaire

- Raspberry Pi 3B+
- Carte micro SD ≥ 8 Go (16-32 Go conseillé), classe 10
- Alimentation officielle Pi : 5V / 2,5A micro-USB (une alimentation
  sous-dimensionnée cause des reboots aléatoires sous charge — c'est la
  cause n°1 de bugs fantômes sur Pi 3B+)
- Câble USB pour relier le Pi au Nano (le même connecteur que la
  photo dans `docs/BOM.md` — Micro-USB ou USB-C côté Nano selon le clone)
- Le rig assemblé avec son alimentation 12V (voir `docs/BOM.md` et
  `docs/electronics_design.md`)
- Un ordinateur pour flasher la carte SD (Windows/macOS/Linux, peu importe)
- Optionnel : écran HDMI + clavier pour le premier démarrage (sinon tout
  se fait en headless via SSH, voir étape 2)

## 2. Flasher la carte SD

1. Installer [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
   sur l'ordinateur.
2. Insérer la carte micro SD.
3. Dans Raspberry Pi Imager :
   - **Device** : Raspberry Pi 3
   - **OS** : *Raspberry Pi OS Lite (64-bit)* — pas besoin d'interface
     graphique puisque le Pi tournera headless ; ça laisse aussi plus de
     RAM libre pour PyTorch/stable-baselines3.
   - **Storage** : la carte SD insérée
4. Cliquer sur l'engrenage (⚙️, "Edit Settings" / paramètres avancés)
   **avant** d'écrire l'image, et configurer :
   - Nom d'hôte (ex. `pendulum-pi`)
   - Activer SSH → "Use password authentication" (ou clé publique si tu
     en as une)
   - Nom d'utilisateur + mot de passe
   - Wi-Fi (SSID + mot de passe + pays) si le Pi n'est pas en Ethernet
   - Fuseau horaire / disposition clavier
5. Écrire l'image, attendre la vérification, éjecter la carte.

Ces réglages évitent tout écran/clavier physique : le Pi démarre déjà en
SSH sur le bon réseau.

## 3. Premier démarrage et connexion

1. Insérer la carte SD dans le Pi, brancher l'alimentation officielle
   (pas encore le rig).
2. Attendre ~1-2 min le premier boot (expansion du système de fichiers,
   reboot automatique).
3. Se connecter :
   ```bash
   ssh <utilisateur>@pendulum-pi.local
   ```
   (ou l'IP directement si `.local`/mDNS ne résout pas sur ton réseau).
4. Mettre à jour le système :
   ```bash
   sudo apt update && sudo apt full-upgrade -y
   sudo reboot
   ```

## 4. Dépendances système

```bash
sudo apt install -y git python3-venv python3-pip curl
```

**arduino-cli** (nécessaire pour flasher le Nano depuis le Pi) :

```bash
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh
sudo mv bin/arduino-cli /usr/local/bin/
arduino-cli core update-index
arduino-cli core install arduino:avr
arduino-cli lib install FastAccelStepper AS5600
```

Vérifier que le Nano est bien vu une fois branché en USB (étape 7) avec
`arduino-cli board list`.

## 5. Cloner le dépôt

Pour un Pi dédié uniquement à faire tourner le démo (pas d'entraînement),
la branche `demo` suffit — elle ne contient que le firmware, les policies
de référence, et `tools/pi_demo/` :

```bash
git clone --branch demo --single-branch \
    https://github.com/csileo/rotary-inverted-pendulum.git
cd rotary-inverted-pendulum
```

## 6. Environnement Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Sur un Pi 3B+ (ARM, pas de GPU), `stable-baselines3` installe PyTorch en
CPU-only automatiquement — ça peut prendre plusieurs minutes, c'est normal.

## 7. Brancher le Nano et vérifier le module AS5600

1. Brancher le Nano au Pi en USB.
2. `RotaryInvertedPendulum-arduino/LowLevelServer/hw_config.h` est
   volontairement absent du dépôt (pas de valeur par défaut sûre — voir
   CLAUDE.md) : copier le profil qui correspond au module AS5600 monté sur
   ce rig depuis
   `RotaryInvertedPendulum-arduino/LowLevelServer/hw_profiles/` :
   ```bash
   cp RotaryInvertedPendulum-arduino/LowLevelServer/hw_profiles/as5600_hailege_clone.h \
      RotaryInvertedPendulum-arduino/LowLevelServer/hw_config.h
   # ou as5600_seeed.h si c'est un module Seeed d'origine — voir docs/BOM.md
   ```
3. Détecter le Nano pour ce Pi précis (à faire une seule fois, ou de
   nouveau si le Nano est remplacé par un modèle avec une autre puce
   USB-série) — **débrancher tout le reste en USB avant de lancer ceci**,
   le script suppose qu'un seul périphérique série est branché :
   ```bash
   cd tools/pi_demo
   python detect_usb_config.py
   cd ../..
   ```

## 8. Test manuel avant automatisation

Avant de tout automatiser, valider que la chaîne complète fonctionne à la
main, avec le rig sous 12V et le Nano en USB :

```bash
cd RotaryInvertedPendulum-python/src/rl
python run_policy.py --policy models/policy_working_balance.zip \
    --frame-stack 3 --duration-s 30 --port <PORT>
```

`<PORT>` : trouvé avec `arduino-cli board list` (typiquement
`/dev/ttyUSB0` sur un Pi). Si ça compile/flashe et balance, tout est en
place ; passer à l'automatisation.

## 9. Automatiser : `tools/pi_demo/run_demo.py`

C'est ce script qui rend le démo tolérant à l'ordre de branchement —
alimentation du Pi, 12V du pendule, USB du Nano peuvent être branchés dans
n'importe quel ordre et avec n'importe quel délai entre eux, il attend
chaque précondition au lieu d'échouer sur la première absente (voir
`tools/pi_demo/README.md`).

Lancer une fois à la main pour vérifier :

```bash
cd tools/pi_demo
python run_demo.py
```

Puis l'enrober dans un service `systemd` pour qu'il tourne automatiquement
à chaque démarrage du Pi (ex. après une coupure de courant) :

```bash
sudo tee /etc/systemd/system/pendulum-demo.service > /dev/null <<'EOF'
[Unit]
Description=Rotary inverted pendulum - démo autonome
After=network.target

[Service]
Type=simple
User=<utilisateur>
WorkingDirectory=/home/<utilisateur>/rotary-inverted-pendulum/tools/pi_demo
ExecStart=/home/<utilisateur>/rotary-inverted-pendulum/.venv/bin/python run_demo.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now pendulum-demo.service
```

Remplacer `<utilisateur>` par le nom d'utilisateur configuré à l'étape 2.
`run_demo.py` bloque jusqu'à la fin de la policy (`--duration-s`),
Ctrl-C/SIGTERM, ou un timeout d'attente — `Restart=on-failure` relance
proprement le cycle d'attente si besoin (ex. après un débranchement du
Nano).

Variables d'environnement optionnelles (à ajouter sous `[Service]` avec
`Environment=`, voir `tools/pi_demo/README.md`) :

| Variable | Rôle |
|---|---|
| `PENDULUM_POLICY` | Chemin vers le checkpoint `.zip`/`.pt` à charger |
| `PENDULUM_FRAME_STACK` | Doit correspondre au frame-stack d'entraînement du checkpoint |
| `PENDULUM_DURATION_S` | Durée de la balance avant arrêt |
| `PENDULUM_MOTOR_POWER_TIMEOUT_S` | Délai max d'attente du 12V avant abandon |

## 10. Montage physique sur le rig

1. Fixer le Pi près du rig (ex. sous la base), à l'abri des vibrations du
   moteur et des câbles qui pourraient s'accrocher au bras.
2. Câble USB : Pi → Nano. Longueur suffisante pour ne pas contraindre le
   bras en rotation.
3. Alimentation du Pi : **séparée** du 12V moteur — un adaptateur secteur
   micro-USB 5V/2,5A dédié, sur une prise différente ou une multiprise
   fixe (pas de branchement volant qui pourrait tirer sur les fils du
   rig).
4. Alimentation 12V du pendule : inchangée, voir `docs/electronics_design.md`
   et `docs/BOM.md`.
5. Test final : couper toutes les alimentations, puis les rebrancher dans
   un ordre totalement arbitraire (12V d'abord, ou USB d'abord, ou Pi
   d'abord, avec des délais variables entre chaque). Le service doit
   attendre patiemment et démarrer la balance dès que tout est présent —
   observer les logs avec :
   ```bash
   journalctl -u pendulum-demo.service -f
   ```

## 11. Dépannage

- **`arduino-cli board list` ne voit pas le Nano** : câble USB
  défectueux (fréquent avec les câbles "charge only"), ou puce
  USB-série (CH340 etc.) sans driver — rare sur Raspberry Pi OS qui
  inclut déjà les drivers usuels.
- **`flash_if_needed.py` refuse de compiler** : `hw_config.h` manquant —
  revenir à l'étape 7.
- **"Timed out waiting for motor power"** : vérifier que l'alimentation
  12V est bien branchée et que l'interrupteur du rig (si présent, voir
  BOM) est sur ON ; sinon, défaut de câblage driver/Vref/enable — voir
  `docs/electronics_design.md`.
- **Le service redémarre en boucle** : `journalctl -u
  pendulum-demo.service -f` pour voir l'erreur exacte ; le plus souvent
  un chemin de venv/policy incorrect dans le fichier de service.
- **Détection USB instable après changement de Nano** : relancer
  `detect_usb_config.py` (étape 7) avec un seul périphérique série
  branché.
