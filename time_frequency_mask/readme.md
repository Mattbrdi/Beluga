# Time Frequency Mask pour BelugaForcast

    Ce markdown sert de guide pour utiliser le projet time_frequency_mask dans BelugaForcast qui permet d'établir des masques temps-fréquence
    pour mieux localiser les vocalises de beluga et pouvoir faire un certain nombre d'opérations pour affiner la localisation des beluga. 

    L'idée d'utiliser un masque temps fréquence vient du fait qu'actuellement dans la pipeline l'estimation des TDOA ce base sur des signaux
    filtrés sur une large bande de fréquence et de temps qui ne correpsond pas forcément à la ROI des vocalises de béluga. La pipeline utilise
    l'aogirhtme de détection d'Emanuel qui permet de détecter les extraits audio contenant des vocalises de bélugas mais pas d'en tirer des 
    informations plus spécifiques. Time Frequency Mask viens enc omplémant de la détection par l'algoritme d'Emanuel pour pouvoir faire  un 
    certain nombre d'opérations de processing qui permettent de fiare la localisation. Le time frequency mask est utilisable à la fois avec les
    TDOA et le beamforming et SAWADA. 

    Le masque temps fréquence permet en théorie d'améliorer significativement les performances en faible SNR. 

## I.Data generation 

    La partie data génération du repo time freuqency mask permet de générer des extraits wav bruités avec des labels de masque utilisés pour 
    entrainer le réseau qui prédis les masques temps-fréquence. Celui-ci est configurable avec les fichiers config donc le `config.json` en est 
    example 
    
### I.1 Générer des données
    Une fois le json rempli il suffit de lancer le fichier `main.py` qui génère des samples aléatoires pour le réseau. Il faut fournir en argument
    le fichier de configuration. Pour vérifier la forme des données générées, il est possible d'utiliser le mode `--showcase` qui affiche la 
    représentation temps-fréquences des fichiers générés ainsi que les masques associées sans les sauvegarder. 

### I.2 Préparer la génération de données
    Pour générer des données le data génération nécessite de fournir un certain nombre de données en entrée: 
    Tout d'abord le générateur de données se base sur une banque de données de vocalises de Bélugas débruitées fournies par Irène, qu'on labélise
    en fonction de l'objectif d'entrainement. Cette banque de whistle doit être spécifiée  dans `whistle_bank_path`.
    L'arborescence est la suivante : un dossier wav qui contietn les wav des vocalises et un fichier mask qui contient les masques temps fréquences 
    des vocalises. Attention, les fichiers wav doivent être écris sous le format whistle{i}.wav et les masques sous le format whistle{i}_mask.png.

### I.3 Récuperer les données
    Les données sont toutes sauvegardées selon la structure suivante 
        Output_dir/
        ├── wav/
        │   ├── sample_i.wav
        │   └── ...
        ├── mask/
        │   ├── sample_i.png
        │   └── ...
        └── png/
            ├── sample_i.png
            └── ...
    
    L'arborescence est créee automatiquement à partir du dossier `Output_dir` fourni. 

    Le dossier png permet simplement de vérifier que les données générées sont cohérentes. 
    

## II.Masknet
    Masknet est le réseau utilisé pour calculer les masques temps frquence dans BelugaForcast.
    Il existe deux réseaux dans masknet actuellement basés tout deux sur l'architecture U-Net: Un réseau qui ne prend que l'information fréquentielle
    et un réseau qui prend aussi en entrée l'information de phase. 

### II.1 Entrainement

### II.2 Inférence 

## III. TDOA Estimation
    La partie TDOA estimation fournis les outils pour estimer les TDOA entre hydrophones de manière à tirer parti du masque temps-fréquence. 
    L'idée ici est de pouvoir agréger des estimations de TDOAS de partie connexes de sous fichiers wav filtrés afin de n'avoir que l'information utile
    dans l'estimation.

### III.1 TDOA Non biaisée

### III.2 Blobs Post Processing

commandes:
python -m time_frequency_mask.data_generation.main --showcase --config .\time_frequency_mask\parameters\config.json
