## How to install the pipeline

1. create conda environnement `conda create -n env_name --python=3.10 -y` Version is really important for compatibility between different modules

2. Enter pipeline `pip install -r requirements.txt`

3. `pip  install tqdm seaborn` These module needs to be included in the requirements.txt

4. `pip install dash-leaflet`

normalement ça marche après ça ptdr. Si jamais ya des problèmes avec torch conda cache clean --a / conda remove -n env_name --all 

Follow les instructions du début prier pour que ça marche. 


