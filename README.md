# Programme de vision par ordinateur
## Utilisant DLIB et Tensorflow
*Python 3.6*

Installation des dependances:
```
pip install --user pipenv
pipenv install
```
(Le programme utilise Dlib qui nécessite un compilateur C++ comme CMake)


Il faudra aussi installer un serveur Postgres sur la machine.



Lancement du programme:

```
pipenv run python main.py
```
Il faudra renseigner les informations demandées quant au serveur Postgres afin d'y permettre une connexion.

Le programme ecoutera ensuite par défaut sur le port 5000 et il suffira de se rendre sur *localhost:5000* pour acceder au site.

Il y a 4 paths utilisables:

1. / : Permet d'envoyer une image de visage et d'y associer un nom, ce qui permettra au programme de donner le nom de la personne quand il reconnaitra son visage.
2. /test : Permet d'envoyer une image de visage pour voir si le programme le reconnaitra.
3. /webcam : La webcam va démarrer et le programme va verifier si un visage existe, si oui alors il tentera de determiner de qui il s'agit et demandera son nom s'il ne reconnait pas la personne afin de se souvenir d'elle.
4. /tracker : La webcam va demarrer pour détecter une tete puis suivre son mouvement.
