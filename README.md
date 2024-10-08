# Neural-network-for-image-recognition

Ce programme crée un reseau de neurones de type perceptron multicouche avec 784 neurones en couche d'entree, deux couches de 200 neurones en couches cachées et 10 neurones en couche de sortie.
Il effectue 35 itérations sur la base de données d'entraînement MNIST (reconnaissance de caractères numeriques) avant d'effectuer un test sur la base de données de test.

Compilation:
  Effectuer les commande suivantes:
    -mkdir build
    -cd build
    -cmake ..
    -make install
  L'exécutable se trouve maintenant dans le repertoire bin/ de la racine du projet.

La documentation du code source se trouve dans doc/html/index.html.
Le répertoire graphs/ contient des sous-repertoires dont le nom correspond a un certain nombre d'itérations. Dans chacun de ces repertoires se trouve un ensemble de graphiques du taux de reussite de differents réseaux après chaque itération. Les nombres indiquées dans les noms de fichiers des grahiques correpondent à la configuration du réseau. Par exemple "300_300_300_10" dans le répertoire 100 correspond à un réseau qui contient 3 couches cachées de 300 neurones et une couche de sortie de 10 neurones. La couche d'entrée contient dans tous les cas 784 neurones.
