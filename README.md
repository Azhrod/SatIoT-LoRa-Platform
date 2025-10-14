# Plateforme de Simulation SatIoT-LoRa

Ce projet, réalisé dans le cadre de mes études à l'ESIEE Paris, est une plateforme web complète pour la simulation et la visualisation de la communication entre des objets connectés (IoT) et la constellation de satellites Kinéis, via la technologie LoRa.

## 🚀 Fonctionnalités principales

* **Suivi en Temps Réel :** Visualisation des satellites de la constellation sur un globe 3D interactif.
* **Prédictions de Passage :** Calcule les prochaines fenêtres de communication optimales pour un objet IoT situé n'importe où sur Terre, en se basant sur sa position géographique.
* **Simulation de Communication LoRa :** Un simulateur détaillé permettant de configurer une transmission de données multi-IoT en ajustant des paramètres clés (nombre d'objets, mode LoRa SF/LRFHSS, angle d'élévation, etc.).
* **Visualisation des Données :** Des graphiques et des statistiques en temps réel sur le taux de succès des transmissions et la latence.
* **Documentation Complète :** Le site intègre le rapport de projet détaillé qui explique les technologies, la recherche et les enjeux éthiques.

## 🛠️ Technologies utilisées

* **Frontend :** HTML, Tailwind CSS, JavaScript
* **Bibliothèques JavaScript :**
    * `globe.gl` (basé sur Three.js) pour la visualisation 3D du globe.
    * `Chart.js` pour l'affichage des graphiques de simulation.
* **Backend (Calculs Orbitaux) :**
    * Le script **`pos_sat.py`** en **Python**.
    * La librairie **`skyfield`** pour les calculs de mécanique orbitale (position des satellites, prédiction des passages).

## 📄 Rapport de Projet

Le rapport complet, qui détaille la conception, la recherche technique et l'analyse des enjeux, est directement consultable au sein du projet.

> [!NOTE]
> Les données TLE des satellites, nécessaires aux calculs, sont contenues dans le fichier `satellites_tle.txt` et utilisées par le script Python.