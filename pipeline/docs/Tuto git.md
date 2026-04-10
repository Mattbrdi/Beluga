# Tutoriel Git

Git est un système de contrôle de version distribué qui permet de suivre les modifications apportées aux fichiers au fil du temps. Ce guide vous expliquera les bases de Git, y compris les commandes `commit`, `push`, `pull`, `clone`, ainsi que la création et la fusion de branches.

## Installation de Git

Avant de commencer, assurez-vous que Git est installé sur votre machine. Vous pouvez télécharger Git depuis [git-scm.com](https://git-scm.com/).

## Configuration de Git

Une fois Git installé, configurez votre nom d'utilisateur et votre adresse e-mail :

```bash
git config --global user.name "Votre Nom"
git config --global user.email "votre.email@example.com"
```

## Utilisation de Git dans Visual Studio Code

Visual Studio Code (VS Code) offre une intégration Git intuitive. Voici comment effectuer les opérations de base avec VS Code :

### 1. Initialiser un dépôt Git

*Version facile :* Aller sur le site github, créer un nouveau repository et le cloner avec git clone, puis rajouter du code dedans et le push.

*Version dure :*

- Ouvrez votre projet dans VS Code.
- Allez dans l'onglet "Source Control" (Contrôle de code source) dans la barre latérale.
- Cliquez sur "Initialize Repository" (Initialiser le dépôt) si ce n'est pas déjà fait.

### 2. Ajouter des fichiers à l'index

- Les fichiers modifiés apparaîtront dans la section "Changes" (Modifications).
- Cliquez sur le `+` à côté de chaque fichier pour les ajouter à l'index (ou sur le + à côté de changes directement)

### 3. Faire un commit

- Entrez un message de commit dans la zone de texte prévue à cet effet, dans le format 
`[+] Added a thing`, `[-] Removed a thing` ou `[~] Modified a thing`
- Cliquez sur l'icône `✓` pour valider le commit.

### 4. Cloner un dépôt

*Version VS Code :*

- Utilisez `Ctrl+Shift+P` pour ouvrir la palette de commandes.
- Tapez `Git: Clone` et entrez l'URL du dépôt à cloner.

*Version terminal (plus simple) :* Aller dans le terminal git et faire ``git clone <URL>``

### 5. Push vers un dépôt distant

- Après avoir commité vos modifications, cliquez sur l'icône `...` dans l'onglet Source Control.
- Sélectionnez `Push` pour envoyer vos commits vers le dépôt distant.

### 6. Pull depuis un dépôt distant

*Version VS Code :*
- Cliquez sur l'icône `...` dans l'onglet Source Control.
- Sélectionnez `Pull` pour récupérer les dernières modifications du dépôt distant.

*Version terminal :* Aller dans le dossier du repo, ``git pull``

## Version full terminal

### 1. Initialiser un dépôt Git

Pour commencer à utiliser Git dans un projet, vous devez initialiser un dépôt. Naviguez vers le répertoire de votre projet et exécutez :

```bash
git init
```

### 2. Ajouter des fichiers à l'index

Pour ajouter des fichiers à l'index (staging area), utilisez la commande suivante :

```bash
git add <nom_du_fichier>
```

Pour ajouter tous les fichiers modifiés :

```bash
git add .
```

### 3. Faire un commit

Un commit enregistre les modifications dans l'historique du dépôt. Chaque commit doit avoir un message descriptif :

```bash
git commit -m "Votre message de commit"
```

### 4. Cloner un dépôt

Pour copier un dépôt existant depuis un serveur distant, utilisez la commande `clone` :

```bash
git clone <URL_du_dépôt>
```

### 5. Push vers un dépôt distant

Pour envoyer vos commits vers un dépôt distant, utilisez la commande `push` :

```bash
git push origin <nom_de_la_branche>
```

### 6. Pull depuis un dépôt distant

Pour récupérer les modifications depuis un dépôt distant et les fusionner avec votre branche locale, utilisez la commande `pull` :

```bash
git pull origin <nom_de_la_branche>
```

## Gestion des branches

### 1. Créer une nouvelle branche

Pour créer une nouvelle branche, utilisez la commande suivante :

```bash
git branch <nom_de_la_branche>
```

Pour créer et basculer vers une nouvelle branche en une seule commande :

```bash
git checkout -b <nom_de_la_branche>
```

### 2. Basculer entre les branches

Pour basculer vers une autre branche, utilisez :

```bash
git checkout <nom_de_la_branche>
```

### 3. Fusionner des branches

Pour fusionner une branche dans la branche actuelle, utilisez la commande `merge` :

```bash
git merge <nom_de_la_branche>
```

## Conclusion

Ce tutoriel couvre les bases de Git. En maîtrisant ces commandes, vous serez en mesure de gérer efficacement vos projets avec Git. Pour des fonctionnalités plus avancées, consultez la [documentation officielle de Git](https://git-scm.com/doc).