# Documentation detaillee — comprendre le projet de A a Z

Ce document explique chaque brique du projet en langage accessible, pour
que tu puisses le comprendre, le modifier, et surtout le **defendre a
l'oral** (entretien, soutenance) meme si tu ne l'as pas ecrit ligne par
ligne toi-meme au depart.

---

## 1. C'est quoi le probleme qu'on resout ?

On veut pouvoir poser une question en langage naturel ("A quelle
frequence faut-il nettoyer les capteurs ?") et obtenir une reponse
fondee sur le contenu de documents PDF, **sans avoir a lire les
documents soi-meme**, et avec une garantie que la reponse vient bien
des documents (pas inventee).

Deux complications concretes :

1. Tous les documents ne sont pas de "vrais" PDF numeriques — certains
   sont des **scans** (une photo/image d'une page papier) : il n'y a
   aucun texte "cliquable" dedans, juste des pixels. Il faut donc
   d'abord **lire l'image** pour en extraire le texte : c'est l'OCR.
2. On ne peut pas juste balancer tout le contenu de tous les documents
   a un modele de langage a chaque question (trop long, trop cher,
   et le modele se noierait dans l'information). Il faut d'abord
   **retrouver les passages pertinents** avant de generer une reponse :
   c'est le principe du RAG.

---

## 2. OCR — Optical Character Recognition

**Le probleme :** une page scannee est une image. Un ordinateur ne
"voit" pas des mots dedans, juste des pixels colores.

**Le principe :** un logiciel d'OCR (ici, **Tesseract**, un moteur
open-source developpe a l'origine par HP puis repris par Google)
analyse l'image, repere des formes qui ressemblent a des lettres, et
les convertit en texte reel.

**Dans le code (`src/ocr.py` et `src/loader.py`) :**

1. On essaie d'abord d'extraire le texte "normalement" (texte natif,
   via la bibliotheque PyMuPDF) — rapide et fiable quand le PDF est
   deja numerique.
2. Si une page ne contient (quasiment) pas de texte recuperable comme
   ca, on suppose que c'est une image/un scan, et on bascule sur l'OCR :
   - `pdf2image` convertit chaque page du PDF en image (comme une
     capture d'ecran de la page).
   - `pytesseract` (l'interface Python de Tesseract) lit cette image et
     en extrait le texte, en precisant la langue (francais ici, pour
     que Tesseract utilise le bon modele de reconnaissance).

**Pourquoi c'est teste reellement et pas juste "ecrit sur le papier" :**
le script `scripts/generate_test_docs.py` cree volontairement deux
versions du meme document : une "numerique" normale, et une "scannee"
(le texte est transforme en image puis sauvegarde en PDF, sans aucune
couche de texte). Le pipeline traite les deux, et on peut verifier que
le texte extrait par OCR du document scanne est bien le meme que celui
du document numerique — preuve que l'OCR fonctionne vraiment.

---

## 3. RAG — Retrieval-Augmented Generation

Le RAG, c'est une methode en 2 etapes pour repondre a une question a
partir de documents, sans tout faire relire au modele de langage :

**Etape "Retrieval" (recherche) :** on retrouve, parmi tous les
passages de tous les documents, les quelques-uns qui sont les plus
pertinents par rapport a la question posee.

**Etape "Generation" :** seulement ces quelques passages (pas tout le
corpus) sont donnes au modele de langage, avec la question, pour qu'il
formule une reponse fondee dessus.

L'avantage : la reponse est **tracable** (on sait de quel passage elle
vient) et le modele ne "brode" pas sur un sujet qu'il ne connait pas —
il repond a partir de ce qu'on lui montre.

### 3.1 Chunking (decoupage) — `src/chunking.py`

Avant de pouvoir chercher dedans, chaque document est decoupe en
petits morceaux ("chunks") de taille geree (500 caracteres ici), avec
un leger chevauchement entre deux morceaux consecutifs. Pourquoi un
chevauchement ? Pour eviter qu'une information coupee pile a la
frontiere entre deux chunks soit perdue ou incomprehensible.

### 3.2 Embeddings / vectorisation — `src/embeddings.py`

**Le probleme :** comment faire "chercher" un ordinateur par le *sens*
d'une phrase, et pas juste par les mots exacts qu'elle contient ?

**Le principe :** on transforme chaque passage de texte en une liste de
nombres (un **vecteur**), de telle sorte que deux textes au sens proche
aient des vecteurs proches (au sens mathematique : proches dans
l'espace). C'est ce qu'on appelle un **embedding**.

Dans ce projet, les vecteurs viennent du modele francais de spaCy
(`fr_core_news_md`) : chaque mot a un vecteur pre-appris (a partir de
tres grandes quantites de texte francais), et le vecteur d'un passage
est la moyenne des vecteurs de ses mots. On a verifie concretement que
ca marche : la phrase "Le chat dort sur le canape" et "Un felin se
repose sur le sofa" (memes idees, mots differents) ont une similarite
de 0.76 sur une echelle de 0 a 1, alors qu'avec une phrase sans rapport
("La fusee decolle vers la lune") la similarite tombe a 0.04.

> **Note technique importante :** la version la plus repandue de cette
> technique utilise des modeles neuronaux plus recents (type
> "sentence-transformers", telecharges depuis Hugging Face). Dans
> l'environnement de developpement utilise pour reconstruire ce projet,
> l'acces reseau a Hugging Face etait bloque (restrictions de
> securite du sandbox) — on a donc utilise les vecteurs de mots spaCy,
> telecharges depuis GitHub (autorise), qui restent de vrais embeddings
> semantiques, juste un peu plus simples. Si tu rejoues ce projet sur
> ta propre machine avec un acces internet complet, tu peux switcher
> vers sentence-transformers pour des embeddings plus fins (voir
> section 6).

### 3.3 Index vectoriel et recherche — `src/index_store.py`

Tous les vecteurs de tous les chunks sont stockes dans un **index
FAISS** (une bibliotheque open-source de Facebook/Meta specialisee
dans la recherche rapide parmi des millions de vecteurs). Quand on pose
une question, elle est elle-meme transformee en vecteur, et FAISS
retrouve les chunks dont le vecteur est le plus proche (produit
scalaire sur des vecteurs normalises = similarite cosinus).

### 3.4 Prompt engineering — `src/prompt.py`

Le "prompt" est le texte final envoye au modele de generation. Il
contient 3 parties, assemblees explicitement dans le code :

1. **Des instructions systeme** qui cadrent le comportement du modele :
   ne repondre qu'a partir des passages fournis, dire "je ne sais pas"
   plutot que d'inventer, citer le numero du passage utilise.
2. **Le contexte** : les passages retrouves a l'etape precedente.
3. **La question** posee par l'utilisateur.

C'est exactement ca, le "prompt engineering" : choisir et structurer
precisement ce qu'on envoie au modele pour orienter sa reponse (ici,
limiter les hallucinations et forcer la citation des sources).

### 3.5 Generation — `src/generate.py`

Deux backends possibles (voir aussi la section 6) :

- **Extractif (par defaut)** : ne depend d'aucune API, selectionne les
  phrases des passages retrouves qui recoupent le plus de mots avec la
  question, et les assemble en reponse. Simple, previsible, et
  testable sans connexion a un service externe.
- **LLM (optionnel)** : envoie le prompt complet a un vrai modele de
  langage via une API compatible OpenAI, pour une reponse plus
  naturelle et mieux redigee.

---

## 4. Ce qui a reellement ete teste dans cet environnement

- OCR : teste de bout en bout (voir `scripts/demo.py`), avec
  comparaison directe entre la version numerique et la version scannee
  du meme document.
- Embeddings + recherche semantique : testes et valides (exemple de
  similarite ci-dessus).
- Index FAISS : teste (ajout, recherche).
- Prompt engineering : le prompt est construit et visible (on peut
  l'inspecter via `resultat["prompt"]` dans le code).
- Generation extractive : testee de bout en bout.
- Generation via un vrai LLM (API) : **implementee mais non testee en
  conditions reelles ici**, faute d'acces reseau sortant vers les
  fournisseurs de LLM dans cet environnement de developpement. A tester
  sur ta machine avec une cle API avant de t'appuyer dessus en
  demonstration live.

---

## 5. Comment en parler honnetement (CV, lettre, entretien)

- "J'ai developpe un assistant documentaire RAG qui gere a la fois les
  documents numeriques et les documents scannes via OCR (Tesseract)."
  -> vrai, teste.
- "Le systeme vectorise les documents pour permettre une recherche
  semantique." -> vrai (embeddings spaCy + FAISS), teste.
- "J'ai travaille le prompt engineering pour limiter les
  hallucinations et forcer la citation des sources." -> vrai, le code
  le montre explicitement.
- Si on te demande en entretien "quel LLM tu utilises exactement ?" :
  reponse honnete -> "l'architecture est prevue pour brancher un LLM
  via API (OpenAI ou equivalent), avec un mode de repli extractif
  entierement local pour les tests et les environnements sans acces
  reseau. Je peux basculer sur un vrai LLM des que j'ai une cle API."
  C'est une vraie decision d'architecture (backend interchangeable),
  pas une excuse — assume-la comme telle.

---

## 6. Pour aller plus loin (sur ta propre machine, avec internet)

- Remplacer les embeddings spaCy par `sentence-transformers`
  (`all-MiniLM-L6-v2` ou un modele francais comme `camembert`) pour des
  embeddings de phrases plus fins.
- Tester reellement le backend `LLMGenerator` avec une cle API OpenAI
  ou Anthropic (il faudrait alors adapter legerement le format d'appel
  pour l'API Anthropic, qui differe du format OpenAI).
- Ajouter une evaluation quantitative (ex. un petit jeu de questions/
  reponses de reference, pour mesurer la precision du systeme).
- Gerer d'autres formats que le PDF (Word, HTML, images seules).
