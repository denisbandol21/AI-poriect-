# AI-Driven Smart Battery Management System (ML + RL) 🔋☀️

Un sistem inteligent hibrid dezvoltat în **Python** pentru optimizarea managementului bateriilor în rețelele solare. Proiectul folosește **Machine Learning (ML)** pentru a prezice producția de energie solară în funcție de condițiile meteorologice și utilizează un agent de **Reinforcement Learning (RL)** pentru a lua decizii optime în timp real (Încărcare, Descărcare sau Idle).

---

### 🏗️ Arhitectura Proiectului & Fluxul AI

Sistemul este împărțit în două componente principale conectate într-un flux hibrid inteligent:

1. **Modelul de Predicție Solară (Machine Learning):**
   * Antrenat pe un set de date real de la o uzină solară (Kaggle Dataset)[cite: 3].
   * Utilizează un algoritm de **Regresie Liniară** pentru a prognoza puterea DC generată de panouri pe baza temperaturii ambientale, a temperaturii modulelor, a nivelului de iradiere solară și a markerilor temporali (ora și ziua din an)[cite: 3].
2. **Agentul de Management (Reinforcement Learning):**
   * Bazat pe algoritmul **PPO (Proximal Policy Optimization)** din framework-ul `stable-baselines3`.
   * Primește starea curentă a bateriei și cererea de energie, procesează predicția generată de componenta ML și decide acțiunea optimă pentru a maximiza eficiența rețelei și recompensele sistemului.

---

### 📂 Structura Fișierelor Recomandată

* `src/train_ml_model.py` - Script pentru descărcarea setului de date, prelucrarea datelor și antrenarea modelului ML[cite: 2, 3].
* `src/rl/train_agent.py` - Script pentru antrenarea agentului RL în mediul de simulare.
* `src/rl/battery_env.py` - Mediul personalizat de simulare a bateriei (compatibil Open AI Gym/Gymnasium).
* `main.py` - Simulatorul principal care încarcă modelele salvate și rulează simularea pas cu pas.

---

### ⚙️ Ghid de Instalare și Utilizare

#### 1. Instalarea Dependențelor
Asigură-te că ai pachetele necesare instalate în mediul tău virtual:
```bash
pip install numpy pandas scikit-learn joblib stable-baselines3 opendatasets
