# Imputação de Dados Ausentes — Dataset Palmer Penguins

Tratamento de 30% de valores ausentes inseridos artificialmente no dataset `penguins`, combinando **regras de domínio** e **KNN (classificação e regressão)** até zerar os NaN sem descartar a base.

Trabalho da disciplina **Mineração e Visualização de Dados** — Ciência de Dados & IA, PUC-Campinas.

---

## Problema

- Base original: 344 pinguins, 7 atributos (espécie, ilha, bico, nadadeira, massa, sexo)
- 30% das células removidas aleatoriamente (`np.random.seed(42)` → replicável)
- Objetivo: reconstruir os valores ausentes preservando a estrutura estatística da base

Descartar linhas incompletas não era opção: com 30% de remoção, quase nada sobraria.

---

## Estratégia — imputação em cascata

A ordem importa: cada etapa usa os valores recuperados na anterior.

1. **Espécie** → atributo âncora, define o perfil físico de todo o resto
2. **Ilha** → derivada da espécie (determinística para Gentoo e Chinstrap)
3. **Atributos numéricos** → KNN Regressor por espécie
4. **Sexo** → KNN Classifier por espécie (dimorfismo sexual)

---

## Como cada etapa foi resolvida

### Limpeza prévia

- Linhas com ≤ 2 atributos válidos → removidas (informação insuficiente para qualquer imputação confiável)

### 1. Espécie — regras de domínio + KNN de fallback

Análise exploratória (pairplot, boxplot, violinplot, describe por espécie) revelou limiares que separam as espécies dentro de cada ilha:

- **Biscoe** (Adelie + Gentoo) → massa > 4300g ou nadadeira ≥ 214.5mm = Gentoo; massa < 4100g ou nadadeira ≤ 199mm = Adelie
- **Torgersen** → exclusivamente Adelie, imputação direta
- **Dream** (Adelie + Chinstrap) → separadas por comprimento de bico (< 40.9mm = Adelie, > 44.1mm = Chinstrap) e nadadeira > 201mm = Chinstrap
- **Regra global** → nadadeira ≥ 214.5mm = Gentoo em qualquer ilha (cobre linhas com ilha ausente)
- **Casos restantes** → KNN Classifier (`metric="nan_euclidean"`) sobre os 4 atributos físicos
- **Zona de sobreposição** → linha ambígua descartada em vez de imputada no achismo

### 2. Ilha

- Gentoo → sempre Biscoe; Chinstrap → sempre Dream (determinístico)
- Adelie vive nas 3 ilhas → KNN Classifier com os atributos físicos

### 3. Atributos numéricos — KNN Regressor por espécie

Cada espécie tem escala física própria, então o processo roda separado para Adelie, Chinstrap e Gentoo:

- `MinMaxScaler` antes do KNN → impede que `body_mass_g` (gramas) domine a distância sobre `bill_depth_mm` (mm)
- Preditores escolhidos pela matriz de correlação de cada espécie
- K testado de 3 a 11, escolhido pelo R² no conjunto de validação
- `inverse_transform` ao final → valores voltam à escala original
- `bill_depth_mm` dos Adelie: R² máximo negativo → KNN descartado, imputação pela mediana da espécie

### 4. Sexo — KNN Classifier por espécie

Machos são maiores, mas um macho Adelie pesa como uma fêmea Gentoo. Misturar espécies quebraria o cálculo de distância, então o modelo roda por espécie.

---

## Resultado

- 0 valores ausentes na base final
- Distribuições por espécie preservadas → pairplot, boxplot e violinplot finais comparáveis aos originais
- Saída: `penguins_imputado.csv`

---

## Stack

`Python` · `pandas` · `numpy` · `scikit-learn` (KNeighborsClassifier, KNeighborsRegressor, MinMaxScaler, train_test_split) · `seaborn` · `matplotlib` · Google Colab

---

## Autores

Caio de Moraes — [GitHub](https://github.com/moraes-caio) · [LinkedIn](https://linkedin.com/in/moraes-caio)
