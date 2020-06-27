Curso de Especialização de Inteligência Artificial Aplicada

Setor de Educação Profissional e Tecnológica - SEPT

Universidade Federal do Paraná - UFPR

---

**IAA003 - Linguagem de Programação Aplicada**

Prof. Alexander Robert Kutzke

# Implementação com Scikit-Learn

Utilizando a base de dados presente no repositório:

1. Escreva *pipeline de classificação de texto* para classificar reviews de filmes como positivos e negativos;
2. Encontre um bom conjunto de parâmetros utilizando `GridSearchCV`;
3. Avalie o classificador utilizando parte do conjunto de dados (previamente separado para testes).
4. Repita os passos 1, 2 e 3 utilizando um algoritmo de classificação diferente;
5. Escreva um pequeno texto comparando os resultados obtidos para cada algoritmo.

O texto pode ser escrito em um "Jupyter Notebook" juntamente com o código. Ou qualquer outro tipo de documento.

---

# Implementações realizadas

Foi realizado a implementação seguindo todos os passos solicitados na descrição da atividade e os resultados, utilizando a divisão de ``30%`` do dataset para o treino dos algoritmos e os resultados obtidos foram os seguintes:

### Utilizando MultinomialNB

Com o algoritimo ``MultinomialNB``, obtivemos a seguinte matriz de confusão:

Com os seguintes dados:

|         | Neg  | Pos  |
|---------|------|------|
| **Neg** | 260  | 48   |
| **Pos** | 41   | 251  |

Como demonstrado na imagem abaixo:

![multinomial](https://raw.githubusercontent.com/theusFalconde/sklearn-assignment/master/img/MultinomialNB.png "Matriz de confusão do algoritmo MultinomialNB")

> Os parâmetros escolhidos que retornaram o melhor resultado pelo ``GridSearchCV`` foram os seguintes:
> 
> * **clf__alpha**: 0.01
> * **tfidf__use_idf**: False
> * **vect__ngram_range**: (1, 2)

A matriz de reports da classificação foi a seguinte:

|                  | Precision  | Recall  | f1-score  | Support  |
|------------------|------------|---------|-----------|----------|
| **Neg**          | 0.86       | 0.84    | 0.85      | 308      |
| **Pos**          | 0.84       | 0.86    | 0.85      | 292      |
| **Accuracy**     |            |         | 0.85      | 600      |
| **Macro AVG**    | 0.85       | 0.85    | 0.85      | 600      |
| **Weighted AVG** | 0.85       | 0.85    | 0.85      | 600      |

> E a média do resultado do ``Predict`` com o ``y_test`` foi de  : **0.8516666666666667**

---

### Utilizando SGDClassifier

Com o algoritimo ``SGDClassifier``, obtivemos a seguinte matriz de confusão:

Com os seguintes dados:

|         | Neg  | Pos  |
|---------|------|------|
| **Neg** | 275  | 33   |
| **Pos** | 50   | 242  |

Como demonstrado na imagem abaixo:

![sgd](https://raw.githubusercontent.com/theusFalconde/sklearn-assignment/master/img/SGDClassifier.png "Matriz de confusão do algoritmo SGDClassifier")

> Os parâmetros escolhidos que retornaram o melhor resultado pelo ``GridSearchCV`` foram os seguintes:
> 
> * **clf__alpha**: 0.001
> * **tfidf__use_idf**: True
> * **vect__ngram_range**: (1, 1)

A matriz de reports da classificação foi a seguinte:

|                  | Precision  | Recall  | f1-score  | Support  |
|------------------|------------|---------|-----------|----------|
| **Neg**          | 0.85       | 0.89    | 0.87      | 308      |
| **Pos**          | 0.88       | 0.83    | 0.85      | 292      |
| **Accuracy**     |            |         | 0.86      | 600      |
| **Macro AVG**    | 0.86       | 0.86    | 0.86      | 600      |
| **Weighted AVG** | 0.86       | 0.86    | 0.86      | 600      |

> E a média do resultado do ``Predict`` com o ``y_test`` foi de  : **0.8616666666666667**

---

### Conclusões

Com os resultados demonstrados acima, podemos identificar que o ``GridSearchCV`` escolheu parâmetros diferentes para cada algoritmo e também nos retornou resultados diferentes. O melhor resultado obtido foi utilizando o algoritmo ``SGDClassifier``, que nos retornou a média de acerto de: ``86%``, comparado com ``85%`` do algoritmo ``MultinomialNB``.

Os dois algoritmos nos retornaram bons resultados, e podemos observar que a diferença entre os dois é muito pequena.



















