# Algoritmo K-means IA

Este repositório contém uma implementação do algoritmo de agrupamento K-means em Python, utilizando o conjunto de dados `Mall_Customers.csv`. O objetivo é aplicar técnicas de aprendizado não supervisionado para segmentar clientes com base em seus dados demográficos e comportamentais.

## 📁 Estrutura do Projeto

- `AED2.py`: Script principal que realiza:
  - Leitura e análise exploratória dos dados
  - Normalização e pré-processamento
  - Aplicação do algoritmo K-means
  - Visualizações dos clusters
  - Cálculo da inércia, silhouette score e método do cotovelo
- `Mall_Customers.csv`: Conjunto de dados contendo:
  - `CustomerID`: ID único de cada cliente
  - `Gender`: Gênero
  - `Age`: Idade
  - `Annual Income (k$)`: Renda anual em milhares de dólares
  - `Spending Score (1-100)`: Pontuação de gasto atribuída pelo shopping

## 📊 Tecnologias Utilizadas

- Python 3.x
- Pandas
- Matplotlib
- Scikit-learn
- Plotly (caso queira gráficos interativos)

## 🚀 Como Executar o Projeto

1. **Clone o repositório:**

```bash
git clone https://github.com/soares2107/Algoritmo-K-means-IA.git
cd Algoritmo-K-means-IA
```

2. **Instale as dependências:**

```bash
pip install pandas matplotlib scikit-learn
```

3. **Execute o script:**

```bash
python AED2.py
```

## 📈 Saídas Esperadas

- Gráficos de dispersão mostrando os clusters formados
- Número ideal de clusters (usando o método do cotovelo)
- Silhouette score para avaliar a qualidade do agrupamento
- Estatísticas descritivas de cada grupo

## 📚 Conceitos Envolvidos

- **K-means**: Algoritmo que particiona os dados em K grupos com base na minimização da distância entre os pontos e o centróide de seu grupo.
- **Método do Cotovelo**: Ajuda a encontrar o número ideal de clusters analisando a inércia.
- **Silhouette Score**: Mede a coesão e separação dos clusters formados.

## 🧠 Objetivo Educacional

Este projeto foi desenvolvido com fins acadêmicos, para fixar o conteúdo da disciplina de Inteligência Artificial, aplicando aprendizado não supervisionado e análise de dados.

## 📄 Licença

Este projeto está sob a licença MIT. Consulte o arquivo `LICENSE` para mais informações.

---

Desenvolvido por [Joao Gabriel Soares](https://github.com/soares2107)