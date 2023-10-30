import matplotlib.pyplot as plt

# Métricas gerais do relatório de classificação
precision = 0.96  
recall = 0.58  
f1_score = 0.62 
accuracy = 0.93  

# Rótulos para as métricas
metrics = ['Recall','F1-Score' ,'Accuracy' ,'Precision']

# Valores das métricas
values = [recall, f1_score, accuracy, precision]

# Criação do gráfico de linhas
plt.plot(values, metrics, marker='o')
plt.xlabel('Score')
plt.ylabel('Métricas')
plt.title('Gráfico das Métricas do Relatório de Classificação')
plt.grid(True)
plt.savefig('graf_random_error_fraud_fixed_8%.png')
