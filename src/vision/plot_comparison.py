import matplotlib.pyplot as plt
from pathlib import Path

def create_comparison_plot():
    # Datos obtenidos en tus pruebas
    models = ['Random Forest', 'KNN', 'SVM', 'Simple CNN']
    accuracies = [0.8611, 0.6944, 0.7778, 0.8889] # Tus datos reales
    
    plt.figure(figsize=(10, 6))
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
    
    bars = plt.bar(models, accuracies, color=colors)
    
    # Añadir los porcentajes encima de las barras
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')

    plt.ylim(0, 1.0)
    plt.ylabel('Accuracy')
    plt.title('Model Comparison: Classical ML vs. CNN')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Guardar el gráfico
    save_path = Path("reports/model_comparison_plot.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    create_comparison_plot()