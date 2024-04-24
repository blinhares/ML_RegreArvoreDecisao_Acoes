plt.scatter(y.index, y, s=20, edgecolor="black",
                c="darkorange", label="V. Reais")
    plt.plot(x_teste.index, x_teste['Previsao'],
            color="cornflowerblue",
            label="Previsão", linewidth=2)
    plt.xlabel("Data")
    plt.ylabel("$Close")
    plt.title("Decision Tree Regression")
    plt.legend()