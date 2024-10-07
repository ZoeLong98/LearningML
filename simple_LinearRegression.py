from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def real_estate_regression(filename):
    # prepare the data
    data = pd.read_csv(filename)
    x = data['size'].values.reshape(-1, 1)
    y = data['price']

    # split as train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=98)

    # define model and
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(x_train, y_train)
    print('R2 is {}'.format(model.score(x_train, y_train)))
    y_pred = model.predict(x_test)
    print('MSE is {}'.format(mean_squared_error(y_test, y_pred)))

    # visualize the data
    plt.figure(figsize=(10, 6))
    plt.scatter(x_test, y_test, alpha=0.5)
    plt.plot(x_test, y_pred, color='red')
    plt.xlabel('Size')
    plt.ylabel('Price')
    plt.title('Real Estate Price Prediction')
    plt.show()

if __name__ == '__main__':
    real_estate_regression('datasets/real_estate_price_size.csv')