import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data from csv file
df = pd.read_csv("wush_new.csv")
prices = df['Avg'].to_numpy()

# Calculate daily returns
returns = np.log(prices[1:]) - np.log(prices[:-1])
print(returns)

# Define simulation parameters
n_simulations = 100000
n_days = 70
starting_price = prices[-1]

# Average return of a stock or portfolio over a certain period of time
mean_return = np.mean(returns)
print('Average daily return is: ', mean_return)

# Calculated by taking the standard deviation of all returns
volatility = np.std(returns)
print('Standard deviation is: ', volatility)
i = 0
# Run Monte Carlo simulation
simulations = np.zeros((n_simulations, n_days))
for i in range(n_simulations):
    prices = np.zeros(n_days)
    prices[0] = starting_price
    for j in range(1, n_days):
        daily_volatility = np.random.normal(mean_return, volatility * 1.5)
        daily_return = np.exp(daily_volatility) - 1
        prices[j] = prices[j-1] * (1 + daily_return)

    simulations[i] = prices

# Calculate probabilities
prob_below_192 = np.sum(simulations[:, -1] < 195) / n_simulations
prob_above_300 = np.sum(simulations[:, -1] > 348) / n_simulations

print("Probability of price falling below 195: {:.2f}%".format(prob_below_192 * 100))
print("Probability of price rising above 348: {:.2f}%".format(prob_above_300 * 100))

y = []
days = 360/n_days
days_after = 360/n_days
i = 0
while i < n_days:
    y.append(days)
    days += days_after
    i+=1
    
# Plot results
plt.figure(figsize=(10, 6))
for i in range(n_simulations):
    plt.plot(y, simulations[i], linewidth=0.5, color='#E3242B')
plt.title("Monte Carlo Simulation for Stock Price Predictions")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()

# Extract final simulated prices
final_prices = simulations[:, -1]

# Plot histogram of final prices
plt.figure(figsize=(8, 6))
plt.hist(final_prices, bins=50, color='#E3242B')
plt.axvline(x=np.percentile(final_prices, 5), color='r', linestyle='--')
plt.axvline(x=np.percentile(final_prices, 95), color='r', linestyle='--')
plt.xlabel('Final Stock Price ($)')
plt.ylabel('Frequency')
plt.title('Monte Carlo Simulation Results for WUSH')
plt.show()

#from here, we can check the mean of all ending prices
#allowing us to arrive at the most probable ending point
mean_end_price = round(np.mean(final_prices), 2)
print("Expected price: ", str(mean_end_price))
