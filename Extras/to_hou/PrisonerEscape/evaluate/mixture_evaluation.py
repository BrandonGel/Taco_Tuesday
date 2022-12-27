from scipy.stats import norm
from scipy.stats import multivariate_normal
import numpy as np

weights = [0.163, 0.131, 0.486, 0.112, 0.107]
means = [45.279, 55.969, 49.315, 53.846, 61.953]
covars = [0.047, 1.189, 3.632, 0.040, 0.198]


def mix_norm_cdf(x, weights, means, covars):
    mcdf = 0.0
    std = np.sqrt(covars)
    for i in range(len(weights)):
        mcdf += weights[i] * norm.cdf(x, loc=means[i], scale=std[i])
    return mcdf


# print(mix_norm_cdf(50, weights, means, covars))

def mix_multinorm_cdf_nn(nn_output, index, true_location):
    pi, mu, sigma = nn_output
    pi = pi.detach().squeeze().cpu().numpy()
    sigma = sigma.detach().squeeze().cpu().numpy()
    mu = mu.detach().squeeze().cpu().numpy()
    # print(mu.shape)
    val = mix_multinorm_cdf(true_location[index], mu[index], sigma[index], pi[index])
    return val

def mix_multinorm_cdf(x, mean, std, pi, bound_size=0.05):
    mcdf = 0
    for i in range(mean.shape[0]):
        mu = mean[i]
        s = std[i]
        
        # Calculate covariance matrix from logstd. Assuming covariance as a diagonal matrix
        var = s ** 2
        var = np.clip(var, 0.00001, 20) 
        # print(var)
        cov = np.eye(2) * var

        # print(x, mu, s, pi[i])
        upper = multivariate_normal.cdf(x + bound_size, mean=mu, cov=cov)
        lower = multivariate_normal.cdf(x - bound_size, mean=mu, cov=cov)
        mcdf += pi[i] * (upper - lower)

    return mcdf

if __name__ == "__main__":
    import numpy as np
    from scipy.stats import multivariate_normal

    mean = np.array([0.4,0.8])
    cov = np.array([[0.1, 0.3], [0.3, 1.0]])
    # x = np.random.uniform(size=(2, 2))
    x = np.array([[0.79690408, 0.1855946 ]])
    y = multivariate_normal.cdf(x, mean=mean, cov=cov)
    print(y)
    # print("Tha data and corresponding cdfs are:")
    # print("Data-------CDF value")
    # for i in range(len(x)):
    #     print(x[i],end=" ")
    #     print("------->",end=" ")
    #     print(y[i],end="\n")