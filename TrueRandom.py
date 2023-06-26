import requests
import math

class TrueRandom:
    
    @staticmethod
    def get_true_random_numbers(n=1):
        url = f"https://qrng.anu.edu.au/API/jsonI.php?length={n}&type=uint16"
        response = requests.get(url)
        return response.json()['data'] if response.status_code == 200 else None

    @staticmethod
    def randint(a, b):
        random_numbers = TrueRandom.get_true_random_numbers(1)
        if random_numbers:
            scale = random_numbers[0] / 0xFFFF
            return a + int(scale * (b - a + 1))

    @staticmethod
    def random():
        random_numbers = TrueRandom.get_true_random_numbers(1)
        if random_numbers:
            return random_numbers[0] / 0xFFFF

    @staticmethod
    def choice(sequence):
        if sequence:
            index = TrueRandom.randint(0, len(sequence) - 1)
            return sequence[index]

    @staticmethod
    def shuffle(sequence):
        for i in reversed(range(1, len(sequence))):
            j = TrueRandom.randint(0, i)
            sequence[i], sequence[j] = sequence[j], sequence[i]

    @staticmethod
    def uniform(a, b):
        return a + (TrueRandom.random() * (b - a))

    @staticmethod
    def randrange(start, stop=None, step=1):
        if stop is None:
            start, stop = 0, start
        width = stop - start
        n, rem = divmod(width, step)
        if rem:
            n += 1
        return start + TrueRandom.randint(0, n-1) * step

    @staticmethod
    def sample(population, k):
        if isinstance(population, set):
            population = list(population)
        n = len(population)
        if not 0 <= k <= n:
            raise ValueError("Sample larger than population")
        result = [None] * k
        selected = set()
        for i in range(k):
            j = TrueRandom.randint(0, n-1)
            while j in selected:
                j = TrueRandom.randint(0, n-1)
            selected.add(j)
            result[i] = population[j]
        return result

    @staticmethod
    def gauss(mu, sigma):
        # Using Box-Muller transform
        u1, u2 = TrueRandom.random(), TrueRandom.random()
        z0 = (-2 * math.log(u1)) ** 0.5 * math.cos(2 * math.pi * u2)
        return mu + z0 * sigma
    
    @staticmethod
    def expovariate(lambd):
        """Exponential distribution."""
        return -math.log(TrueRandom.random()) / lambd

    @staticmethod
    def triangular(low=0.0, high=1.0, mode=None):
        """Triangular distribution."""
        if mode is None:
            mode = (low + high) / 2.0
        u = TrueRandom.random()
        c = (mode - low) / (high - low)
        if u > c:
            u = 1.0 - u
            c = 1.0 - c
            low, high = high, low
        return low + (high - low) * math.sqrt(u * c)

    @staticmethod
    def betavariate(alpha, beta):
        """Beta distribution."""
        if alpha <= 0 or beta <= 0:
            raise ValueError('The parameters alpha and beta must be positive')
        total = alpha + beta
        converted_alpha = (total ** (alpha + beta) / (alpha ** alpha * beta ** beta)) * math.gamma(alpha + beta) / (math.gamma(alpha) * math.gamma(beta))
        ratio = alpha / total
        return ratio * converted_alpha