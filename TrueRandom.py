import requests
import math
import itertools
import bisect
import qiskit
from qiskit import Aer, transpile, assemble
from qiskit.circuit import QuantumCircuit
from collections.abc import Sequence


class TrueRandom:
    
    @staticmethod
    def get_true_random_numbers(n=1, enable_decimals=False, enable_negative=False):
        try: 
            url = f"https://qrng.anu.edu.au/API/jsonI.php?length={n}&type=uint16"
            response = requests.get(url)
            if response.status_code == 200:
                random_numbers = response.json()["data"]
                if enable_decimals:
                    random_numbers = [float(num) / (2**16 - 1) for num in random_numbers]
                if enable_negative:
                    random_numbers = [num if num < 0.5 else -num for num in random_numbers]
                return random_numbers
            else: 
                raise Exception("API request failed")
        except Exception as e:
            # Fallback to Qiskit simulation
            # Number of qubits, 34 qubits for numbers 0 - 2^34-1
            num_qubits = 34

            # Create a quantum circuit with the number of qubits specified
            qc = QuantumCircuit(num_qubits)

            # Apply the Hadamard gate to all qubits to put them in superposition
            for i in range(num_qubits):
                qc.h(i)

            # Measure all the qubits
            qc.measure_all()

            # Execute the circuit on a simulator backend
            backend = Aer.get_backend('qasm_simulator')
            tqc = transpile(qc, backend)
            qobj = assemble(tqc)
            result = backend.run(qobj).result()

            # Extract the counts (how many times each result was generated)
            counts = result.get_counts()

            # Extract random results from the counts
            random_binary_results = list(counts.keys())[:n]

            # Convert binary strings to integers
            random_numbers = [int(result, 2) for result in random_binary_results]

            if enable_decimals:
                random_numbers = [float(num) / (2**num_qubits - 1) for num in random_numbers]
            if enable_negative:
                random_numbers = [num if num < 0.5 else -num for num in random_numbers]

            return random_numbers

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
    
    @staticmethod
    def gammavariate(alpha, beta):
        """Gamma distribution."""
        if alpha <= 0 or beta <= 0:
            raise ValueError("gammavariate: alpha and beta must be > 0.0")
        
        # Use the Marsaglia and Tsang method
        if alpha > 1.0:
            ainv = math.sqrt(2.0 * alpha - 1.0)
            bbb = alpha - math.log(4.0)
            ccc = alpha + ainv
            while True:
                u1 = TrueRandom.random()
                if not 1e-7 < u1 < 0.9999999:
                    continue
                u2 = 1.0 - TrueRandom.random()
                v = math.log(u1 / (1.0 - u1)) / ainv
                x = alpha * math.exp(v)
                z = u1 * u1 * u2
                r = bbb + ccc * v - x
                if r + 4.5 * z >= 4.0 or r >= math.log(z):
                    return x * beta
        else:
            while True:
                x = TrueRandom.random()
                u = TrueRandom.random()
                if x > 0.0:
                    v = (-math.log(x)) ** (1.0 / alpha)
                    if u <= math.exp(-v):
                        return v * beta
                    if u <= v ** (alpha - 1.0):
                        return v * beta

    @staticmethod
    def vonmisesvariate(mu, kappa):
        """Circular data distribution (von Mises)."""
        if kappa <= 0.0:
            raise ValueError("vonmisesvariate: kappa must be > 0.0")
        random = TrueRandom.random
        TWOPI = math.pi * 2.0
        s = 0.5 / kappa
        r = s + math.sqrt(1.0 + 4.0 * kappa * s)
        rho = 1.0 / (2.0 * kappa * r)
        while True:
            u1 = random()
            z = math.cos(math.pi * u1)
            f = (1.0 + rho * z) / (r + z)
            c = kappa * (r - f)
            u2 = random()
            if u2 < c * (2.0 - c) or u2 <= c * math.exp(1.0 - c):
                return (mu + math.atan2((r - f) * math.sin(TWOPI * u1), f) + math.pi) % TWOPI - math.pi

    @staticmethod
    def paretovariate(alpha):
        """Pareto distribution."""
        if alpha <= 0:
            raise ValueError("paretovariate: alpha must be > 0")
        u = TrueRandom.random()
        return 1.0 / (u ** (1.0 / alpha))
    
    @staticmethod
    def weibullvariate(alpha, beta):
        """Weibull distribution."""
        if alpha <= 0 or beta <= 0:
            raise ValueError("weibullvariate: alpha and beta must be > 0")
        u = TrueRandom.random()
        return alpha * (-math.log(u))**(1.0/beta)

    @staticmethod
    def lognormvariate(mu, sigma):
        """Log normal distribution."""
        if sigma <= 0:
            raise ValueError("lognormvariate: sigma must be > 0")
        return math.exp(mu + sigma * TrueRandom.gauss(0, 1))

    @staticmethod
    def binomial(n, p):
        """Binomial distribution."""
        if not (0 < p < 1):
            raise ValueError("binomial: p must be in the range 0 < p < 1")
        if n <= 0:
            raise ValueError("binomial: n must be > 0")
        
        successes = 0
        for _ in range(n):
            if TrueRandom.random() < p:
                successes += 1
        return successes
    
    @staticmethod
    def weibullvariate(alpha, beta):
        """Weibull distribution."""
        if alpha <= 0 or beta <= 0:
            raise ValueError("weibullvariate: alpha and beta must be > 0")
        u = TrueRandom.random()
        return alpha * (-math.log(u))**(1.0/beta)

    @staticmethod
    def lognormvariate(mu, sigma):
        """Log normal distribution."""
        if sigma <= 0:
            raise ValueError("lognormvariate: sigma must be > 0")
        return math.exp(mu + sigma * TrueRandom.gauss(0, 1))

    @staticmethod
    def binomial(n, p):
        """Binomial distribution."""
        if not (0 < p < 1):
            raise ValueError("binomial: p must be in the range 0 < p < 1")
        if n <= 0:
            raise ValueError("binomial: n must be > 0")
        
        successes = 0
        for _ in range(n):
            if TrueRandom.random() < p:
                successes += 1
        return successes
    
    @staticmethod
    def getstate():
        """Dummy method to maintain compatibility with random module"""
        return None

    @staticmethod
    def seed(a=None, version=2):
        """Dummy method to maintain compatibility with random module"""
        pass
    
    @staticmethod
    def choices(population, weights=None, *, cum_weights=None, k=1):
        """Return a k sized list of population elements chosen with replacement."""
        if cum_weights is None:
            if weights is None:
                _random = TrueRandom.random
                total = len(population)
                return [population[int(_random() * total)] for i in range(k)]
            cum_weights = list(itertools.accumulate(weights))
        elif weights is not None:
            raise TypeError('Cannot specify both weights and cumulative weights')
        if len(cum_weights) != len(population):
            raise ValueError('The number of weights does not match the population')
        total = cum_weights[-1]
        hi = len(cum_weights) - 1
        _random = TrueRandom.random
        return [population[bisect.bisect(cum_weights, _random() * total, 0, hi)] for i in range(k)]
    
    @staticmethod
    def shuffle(x, random=None):
        """Shuffle list x in place, and return None."""
        if random is None:
            randbelow = TrueRandom._randbelow
        else:
            _int = int
            randbelow = lambda n: _int(random() * n)
        n = len(x)
        for i in reversed(range(1, n)):
            # pick an element in x[:i+1] with which to exchange x[i]
            j = randbelow(i+1)
            x[i], x[j] = x[j], x[i]
    
    @staticmethod
    def sample(population, k):
        """Return a k length list of unique elements chosen from the population sequence."""
        if not (isinstance(population, Sequence)
                and isinstance(k, int)):
            raise TypeError("Population must be a sequence and k must be an integer")
        n = len(population)
        if not 0 <= k <= n:
            raise ValueError("Sample larger than population or is negative")
        randbelow = TrueRandom._randbelow
        _int = int
        result = [None] * k
        setsize = 21        # size of a small set minus size of an empty list
        if k > 5:
            setsize += 4 ** _int(math.ceil(math.log(k * 3, 4))) # table size for big sets
        if n <= setsize:
            # An n-length list is smaller than a k-length set
            pool = list(population)
            for i in range(k):         # invariant:  non-selected at [0,n-i)
                j = randbelow(n-i)
                result[i] = pool[j]
                pool[j] = pool[n-i-1]   # move non-selected item into vacancy
        else:
            try:
                selected = set()
                selected_add = selected.add
                for i in range(k):
                    j = randbelow(n)
                    while j in selected:
                        j = randbelow(n)
                    selected_add(j)
                    result[i] = population[j]
            except (TypeError, KeyError):   # handle (at least) sets
                if isinstance(population, list):
                    raise
                return sorted(TrueRandom.sample(tuple(population), k), key=_int)
        return result

    @staticmethod
    def _randbelow(n):
        """Helper function for the shuffle and sample methods."""
        return int(TrueRandom.random() * n)