import math

def pearson(vector1, vector2):
    n = len(vector1)
    sum1 = sum(float(vector1[i]) for i in range(n))
    sum2 = sum(float(vector2[i]) for i in range(n))
    sum1_pow = sum([pow(v, 2.0) for v in vector1])
    sum2_pow = sum([pow(v, 2.0) for v in vector2])
    p_sum = sum([vector1[i] * vector2[i] for i in range(n)])
    num = p_sum - (sum1 * sum2 / n)
    den = math.sqrt((sum1_pow - pow(sum1, 2) / n) * (sum2_pow - pow(sum2, 2) / n))
    if den == 0:
        return 0.0
    return num / den