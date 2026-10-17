# NTT primes of hachi-co-math (crates/math/src/ntt.rs, NTT_PRIMES).
#
# Run:  sage crates/math/tools/ntt_primes.sage
#
# The primes p are the NUM_PRIMES largest primes below 2^62 with p = 1 (mod 2^13), in descending
# order. 2^13 | p - 1 gives a primitive 2^13-th root of unity, hence negacyclic transforms of every
# power-of-two size n <= 4096 (a negacyclic transform of size n needs a primitive 2n-th root). The
# bound p < 2^62 is what the Shoup and Montgomery products of the transform need (jali,
# src/math/ntt.rs).
#
# For each prime the root is root = g^((p - 1)/2^13) mod p, where g is the smallest primitive root
# modulo p, found from the definition (g^((p-1)/l) != 1 for every prime l | p - 1), so the choice
# does not depend on how a library picks its generator. The root has order exactly 2^13, i.e.
# root^(2^12) = -1; the Rust tests check this again.
#
# Output: the Rust table, then one line per prime. Sage may leave a preparsed copy
# (ntt_primes.sage.py) next to this file; it is not part of the repository.

LOG_ORDER = 13
ORDER = 2**LOG_ORDER
NUM_PRIMES = 9
BOUND = 2**62


def smallest_primitive_root(p):
    ls = [l for l, _ in factor(p - 1)]
    g = 2
    while True:
        if all(power_mod(g, (p - 1) // l, p) != 1 for l in ls):
            return g
        g += 1


primes = []
p = ((BOUND - 2) // ORDER) * ORDER + 1  # the largest p = 1 (mod 2^13) below 2^62
while len(primes) < NUM_PRIMES:
    assert p < BOUND and p % ORDER == 1
    if is_prime(p):
        g = smallest_primitive_root(p)
        root = power_mod(g, (p - 1) // ORDER, p)
        # order exactly 2^13: root^(2^13) = 1 and root^(2^12) = -1
        assert power_mod(root, ORDER, p) == 1
        assert power_mod(root, ORDER // 2, p) == p - 1
        assert Mod(root, p).multiplicative_order() == ORDER
        primes.append((p, g, root))
    p -= ORDER

print("pub const NTT_PRIMES: [(u64, u64); %d] = [" % NUM_PRIMES)
for p, g, root in primes:
    print("    (%d, %d)," % (p, root))
print("];")
print()
for p, g, root in primes:
    print("p = %d = 2^62 - %d, v2(p - 1) = %d, smallest primitive root g = %d"
          % (p, BOUND - p, (p - 1).valuation(2), g))
total = 1
for p, _, _ in primes:
    total *= p
print("log2(product of the %d primes) = %.3f" % (NUM_PRIMES, float(log(total, 2))))
print(version())
