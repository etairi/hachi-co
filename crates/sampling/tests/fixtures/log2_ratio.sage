# Reference values of hachi_co_sampling::rej::Ratio::log2 for the unit tests in
# src/rej/log2_tests.rs (table SAGE), and the constant TWO_OVER_LN2_Q62 of src/rej.rs.
#
# Run with SageMath 10.9:  sage log2_ratio.sage
#
# Each value is log2(num/den) computed by MPFR at 4000 bits from the exact rational and rounded
# to the nearest double by float() (mpfr_get_d, round to nearest, subnormals included; checked
# below against direct rounding on the 2^-1074 grid).
import struct

R = RealField(4000)


def acc(gamma, beta1, N):
    # hachi_co_sampling::rej::acceptance_probability(gamma, beta1, N), in lowest terms
    a, b = 2*gamma - 2*beta1 + 1, 2*gamma + 1
    g = gcd(a, b)
    return (a // g)^N, (b // g)^N


s1001 = isqrt(2^1001)
CASES = [
    ("toy-n1", *acc(49152, 32, 768)),
    ("toy-n2", *acc(106496, 32, 832)),
    ("toy-n3", *acc(172032, 16, 1792)),
    ("2/1", 2, 1),
    ("1/2", 1, 2),
    ("3/2", 3, 2),
    ("2/3", 2, 3),
    ("1/3", 1, 3),
    ("7/5", 7, 5),
    ("1/2^3000", 1, 2^3000),
    ("(2^1000+1)/2^1000", 2^1000 + 1, 2^1000),
    ("(2^1000-1)/2^1000", 2^1000 - 1, 2^1000),
    ("2^1000/(2^1000-1)", 2^1000, 2^1000 - 1),
    ("(2^1060+1)/2^1060", 2^1060 + 1, 2^1060),
    ("(2^1050-1)/2^1050", 2^1050 - 1, 2^1050),
    ("(2^1100+1)/2^1100", 2^1100 + 1, 2^1100),
    ("3^2000/2^3170", 3^2000, 2^3170),
    ("1/3^1000", 1, 3^1000),
    ("3^1000/1", 3^1000, 1),
    ("(2^64+1)2^5/2^64", (2^64 + 1) * 2^5, 2^64),
    ("1/(2^4000+1)", 1, 2^4000 + 1),
    ("isqrt(2^1001)/2^500", s1001, 2^500),
    ("(isqrt(2^1001)+1)/2^500", s1001 + 1, 2^500),
    ("2^500/isqrt(2^1001)", 2^500, s1001),
    ("2^500/(isqrt(2^1001)+1)", 2^500, s1001 + 1),
    ("65537^1000/65536^1000", 65537^1000, 65536^1000),
    ("F(5001)/F(5000)", fibonacci(5001), fibonacci(5000)),
    ("(10^300+3)/10^300", 10^300 + 3, 10^300),
    ("(2^1022+1)/2^1022", 2^1022 + 1, 2^1022),
    ("(2^1023+1)/2^1023", 2^1023 + 1, 2^1023),
    ("(2^1075+1)/2^1075", 2^1075 + 1, 2^1075),
    ("(2^1076+1)/2^1076", 2^1076 + 1, 2^1076),
    ("2^1100/(2^1100+1)", 2^1100, 2^1100 + 1),
    ("2^100000/3", 2^100000, 3),
    ("3^100000/2^158496", 3^100000, 2^158496),
]

print("const TWO_OVER_LN2_Q62: u128 = %d;" % round(R(2)^63 / log(R(2))))
print("const SAGE: [(&str, u64); %d] = [" % len(CASES))
for name, num, den in CASES:
    exact = R(QQ(num) / QQ(den)).log2()
    c = float(exact)
    if abs(exact) < R(2)^-1022:
        assert R(c) * R(2)^1074 == (exact * R(2)^1074).round(), name
    bits = struct.unpack("<Q", struct.pack("<d", c))[0]
    print('    ("%s", 0x%016x),' % (name, bits))
print("];")
