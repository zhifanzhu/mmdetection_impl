# naive assemble can be seen as the reverse of correlation

In correlation: 6 parameters A, B, C, dA, dB, dC,
where A, B, dC is known.

Correspondance between assemble and correlation:

update      =   dA
B           =   B, i.e. input2 unchanged
Aff         =   dC (gradOutput)
d_update    =   A
dB          =   dB
dAff        =   C (Output)
