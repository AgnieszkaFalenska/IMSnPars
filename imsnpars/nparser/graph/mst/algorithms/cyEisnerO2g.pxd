# cython: language_level=3
from libcpp.vector cimport vector

cdef extern from "EisnerO2g.cpp":
    cdef vector[int] decodeProjective_o2g(int length,double* scores)

