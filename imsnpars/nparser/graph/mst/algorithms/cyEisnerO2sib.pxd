# cython: language_level=3
from libcpp.vector cimport vector

cdef extern from "EisnerO2sib.cpp":
    cdef vector[int] decodeProjective_o2sib(int length,double* scores)

