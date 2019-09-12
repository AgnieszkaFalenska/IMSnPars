# cython: language_level=3
from libcpp.vector cimport vector

cdef extern from "Eisner.cpp":
    cdef vector[int] decodeProjective(int length,double* scores)

