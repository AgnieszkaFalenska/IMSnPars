This implementation comes from [nnpgdparser](https://github.com/zzsfornlp/nnpgdparser). It needed small changes to be adjusted it to Cython.

### Compile:
> python3 setup.py build_ext --inplace

### Use:
```
import cyEisnerO2g
cyEisnerO2g.decodeProjective(np.array([1,2]))
```
