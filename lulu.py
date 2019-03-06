#!/usr/bin/python3
# by Sun Smallwhite <niasw@pku.edu.cn>(https://github.com/niasw)

import numpy

def luluSmooth(matrix_data, order=3, mode="W"):
  '''
# LULU smooth method is proposed by C. H. Rohwer in 1989.
# It has superior effect to deal with pulse noise without destroying normal wave signal.
# Comparing to the traditional median smooth method, LULU has many better properties:
#   monotonicity
#   idempotency and co-idempotency
#   stability
#   statistical properties
#   edge preservation properties
#   variation reduction properties
#
# For more details, this paper introduced it very well:
#   Jankowitz, M. (2019). Some statistical aspects of LULU smoothers. 
#
# Definiton:
#   (Vn x)[i] := min(x[i-n], ..., x[i])
#   (Λn x)[i] := max(x[i], ..., x[i+n])
# Definition:
#   upper smoother:
#   Un x := Λn Vn x
#   lower smoother:
#   Ln x := Vn Λn x
# Remarks:
#   Un removes negative pulses, Ln removes positive pulses.
# Theorem:
#   Ln <= ... <= L2 <= L <= 1 <= U <= U2 <= ... <= Un
#   forall (m<=n): Um Un = Un, Lm Ln = Ln
# =>:
#   Ln = Ln Ln <= Ln Un Ln <= Ln Un
#   Un = Un Un >= Un Ln Un >= Un Ln
# Theorem:
#   forall (m<=n): Lm Un Lm = Un Lm, Um Ln Um = Ln Um
# =>:
#   Ln Un Ln = Un Ln
#   Un Ln Un = Ln Un
# =>:
#   Ln Un >= Ln Un Ln = Un Ln <= Un Ln Un = Ln Un
# =>:
#   Ln Un = Un Ln
#   What?! FIXME! This is wrong. In fact, Un Ln <= Median Smooth <= Ln Un, and Un Ln, Ln Un is not surely greater or lesser with 1
# =>:
#   (Ln Un) (Ln Un) = (Ln Un Ln) Un = (Un Ln) Un = Un Ln Un = Ln Un
#   the same way, (Un Ln) (Un Ln) = Un Ln
# Definition:
#   full smoother:
#   Qn = Un + Ln - 1
# Remarks:
#   Qn removes pulses in both directions.
#   Qn is not syntone, but Q1 is syntone.
#   Ln <= Qn <= Un
#   Qn is not idempotent.
# Definition:
#   full smoother:
#   Gn = (Ln Un + Un Ln) / 2
# Definition:
#   full smoother: (Winsorised)
#   (W*n x)[i] = (x[i] exceed [(Un Ln x)[i], (Ln Un x)[i]]) ? take boundary : x[i]
# Definition:
#   full smoother: (Winsorised)
#   (Wn x)[i] = ((Wn-1 x)[i] exceed [(Un Ln x)[i], (Ln Un x)[i]]) ? take boundary : (Wn-1 x)[i]
#   and set W0 = 1
# Remarks:
#   Wn is idempotent.
# Definition:
#   full smoother:
#   (An x)[i] = (x[i] exceed [(Un Ln x)[i], (Ln Un x)[i]]) ? (Gn x)[i] : x[i]
# Definition:
#   recursive upper smoother:
#   Cn = Ln Un Ln-1 Un-1 ... L2 U2 L U
#   recursive lower smoother:
#   Fn = Un Ln Un-1 Ln-1 ... U2 L2 U L
# Remarks:
#   Un Ln <= Fn <= Cn <= Ln Un
# Definition:
#   full smoother: (Winsorised)
#   (Bn x)[i] = ((Bn-1 x)[i] exceed [(Fn x)[i], (Cn x)[i]]) ? take boundary : (Bn-1 x)[i]
#   and set B0 = 1
# Remarks:
#   Bn is idempotent.
# Theorem:
#   x is n-monotone <=> Ln x = Un x = x
#   x is n-monotone <=> Ln Un x = Un Ln x = x
# Remarks:
#   Ln Un x = x <=> x is n-monotone <=> Un Ln x = x
#   but only knowing Ln Un x = Un Ln x is not enough to conclude that x is n-monotone.
# Theorem:
#   forall x is 0-monotone: Ln Un x and Un Ln x are n-monotone.
# Similar for C,F,W,B smoothers.
# Theorem:
#   LU (1-LU) = 0, UL (1-UL) = 0
#   UL (1-LU) = 0, LU (1-UL) = 0
#   (1-W)(1-W) = 1-W
#   (1-B)(1-B) = 1-B
# Theorem:
#   positive pulse will be widen by n if Λn is applied, and narrowed by n if Vn is applied.
#   negative pulse will be widen by n if Vn is applied, and narrowed by n if Λn is applied.
#   pulse will be removed if its width reaches 0.
#
# C,F are better performance, but the calculation need larger window size (n^2 scale).
#
# Here we use Winsorised smoother Wn in default.
# Supported smoothers:
#   Wn: "W"
#   Un Ln: "UL"
#   Ln Un: "LU"
#   Cn: "C"
#   Fn: "F"
#   Bn: "B"
  '''
  _mat = numpy.array(matrix_data);
  if _mat.ndim == 2:
    (n0,n1) = _mat.shape;
    _ret = [];
    for it in range(0,n0):
        _arr = _mat[it];
        if mode=="W":
            _ret.append(W(_arr, order=order).tolist());
        elif mode=="UL":
            _ret.append(UL(_arr, order=order).tolist());
        elif mode=="LU":
            _ret.append(LU(_arr, order=order).tolist());
        elif mode=="C":
            _ret.append(C(_arr, order=order).tolist());
        elif mode=="F":
            _ret.append(F(_arr, order=order).tolist());            
        elif mode=="B":
            _ret.append(B(_arr, order=order).tolist());
        else:
            raise(Exception("Unknown mode: "+mode));
            _ret.append(_arr.tolist());
  elif _mat.ndim == 1:
        if mode=="W":
            _ret[it] = W(_mat, order=order);
        elif mode=="UL":
            _ret[it] = UL(_mat, order=order);
        elif mode=="LU":
            _ret[it] = LU(_mat, order=order);
        elif mode=="C":
            _ret[it] = C(_mat, order=order);
        elif mode=="F":
            _ret[it] = F(_mat, order=order);            
        elif mode=="B":
            _ret[it] = B(_mat, order=order);
        else:
            raise(Exception("Unknown mode: "+mode));
            _ret[it] = _mat;
  else:
      raise(Exception("Unsupported dimension: "+_mat.ndim));
  return numpy.array(_ret);

def A(array, order):
    '''
# A(array, order)[i]=max(array[i],...,array[i+order]); for i in range(0,n-order)
    '''
    n=array.size;
    if order>=n:
        raise(Exception("Order "+order+" is too large for "+n+" size array."));
    _ret = array[0:n-order].copy();
    for it1 in range(0,n):
        _low = max([it1-order,0]);
        _hgh = min([it1,n-order]);
        for it2 in range(_low,_hgh):
            _ret[it2] = max([_ret[it2],array[it1]]);
    return _ret;

def V(array, order):
    '''
# V(array, order)[i]=min(array[i-order],...,array[i]); for i in range(order,n)
# However, there will be vacuum in the beginning of the array.
# Therefore, we give a shift:
# V(array, order)[j]=min(array[j],...,array[j+order]); for j in range(0,n-order), j=i-order
    '''
    n=array.size;
    if order>=n:
        raise(Exception("Order "+order+" is too large for "+n+" size array."));
    _ret = array[0:n-order].copy();
    for it1 in range(0,n):
        _low = max([it1-order,0]);
        _hgh = min([it1,n-order]);
        for it2 in range(_low,_hgh):
            _ret[it2] = min([_ret[it2],array[it1]]);
    return _ret;

def U(array, order):
    return A(V(array,order),order);

def L(array, order):
    return V(A(array,order),order);

def UL(array, order):
    return U(L(array,order),order);

def LU(array, order):
    return L(U(array,order),order);

def W(array,order):
    n=array.size;
    if 4*order>=n:
        raise(Exception("Order "+order+" is too large for "+n+" size array."));
    _ret = numpy.ndarray([order+1,n]);
    _ret[0] = array;
    for it1 in range(1,order+1):
        _LU = LU(array,it1);
        _UL = UL(array,it1);
        for it2 in range(2*it1,n-2*it1):
            if _ret[it1-1][it2]<=_UL[it2-2*it1]:
                _ret[it1][it2] = _UL[it2-2*it1];
            elif _ret[it1-1][it2]>=_LU[it2-2*it1]:
                _ret[it1][it2] = _LU[it2-2*it1];
            else:
                _ret[it1][it2] = _ret[it1-1][it2];
    return _ret[order][(2*order):(n-2*order)].copy();

def C(array,order):
    _ret = array;
    for it in range(1,order+1):
        _ret = LU(_ret,it);
    return _ret;

def F(array,order):
    _ret = array;
    for it in range(1,order+1):
        _ret = UL(_ret,it);
    return _ret;

def B(array,order):
    n=array.size;
    if 2*order*(order+1)>=n:
        raise(Exception("Order "+order+" is too large for "+n+" size array."));
    _ret = numpy.ndarray([order+1,n]);
    _ret[0] = array;
    for it1 in range(1,order+1):
        _C = C(array,it1);
        _F = F(array,it1);
        for it2 in range(it1*(it1+1),n-it1*(it1+1)):
            if _ret[it1-1][it2]<=_F[it2-it1*(it1+1)]:
                _ret[it1][it2] = _F[it2-it1*(it1+1)];
            elif _ret[it1-1][it2]>=_C[it2-it1*(it1+1)]:
                _ret[it1][it2] = _C[it2-it1*(it1+1)];
            else:
                _ret[it1][it2] = _ret[it1-1][it2];
    return _ret[order][(order*(order+1)):(n-order*(order+1))].copy();
