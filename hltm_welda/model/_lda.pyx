#cython: language_level=3
#cython: boundscheck=True
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, RAND_MAX
from cython.operator cimport preincrement as inc, predecrement as dec
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf

np.import_array()


cdef int searchsorted(double[:] arr, int length, double value) nogil:
  '''
  Bisection search (c.f. numpy.searchsorted)
  Find the index into sorted array `arr` of length `length` such that, if
  `value` were inserted before the index, the order of `arr` would be
  preserved.
  '''
  cdef int imin, imax, imid
  imin = 0
  imax = length
  while imin < imax:
    imid = imin + ((imax - imin) >> 1)
    if value > arr[imid]:
      imin = imid + 1
    else:
      imax = imid
  return imin


cpdef _sample_topics(
  int[:] WS,
  int[:] DS,
  int[:] ZS,
  int[:, :] nzw,
  int[:, :] ndz,
  int[:] nz,
  double[:,:] alpha,
  double[:,:] eta,
  double[:] eta_sum_axis1,
):
  cdef int i, k, w, d, z, z_new
  cdef double r, dist_cum
  cdef int N = WS.shape[0]
  cdef int n_topics = nz.shape[0]
  cdef double eta_sum = 0
  cdef double[:] dist_sum = np.zeros(shape=n_topics, dtype=np.float64)

  try:
    with nogil:
      for i in range(N):
        w = WS[i]
        d = DS[i]
        z = ZS[i]

        dec(nzw[z, w])
        dec(ndz[d, z])
        dec(nz[z])

        dist_cum = 0
        for k in range(n_topics):
          # eta is a double so cdivision yields a double
          dist_cum += (nzw[k, w] + eta[k, w]) / (nz[k] + eta_sum_axis1[k]) * (ndz[d, k] + alpha[d, k])
          dist_sum[k] = dist_cum

        r = (<double>rand()/(<double>RAND_MAX)) * dist_cum # dist_cum == dist_sum[-1]
        z_new = searchsorted(dist_sum, n_topics, r)

        ZS[i] = z_new
        inc(nzw[z_new, w])
        inc(ndz[d, z_new])
        inc(nz[z_new])

  except Exception as e:
    return f'Exception in _sample_topics: {e}'



cpdef fit(
  int iterations,
  np.ndarray[np.int64_t, ndim=1] Nt_in,
  np.ndarray[np.int64_t, ndim=1] Nw_in,
  np.ndarray[np.int64_t, ndim=1] Nd_in,
  np.ndarray[np.int64_t, ndim=2] dt,
  np.ndarray[np.int64_t, ndim=2] wt,
  np.ndarray[np.float64_t, ndim=2] alpha,
  np.ndarray[np.float64_t, ndim=2] eta,
  int K, # number of topics
):
  cdef int it, idx
  cdef int token_count = len(Nw_in)
  Nt_np_arr = np.zeros(shape=token_count, dtype='int')
  nz = np.zeros(shape=K, dtype='int')

  try:
    WS = Nw_in.astype(np.intc)
    DS = Nd_in.astype(np.intc)
    ZS = Nt_in.astype(np.intc)

    alpha = alpha.astype(np.float64)
    eta = eta.astype(np.float64)

    nzw = wt.astype(np.intc)
    ndz = dt.astype(np.intc)

    for idx_wt, val_wt in enumerate(wt.sum(axis=1)):
      nz[idx_wt] = val_wt

    nz = nz.astype(np.intc)

    eta_sum_axis1 = eta.sum(axis=1).astype(np.float64)

    for it in range(iterations):
      _sample_topics(
        WS=WS,
        DS=DS,
        ZS=ZS,
        nzw=nzw,
        ndz=ndz,
        nz=nz,
        alpha=alpha,
        eta=eta,
        eta_sum_axis1=eta_sum_axis1
      )

    for idx in range(token_count):
      Nt_np_arr[idx] = ZS[idx]

    # print(Nt_np_arr)

    return Nt_np_arr

  except Exception as e:
    return f'Exception in fit: {e}'


cpdef form_wt(
  np.ndarray[np.int64_t, ndim=1] Nt_in,
  np.ndarray[np.int64_t, ndim=1] Nw_in,
  int K,
  int W,
):
  nzw_np_arr = np.zeros(shape=(K, W), dtype='int')
  cdef int topic_index, word_index
  cdef int token_count = len(Nw_in)

  cdef int i = 0
  cdef int j = 0
  cdef int k, w

  cdef int[:] Nt
  cdef int[:] Nw
  cdef int[:,:] nzw

  nzw = np.zeros(shape=(K, W), dtype=np.intc)

  try:
    Nt = Nt_in.astype(np.intc)
    Nw = Nw_in.astype(np.intc)

    # print("in form_wt")
    with nogil:
      for i in range(token_count):
        topic_index = Nt[i]
        word_index = Nw[i]
        inc(nzw[topic_index, word_index])

    for k in range(K):
      for w in range(W):
        nzw_np_arr[k, w] = nzw[k, w]

    return nzw_np_arr

  except Exception as e:
    return f'Exception in form_wt: {e}'
    # return f'Exception in form_wt: {e}, token_count: {token_count}, i: {i}, ',
    #        f'len(Nt): {len(Nt)}, len(Nw): {len(Nw)}, Nt[i]: {Nt[i]}, Nw[i]: ',
    #        f'{Nw[i]}, W: {W}, K: {K}'



cpdef form_dt(
  np.ndarray[np.int64_t, ndim=1] Nt_in,
  np.ndarray[np.int64_t, ndim=1] Nd_in,
  int K
):
  cdef int token_index = 0
  cdef int token_count = len(Nt_in)
  cdef int D = len(np.unique(Nd_in))
  cdef int j, i, idx, topic_index, doc_index, d, k
  ndz_np_arr = np.zeros(shape=(D, K), dtype='int')

  cdef int[:] Nt
  cdef int[:] Nd
  cdef int[:,:] ndz

  ndz = np.zeros(shape=(D, K), dtype=np.intc)

  try:
    Nt = Nt_in.astype(np.intc)
    Nd = Nd_in.astype(np.intc)

    with nogil:
      for i in range(token_count):
        topic_index = Nt[i]
        doc_index = Nd[i]
        inc(ndz[doc_index, topic_index])

    for d in range(D):
      for k in range(K):
        ndz_np_arr[d, k] = ndz[d, k]

    return ndz_np_arr

  except Exception as e:
    return f'Exception in form_dt: {e}'


cpdef remove_doc(
  np.ndarray[np.int64_t, ndim=1] Nt_in,
  np.ndarray[np.int64_t, ndim=1] Nw_in,
  np.ndarray[np.int64_t, ndim=1] Nw_copy_in,
  np.ndarray[np.int64_t, ndim=1] Nd_in,
  int doc_idx,
):
  cdef int n_tokens_in_doc = len(np.where(Nd_in == doc_idx)[0])
  cdef int token_count = len(Nw_in)
  cdef int W = len(np.unique(Nw_in))
  cdef int i, j, l, dl_idx, wc_init, index, wct_idx, w_rmv
  cdef int new_idx = 0
  cdef int token_index = 0

  Nt_new_np_arr = np.zeros(shape=(token_count - n_tokens_in_doc), dtype='int')
  Nw_new_np_arr = np.zeros(shape=(token_count - n_tokens_in_doc), dtype='int')
  Nw_copy_new_np_arr = np.zeros(shape=(token_count - n_tokens_in_doc), dtype='int')
  Nd_new_np_arr = np.zeros(shape=(token_count - n_tokens_in_doc), dtype='int')
  word_counts_np_arr = np.zeros(shape=W, dtype='int')

  cdef int[:] Nt = Nt_in.astype(np.intc)
  cdef int[:] Nd = Nd_in.astype(np.intc)
  cdef int[:] Nw = Nw_in.astype(np.intc)
  cdef int[:] Nw_copy = Nw_copy_in.astype(np.intc)
  cdef int[:] Nt_new = np.zeros(shape=(token_count - n_tokens_in_doc), dtype=np.intc)
  cdef int[:] Nd_new = np.zeros(shape=(token_count - n_tokens_in_doc), dtype=np.intc)
  cdef int[:] Nw_new = np.zeros(shape=(token_count - n_tokens_in_doc), dtype=np.intc)
  cdef int[:] Nw_copy_new = np.zeros(shape=(token_count - n_tokens_in_doc), dtype=np.intc)
  cdef int[:] word_counts = np.zeros(shape=W, dtype=np.intc)

  try:
    with nogil:
      for wc_init in range(token_count):
        inc(word_counts[Nw[wc_init]])

      for i in range(token_count):
        if doc_idx != Nd[i]:
          if doc_idx > Nd[i]:
            Nd_new[new_idx] = Nd[i]
          else:
            Nd_new[new_idx] = Nd[i] - 1
          Nw_new[new_idx] = Nw[i]
          Nw_copy_new[new_idx] = Nw_copy[i]
          Nt_new[new_idx] = Nt[i]
          new_idx += 1
        else:
          dec(word_counts[Nw[i]])

    Nt_new_np_arr = np.array(Nt_new, dtype='int')
    Nd_new_np_arr = np.array(Nd_new, dtype='int')
    Nw_new_np_arr = np.array(Nw_new, dtype='int')
    Nw_copy_new_np_arr = np.array(Nw_copy_new, dtype='int')
    word_counts_np_arr = np.array(word_counts, dtype='int')

    Nw_new_np_arr = Nw_new_np_arr - np.digitize(
                                      x=Nw_new_np_arr,
                                      bins=np.where(word_counts_np_arr == 0)[0],
                                      right=True
                                    )

    Nw_copy_new_np_arr = Nw_copy_new_np_arr - np.digitize(
                                      x=Nw_copy_new_np_arr,
                                      bins=np.where(word_counts_np_arr == 0)[0],
                                      right=True
                                    )

    return Nt_new_np_arr, Nw_new_np_arr, Nw_copy_new_np_arr, Nd_new_np_arr, word_counts_np_arr

  except Exception as e:
    return f'Exception in remove_doc: {e}'


cpdef remove_word(
    np.ndarray[np.int64_t, ndim=1] Nt_in,
    np.ndarray[np.int64_t, ndim=1] Nw_in,
    np.ndarray[np.int64_t, ndim=1] Nw_copy_in,
    np.ndarray[np.int64_t, ndim=1] Nd_in,
    int sw_idx,
):
  cdef int sw_token_count = len(np.where(Nw_in == sw_idx)[0])
  cdef int token_count = len(Nw_in)
  cdef int D = len(np.unique(Nd_in))
  cdef np.ndarray[np.int64_t, ndim=1] doc_lengths_np_arr = np.zeros(shape=D, dtype='int')
  cdef int i, j, l, dl_idx, dl_init, index, word_idx
  cdef int new_idx = 0
  cdef int token_index = 0

  Nt_new_np_arr = np.zeros(shape=(token_count - sw_token_count), dtype='int')
  Nw_new_np_arr = np.zeros(shape=(token_count - sw_token_count), dtype='int')
  Nw_copy_new_np_arr = np.zeros(shape=(token_count - sw_token_count), dtype='int')
  Nd_new_np_arr = np.zeros(shape=(token_count - sw_token_count), dtype='int')

  cdef int[:] Nt = Nt_in.astype(np.intc)
  cdef int[:] Nd = Nd_in.astype(np.intc)
  cdef int[:] Nw = Nw_in.astype(np.intc)
  cdef int[:] Nw_copy = Nw_copy_in.astype(np.intc)
  cdef int[:] Nt_new = np.zeros(shape=(token_count - sw_token_count), dtype=np.intc)
  cdef int[:] Nd_new = np.zeros(shape=(token_count - sw_token_count), dtype=np.intc)
  cdef int[:] Nw_new = np.zeros(shape=(token_count - sw_token_count), dtype=np.intc)
  cdef int[:] Nw_copy_new = np.zeros(shape=(token_count - sw_token_count), dtype=np.intc)
  cdef int[:] doc_lengths = np.zeros(shape=D, dtype=np.intc)

  try:
    with nogil:
      for dl_init in range(token_count):
        inc(doc_lengths[Nd[dl_init]])

      for i in range(token_count):
        if sw_idx != Nw[i]:
          if Nw[i] > sw_idx:
            Nw_new[new_idx] = Nw[i] - 1
          else:
            Nw_new[new_idx] = Nw[i]

          if Nw_copy[i] > sw_idx:
            Nw_copy_new[new_idx] = Nw_copy[i] - 1
          else:
            Nw_copy_new[new_idx] = Nw_copy[i]

          Nd_new[new_idx] = Nd[i]
          Nt_new[new_idx] = Nt[i]
          new_idx += 1
        else:
          dec(doc_lengths[Nd[i]])

    Nt_new_np_arr = np.array(Nt_new, dtype='int')
    Nd_new_np_arr = np.array(Nd_new, dtype='int')
    Nw_new_np_arr = np.array(Nw_new, dtype='int')
    Nw_copy_new_np_arr = np.array(Nw_copy_new, dtype='int')
    doc_lengths_np_arr = np.array(doc_lengths, dtype='int')

    Nd_new_np_arr = Nd_new_np_arr - np.digitize(
                                      x=Nd_new_np_arr,
                                      bins=np.where(doc_lengths_np_arr == 0)[0],
                                      right=True
                                    )

    return Nt_new_np_arr, Nw_new_np_arr, Nw_copy_new_np_arr, Nd_new_np_arr, doc_lengths_np_arr

  except Exception as e:
    return f'Exception in remove_word: {e}'
