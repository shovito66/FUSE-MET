# -*- coding: utf-8 -*-

def get_closest_label_for_met(l, M):
  pass

def cluster_k_means(F, k):
  pass

def labelMerger(l_user, k, lambd, word_vecs, met_words, met_values):
  F = []
  for l in l_user:
    v1 = word_vecs[l]
    M=0
    w = get_closest_label_for_met(l, M)
    m = met_values[w]
    f = v1 + [m*lambd]
    F.append(f)

  l_merge = cluster_k_means(F, k)
  return l_merge