#include <cstdio>
#include <cstdlib>
#include <x86intrin.h>

int main() {
  const int N = 16;
  alignas(64) float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  __m512 xvec = _mm512_load_ps(x);
  __m512 yvec = _mm512_load_ps(y);
  __m512 mvec = _mm512_load_ps(m);
  __m512 jvec = _mm512_set_ps(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
  for(int i=0; i<N; i++) {
    __m512 rx = _mm512_sub_ps(_mm512_set1_ps(x[i]), xvec);
    __m512 ry = _mm512_sub_ps(_mm512_set1_ps(y[i]), yvec);
    __m512 r2 = _mm512_fmadd_ps(rx, rx, _mm512_mul_ps(ry, ry));
    __m512 inv = _mm512_rsqrt14_ps(r2);
    __mmask16 mask = _mm512_cmp_ps_mask(jvec, _mm512_set1_ps((float)i), _MM_CMPINT_NE);
    inv = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), inv);
    __m512 inv3 = _mm512_mul_ps(_mm512_mul_ps(inv, inv), inv);
    fx[i] = -_mm512_reduce_add_ps(_mm512_mul_ps(_mm512_mul_ps(rx, mvec), inv3));
    fy[i] = -_mm512_reduce_add_ps(_mm512_mul_ps(_mm512_mul_ps(ry, mvec), inv3));
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
