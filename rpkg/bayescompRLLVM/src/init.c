#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stddef.h>

void *bcn_compile(const char *ir, const char *data, size_t dim, int engine, char *err, size_t errlen);
int bcn_eval(void *h, const double *u, size_t n, int jacobian, int propto, double *lp, double *grad, char *err, size_t errlen);
void bcn_free(void *h);

#define ERRLEN 4096

static void finalize(SEXP p) {
  void *h = R_ExternalPtrAddr(p);
  if (h) {
    bcn_free(h);
    R_ClearExternalPtr(p);
  }
}

static SEXP bcn_r_compile(SEXP ir, SEXP data, SEXP dim, SEXP engine) {
  char err[ERRLEN];
  err[0] = 0;
  void *h = bcn_compile(CHAR(STRING_ELT(ir, 0)), CHAR(STRING_ELT(data, 0)), (size_t) REAL(dim)[0],
                        INTEGER(engine)[0], err, ERRLEN);
  if (!h) return mkString(err);
  SEXP p = PROTECT(R_MakeExternalPtr(h, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(p, finalize, TRUE);
  UNPROTECT(1);
  return p;
}

static SEXP bcn_r_is_null(SEXP p) {
  return ScalarLogical(TYPEOF(p) != EXTPTRSXP || R_ExternalPtrAddr(p) == NULL);
}

static SEXP bcn_r_eval(SEXP p, SEXP u, SEXP jacobian, SEXP propto) {
  char err[ERRLEN];
  err[0] = 0;
  R_xlen_t n = XLENGTH(u);
  SEXP grad = PROTECT(allocVector(REALSXP, n));
  double lp = 0;
  int st = bcn_eval(R_ExternalPtrAddr(p), REAL(u), (size_t) n, LOGICAL(jacobian)[0], LOGICAL(propto)[0],
                    &lp, REAL(grad), err, ERRLEN);
  if (st != 0) {
    UNPROTECT(1);
    return mkString(err);
  }
  SEXP out = PROTECT(allocVector(VECSXP, 2));
  SEXP nm = PROTECT(allocVector(STRSXP, 2));
  SET_VECTOR_ELT(out, 0, ScalarReal(lp));
  SET_VECTOR_ELT(out, 1, grad);
  SET_STRING_ELT(nm, 0, mkChar("value"));
  SET_STRING_ELT(nm, 1, mkChar("grad"));
  setAttrib(out, R_NamesSymbol, nm);
  UNPROTECT(3);
  return out;
}

static const R_CallMethodDef calls[] = {
  {"C_bcn_r_compile", (DL_FUNC) &bcn_r_compile, 4},
  {"C_bcn_r_is_null", (DL_FUNC) &bcn_r_is_null, 1},
  {"C_bcn_r_eval", (DL_FUNC) &bcn_r_eval, 4},
  {NULL, NULL, 0}
};

void R_init_bayescompRLLVM(DllInfo *dll) {
  R_registerRoutines(dll, NULL, calls, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
  R_forceSymbols(dll, TRUE);
}
