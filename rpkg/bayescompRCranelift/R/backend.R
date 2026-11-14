#' Engine used by the cranelift backend
#'
#' `"jit"` (the default) compiles the model to machine code with Cranelift;
#' `"interpreter"` runs the reference interpreter of the shared native
#' runtime. Both give bit identical results. The engine is read when a model
#' is built with `backend = "cranelift"`.
#'
#' @param engine `NULL` to query, or `"jit"` or `"interpreter"`.
#' @return The engine in use (invisibly when setting).
#' @export
cranelift_engine <- function(engine = NULL) {
  if (is.null(engine)) return(getOption("bayescompRCranelift.engine", "jit"))
  engine <- match.arg(engine, c("jit", "interpreter"))
  options(bayescompRCranelift.engine = engine)
  invisible(engine)
}

engine_code <- function(engine) match(engine, c("interpreter", "jit")) - 1L

compile_native <- function(ir, model) {
  inp <- bayescompR::bayescomp_native_inputs(model)
  code <- engine_code(cranelift_engine())
  build <- function() {
    h <- .Call(C_bcn_r_compile, inp$ir, inp$data, as.double(inp$dim), code)
    if (is.character(h)) bayescompR::bayescomp_native_error(h)
    h
  }
  h <- build()
  ldg <- function(u, jacobian, propto) {
    # a handle copied to another process (a PSOCK worker or a saved model)
    # is null there, so the model is compiled again from the stored inputs
    if (.Call(C_bcn_r_is_null, h)) h <<- build()
    r <- .Call(C_bcn_r_eval, h, as.double(u), isTRUE(jacobian), isTRUE(propto))
    if (is.character(r)) bayescompR::bayescomp_native_error(r)
    r
  }
  list(log_density_gradient = ldg)
}

.onLoad <- function(libname, pkgname) {
  bayescompR::bayescomp_register_backend(
    "cranelift", c("log_density", "gradient"), ir_versions = "1.0.0", compile = compile_native,
    description = "Machine code from the Cranelift code generator (log density and gradient)")
}
