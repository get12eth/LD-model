"""
Compatibility shim for unpickling models that reference a top-level
`_loss` module. Some pickled models reference internal sklearn modules
that may have been available when the model was saved. Creating this
shim lets pickle import a module named `_loss` and forward lookups to
`sklearn._loss` where appropriate.

This file simply tries to import symbols from scikit-learn's internal
loss module and exposes them at the top-level `_loss` module namespace.
If scikit-learn isn't installed or the internal API differs, importing
this shim will raise an ImportError/AttributeError which gives clearer
diagnostics.
"""
try:
    # Try the public structured loss module first (newer scikit-learn)
    from sklearn._loss import loss as _sk_loss
except Exception:
    try:
        # Fallback: some versions expose names directly under sklearn._loss
        import sklearn._loss as _sk_loss
    except Exception:
        # Re-raise a helpful error to guide the user
        raise ImportError(
            'Unable to import sklearn._loss. Ensure scikit-learn is installed and '  
            'the runtime scikit-learn version matches the one used when the model '  
            'was saved.'
        )

# Re-export commonly used attributes so pickle.find_class can locate them
globals().update({name: getattr(_sk_loss, name) for name in dir(_sk_loss) if not name.startswith("__")})

# Additionally, try to import the compiled extension (where Cython helpers live)
# and re-export any __pyx_unpickle_* helpers and Cython class symbols. This is
# required when a pickle references compiled Cython unpickling helpers like
# __pyx_unpickle_CyHalfBinomialLoss.
try:
    import importlib
    _sk_cy = importlib.import_module('sklearn._loss._loss')
except Exception:
    # If the compiled extension can't be imported, we don't fail here; the
    # earlier ImportError would have been more helpful. Leave this module as-is
    # so the failure later will clearly indicate the missing compiled extension.
    _sk_cy = None

if _sk_cy is not None:
    # Export Cython unpickle helpers and cython class names
    for name in dir(_sk_cy):
        if name.startswith('__pyx_unpickle') or name.startswith('Cy'):
            try:
                globals()[name] = getattr(_sk_cy, name)
            except Exception:
                # ignore attributes that can't be grabbed
                pass

