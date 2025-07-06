"""
BSD-friendly JAX system pipeline test

- Verifies venv status
- Checks OS is FreeBSD
- Verifies JAX and jaxlib presence
- Optionally tests Flax if installed

"""

import sys
import platform
import os

def test_jax_pipeline():
    print("🔎 Starting JAX pipeline system test...")

    # Check if running inside venv
    in_venv = (
        hasattr(sys, 'real_prefix') or
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    )
    if in_venv:
        print("✅ Running inside a virtual environment.")
    else:
        print("⚠️ Not running inside a virtual environment. Recommend using venv on FreeBSD.")

    # Check OS
    system_os = platform.system()
    if system_os == "FreeBSD":
        print("✅ OS detected: FreeBSD.")
    else:
        print(f"⚠️ Non-FreeBSD system detected: {system_os}. Proceeding anyway.")

    # Check JAX
    try:
        import jax
        import jax.numpy as jnp
        print(f"✅ JAX installed. Version: {jax.__version__}")
    except ImportError:
        print("❌ JAX not installed.")
        return

    # Check jaxlib
    try:
        import jaxlib
        print(f"✅ jaxlib installed. Version: {jaxlib.__version__}")
    except ImportError:
        print("❌ jaxlib not installed. JAX ops will fail.")
        return

    # Check Flax
    flax_available = True
    try:
        import flax
        from flax import linen as nn
        print(f"✅ Flax installed. Version: {flax.__version__}")
    except ImportError:
        flax_available = False
        print("⚠️ Flax not installed. Skipping Flax MLP test.")

    # test JAX forward pass
    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (5, 10))
    W = jax.random.normal(key, (10, 1))
    b = jnp.zeros((1,))

    def forward(X, W, b):
        return jnp.dot(X, W) + b

    out = forward(X, W, b)
    print("✅ JAX forward pass output shape:", out.shape)

    # test pass Flax MLP
    if flax_available:
        class SimpleMLP(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = nn.Dense(32)(x)
                x = nn.relu(x)
                x = nn.Dense(1)(x)
                return x

        model = SimpleMLP()
        variables = model.init(key, X)
        output = model.apply(variables, X)
        print("✅ Flax MLP output shape:", output.shape)

    print("🎉 System JAX pipeline test complete.")

if __name__ == "__main__":
    test_jax_pipeline()
