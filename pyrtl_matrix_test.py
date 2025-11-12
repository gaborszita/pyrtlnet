import argparse

import numpy as np
import pyrtl

import pyrtlnet.pyrtl_matrix as pyrtl_matrix
from pyrtlnet.wire_matrix_2d import WireMatrix2D


def _verify_tensor(name: str, expected: np.ndarray, actual: np.ndarray) -> bool:
    """Compare ``expected`` to ``actual`` and print a report."""
    if expected.shape == actual.shape and np.logical_and.reduce(
        expected == actual, axis=None
    ):
        print(f"{name} results match!")
        print(f"Correct result for {name} tensor:\n{actual}")
        return True
    print(f"{name} results DO NOT MATCH!")
    print(f"\nExpected {name} tensor is:\n{expected}")
    print(f"\nActual {name} tensor is:\n{actual}")
    return False


def _make_np_matrix(shape: tuple[int, int], start: int) -> np.ndarray:
    """Return an integer matrix with the specified shape.

    The matrix will be filled with increasing integers starting from ``start``.

    """
    num_rows, num_columns = shape
    array = np.array(list(range(start, start + num_rows * num_columns)))
    return np.reshape(array, newshape=shape)


def _render_trace(sim: pyrtl.Simulation, prefixes: list[str]) -> None:
    """Display traces that start with a prefix in ``prefixes``.

    This also displays signed integers instead of the default ``hex``, and displays
    state names.

    """
    # Collect trace names that start with one of the desired ``prefixes``.
    trace_list = []
    # Iterate over ``prefixes`` first so the display order matches the prefix order.
    for prefix in prefixes:
        for trace_name in sorted(sim.tracer.trace.keys()):
            if trace_name.startswith(prefix):
                trace_list.append(trace_name)

    sim.tracer.render_trace(
        trace_list=trace_list,
        repr_func=pyrtl.val_to_signed_integer,
        repr_per_name={"mm0.state": pyrtl.enum_name(pyrtl_matrix.State)},
    )

x = _make_np_matrix((4, 4), start=1)
y = _make_np_matrix((4, 4), start=10)
z = _make_np_matrix((4, 4), start=20)

matrix_xy = pyrtl_matrix.make_systolic_array(
    name="mm0",
    a=x,
    b=y,
    b_zero=0,
    input_bitwidth=32,
    accumulator_bitwidth=32,
    first_compute_valid=True
)
matrix_xy.make_outputs("matrix_xy")

matrix_xyz = pyrtl_matrix.make_systolic_array(
    name="mm1",
    a=matrix_xy,
    b=z,
    b_zero=0,
    input_bitwidth=32,
    accumulator_bitwidth=32,
    #first_compute_valid=True
)
matrix_xyz.make_outputs("matrix_xyz")
matrix_xyz.ready <<= True

sim = pyrtl.Simulation()

for i in range(20):
    sim.step()

_render_trace(
    sim=sim,
    prefixes=[
        "mm0.left",
        "mm0.top",
        "mm0.output[",
        "mm0.state",
        "mm0.output.valid",
        "mm1.left",
        "mm1.top",
        "mm1.output[",
        "mm1.state",
        "mm1.output.valid",
    ],
)

expected_xy = x @ y
actual_xy = matrix_xy.inspect(sim=sim)
_verify_tensor("x ⋅ y", expected_xy, actual_xy)

expected_xyz = x @ y @ z
actual_xyz = matrix_xyz.inspect(sim=sim)
_verify_tensor("x ⋅ y ⋅ z", expected_xyz, actual_xyz)