from abc import ABC, abstractmethod
from enum import Enum
import pyrtl
from pyrtl.rtllib.pyrtlfloat import Float16Operations

from pyrtlnet.wire_matrix_2d import WireMatrix2D
from . import pyrtl_matrix
import numpy as np
import pathlib

class TrainingState(Enum):
  IDLE = 0
  INIT_WEIGHTS = 1
  FORWARD = 2
  BACKWARD = 3
  UPDATE = 4


class PyrtlTraining:
  def __init__(
    self,
    bitwidth: int,
  ) -> None:
    self.bitwidth = bitwidth

    self.layer_0_shape = (18, 144)
    self.layer_1_shape = (10, 18)

    self.training_state = pyrtl.Input(bitwidth=3, name="training_state")

    self.inputs = {"training_state": 0}

    self._make_input_memblock()
    self._make_layers()

  def _make_layers(self) -> None:
    reset = pyrtl.Input(bitwidth=1, name="reset")
    layer_0_weight = ProductNode(self.layer_0_shape, "layer_0_weight", bitwidth=self.bitwidth, make_regs=True, inputs=self.inputs, training_state=self.training_state)
    layer_0_bias = BiasNode((18, 1), "layer_0_bias", bitwidth=self.bitwidth, make_regs=True, inputs=self.inputs, training_state=self.training_state)
    relu = ReluNode((18, 1), "relu", bitwidth=self.bitwidth, make_regs=False, inputs=self.inputs, training_state=self.training_state)
    layer_1_weight = ProductNode(self.layer_1_shape, "layer_1_weight", bitwidth=self.bitwidth, inputs=self.inputs, training_state=self.training_state)
    layer_1_bias = BiasNode((10, 1), "layer_1_bias", bitwidth=self.bitwidth, inputs=self.inputs, training_state=self.training_state)
    loss = Loss((10, 1), "loss", bitwidth=self.bitwidth, make_regs=False, inputs=self.inputs, training_state=self.training_state)
    layers = [layer_0_weight, layer_0_bias, relu, layer_1_weight, layer_1_bias, loss]
    forward_output = self.flat_image_matrix
    for layer in layers:
      forward_output = layer.make_forward(forward_output, reset=reset)
    upstream_gradient = loss.make_backward(reset=reset)
    for layer in reversed(layers[:-1]):
      upstream_gradient = layer.make_backward(upstream_gradient, reset=reset)
    learning_rate = pyrtl.Const(np.float16(0.001).view(np.uint16), bitwidth=self.bitwidth)
    layer_0_weight.make_update(learning_rate)
    layer_0_bias.make_update(learning_rate)
    layer_1_weight.make_update(learning_rate)
    layer_1_bias.make_update(learning_rate)
    #upstream_gradient.ready <<= True
    self.forward_output = forward_output
    self.flat_image_matrix.valid <<= True
    valid = pyrtl.Output(name="valid", bitwidth=1)
    valid <<= forward_output.valid
    loss_output = pyrtl.Output(name="loss_output", bitwidth=self.bitwidth)
    loss_output <<= loss.loss
    valid_backward = pyrtl.Output(name="valid_backward", bitwidth=1)
    valid_backward <<= upstream_gradient.valid
  
  def simulate(self, flat_image: np.ndarray) -> None:
    flat_image = np.reshape(flat_image, newshape=(144, 1)).astype(np.float16).view(np.uint16)
    memblock_data = pyrtl_matrix.make_input_memblock_data(
      flat_image.transpose(),
      self.bitwidth,
      self.flat_image_addrwidth,
    )

    memblock_data_dict = dict(enumerate(memblock_data))
    print("fastsim start")
    sim = pyrtl.FastSimulation(
      memory_value_map={self.flat_image_memblock: memblock_data_dict}
    )
    print("fastsim done")

    for input in self.inputs:
      if input.startswith("weight_init_"):
        value = np.random.randn() * 0.1
        self.inputs[input] = np.float16(value).view(np.uint16)
    
    self.inputs["reset"] = 0

    self.inputs["training_state"] = TrainingState.INIT_WEIGHTS.value
    sim.step(provided_inputs=self.inputs)
    
    for epoch in range(10):
      print(f"Epoch {epoch}")
      self.inputs["training_state"] = TrainingState.FORWARD.value
      done = False
      cycle_count = 0
      while not done:
        #print(f"Cycle {cycle_count}")
        sim.step(provided_inputs=self.inputs)
        done = sim.inspect("valid")
        cycle_count += 1
      print(f"Forward pass done in {cycle_count} cycles")
      print(np.uint16(sim.inspect("loss_output")).view(np.float16))
      #layer_0_weight_output = np.array([
      #  sim.inspect(f"layer_0_weight_out_{r}_0") for r in range(18)
      #])
      #print("Layer 0 weight output:", layer_0_weight_output)

      self.inputs["reset"] = 1
      sim.step(provided_inputs=self.inputs)
      self.inputs["reset"] = 0

      self.inputs["training_state"] = TrainingState.BACKWARD.value
      done = False
      cycle_count = 0
      while not done:
        sim.step(provided_inputs=self.inputs)
        done = sim.inspect("valid_backward")
        cycle_count += 1
      print(f"Backward pass done in {cycle_count} cycles")

      self.inputs["training_state"] = TrainingState.UPDATE.value
      sim.step(provided_inputs=self.inputs)

      self.inputs["reset"] = 1
      sim.step(provided_inputs=self.inputs)
      self.inputs["reset"] = 0
  
  def _make_input_memblock(self) -> None:
    """Build the MemBlock that will hold the input image data."""
    weight_shape = self.layer_0_shape
    batch_size = 1
    flat_image_shape = (weight_shape[1], batch_size)

    done_cycle = (
        pyrtl_matrix.num_systolic_array_cycles(weight_shape, flat_image_shape) - 1
    )
    self.flat_image_addrwidth = pyrtl.infer_val_and_bitwidth(done_cycle).bitwidth

    # Create a properly-sized empty MemBlock. The MemBlock's contents will be set
    # at simulation time in `simulate()`.
    _, num_columns = flat_image_shape
    self.flat_image_memblock = pyrtl.MemBlock(
        name="flat_image",
        addrwidth=self.flat_image_addrwidth,
        bitwidth=self.bitwidth * num_columns,
    )

    # Create a WireMatrix2D that wraps the empty MemBlock. This will be the first
    # layer's input.
    self.flat_image_matrix = WireMatrix2D(
        values=self.flat_image_memblock,
        shape=flat_image_shape,
        bitwidth=self.bitwidth,
        name="flat_image",
    )


class Node(ABC):
  def __init__(self,
               shape: tuple[int, int],
               layer_name, bitwidth: int,
               make_regs: bool = True,
               training_state: pyrtl.WireVector = None,
               inputs: dict[str, pyrtl.WireVector] = None) -> None:
    self.bitwidth = bitwidth
    self.training_state = training_state
    self.layer_name = layer_name
    self.shape = shape
    self.inputs = inputs
    rows, cols = shape
    if make_regs:
      self._make_update_regs(rows, cols)
      self.regs = WireMatrix2D(
        values=[
        [self._make_input_reg(r, c) for c in range(cols)]
        for r in range(rows)
        ],
        shape=(rows, cols),
        bitwidth=self.bitwidth,
        name=layer_name + "_regs",
        valid=True,
      )

  def _make_input_reg(self, r, c) -> pyrtl.Register:
    reg = pyrtl.Register(bitwidth=self.bitwidth)
    input_name = f"weight_init_{self.layer_name}_{r}_{c}"
    input = pyrtl.Input(bitwidth=self.bitwidth, name=input_name)
    self.inputs[input_name] = 0
    with pyrtl.conditional_assignment:
      with self.training_state == TrainingState.INIT_WEIGHTS.value:
        reg.next |= input
      with self.training_state == TrainingState.UPDATE.value:
        reg.next |= self.update_regs[r][c]
      with pyrtl.otherwise:
        reg.next |= reg
    #reg.next <<= pyrtl.select(
    #  self.training_state == TrainingState.INIT_WEIGHTS.value,
    #  input,
    #  reg,
    #)
    output = pyrtl.Output(name=f"{self.layer_name}_out_{r}_{c}", bitwidth=self.bitwidth)
    output <<= reg
    return reg
  
  def _make_update_regs(self, rows, cols) -> WireMatrix2D:
    self.update_regs = [
      [pyrtl.Register(bitwidth=self.bitwidth) for c in range(cols)]
      for r in range(rows)
    ]

class ProductNode(Node):
  def make_forward(self, input, reset):
    self.input = input
    return pyrtl_matrix.make_systolic_array(
      name=self.layer_name + "_forward",
      a=self.regs,
      b=input,
      b_zero=None,
      input_bitwidth=self.bitwidth,
      accumulator_bitwidth=self.bitwidth,
      initial_delay_cycles=0,
      quantized=False,
      reset=reset,
    )

  def make_backward(self, upstream_gradient: WireMatrix2D, reset):
    self.input_gradient = pyrtl_matrix.make_systolic_array(
      name=self.layer_name + "_backward_input",
      a=self.regs.transpose(),
      b=upstream_gradient,
      b_zero=None,
      input_bitwidth=self.bitwidth,
      accumulator_bitwidth=self.bitwidth,
      initial_delay_cycles=0,
      quantized=False,
      reset=reset,
    )

    #return self.input_gradient

    self.weight_gradient = pyrtl_matrix.make_systolic_array(
      name=self.layer_name + "_backward_weight",
      a=upstream_gradient,
      b=self.input.transpose(),
      b_zero=None,
      input_bitwidth=self.bitwidth,
      accumulator_bitwidth=self.bitwidth,
      initial_delay_cycles=0,
      quantized=False,
      reset=reset,
    )

    return self.input_gradient

  def make_update(self, learning_rate):
    for r in range(self.shape[0]):
      for c in range(self.shape[1]):
        grad_term = Float16Operations.mul(learning_rate, self.weight_gradient[r][c])
        self.update_regs[r][c].next <<= Float16Operations.sub(
          self.regs[r][c], grad_term
        )


class BiasNode(Node):
  def make_forward(self, input, reset):
    return pyrtl_matrix.make_elementwise_add(
      name=self.layer_name + "_forward",
      a=input,
      b=self.regs,
      output_bitwidth=self.bitwidth,
      quantized=False,
    )

  def make_backward(self, upstream_gradient: WireMatrix2D, reset):
    self.gradient = upstream_gradient
    return self.gradient

  def make_update(self, learning_rate):
    for r in range(self.shape[0]):
      for c in range(self.shape[1]):
        grad_term = Float16Operations.mul(learning_rate, self.gradient[r][c])
        self.update_regs[r][c].next <<= Float16Operations.sub(
          self.regs[r][c], grad_term
        )


class ReluNode(Node):
  def make_forward(self, input, reset):
    # Apply elementwise ReLU and save activation for use during backprop
    relu_out = pyrtl_matrix.make_elementwise_relu(
      name=self.layer_name + "_forward",
      a=input,
    )
    self.input = input
    self.output = relu_out
    return relu_out

  def make_backward(self, upstream_gradient: WireMatrix2D, reset):
    # Compute gradient of ReLU using sign bit of (input - 0.0).
    rows, cols = self.input.shape

    # Constant zero as float16 bitpattern
    zero_f16 = np.float16(0.0).view(np.uint16)
    zero_const = pyrtl.Const(zero_f16, bitwidth=self.bitwidth)

    grad_values = []
    for r in range(rows):
      grad_row = []
      for c in range(cols):
        cond = self.input[r][c][15] == 0
        grad_elem = pyrtl.select(cond, upstream_gradient[r][c], zero_const)
        grad_row.append(grad_elem)
      grad_values.append(grad_row)

    self.gradient = WireMatrix2D(
      values=grad_values,
      shape=(rows, cols),
      bitwidth=self.bitwidth,
      name=self.layer_name + "_backward_grad",
      valid=True,
    )
    return self.gradient


class Loss(Node):
  def _make_expected_input(self, r, c):
    input_name = f"expected_{r}_{c}"
    input = pyrtl.Input(bitwidth=self.bitwidth, name=input_name)
    self.inputs[input_name] = 0
    return input

  def make_forward(self, input, reset):
    expected_values = [
      [self._make_expected_input(r, c) for c in range(self.shape[1])]
      for r in range(self.shape[0])
    ]
    self.expected = WireMatrix2D(
      values=expected_values,
      shape=self.shape,
      bitwidth=self.bitwidth,
      name="expected",
      valid=True,
    )

    # Elementwise subtraction: create a matrix of input - expected
    diff = pyrtl_matrix.make_elementwise_sub(
      name=self.layer_name + "_diff",
      a=input,
      b=self.expected,
      output_bitwidth=self.bitwidth,
    )
    #diff.ready <<= True
    self.diff = diff
    # Elementwise square: create a matrix of diff * diff using PyrtlFloat.mul
    rows, cols = diff.shape
    squared_values = [
      [Float16Operations.mul(diff[r][c], diff[r][c]) for c in range(cols)]
      for r in range(rows)
    ]
    squared = WireMatrix2D(
      values=squared_values,
      shape=(rows, cols),
      bitwidth=self.bitwidth,
      name=self.layer_name + "_squared",
      valid=diff.valid,
      ready=True,
    )

    # Reduce-sum: accumulate all elements of `squared` into a single scalar loss
    acc = pyrtl.Const(0, bitwidth=self.bitwidth)
    for r in range(rows):
      for c in range(cols):
        acc = Float16Operations.add(acc, squared_values[r][c])

    # Divide by number of elements to compute mean (MSE)
    n = rows * cols
    reciprocal_value = 1.0 / n
    reciprocal_const = np.float16(reciprocal_value).view(np.uint16)
    reciprocal_const = pyrtl.Const(reciprocal_const, bitwidth=self.bitwidth)
    loss = Float16Operations.mul(acc, reciprocal_const)

    loss_matrix = WireMatrix2D(
      values=[[loss]],
      shape=(1, 1),
      bitwidth=self.bitwidth,
      name=self.layer_name + "_loss_matrix",
      valid=squared.valid,
      ready=True,
    )

    self.n = n

    # Store the scalar loss (WireVector) so callers can use it
    self.loss = loss
    return loss_matrix

  def make_backward(self, reset):
    div_value = 2.0 / self.n
    div_value = np.float16(div_value).view(np.uint16)
    div_value = pyrtl.Const(div_value, bitwidth=self.bitwidth)
    rows, cols = self.diff.shape
    grad_values = [
      [Float16Operations.mul(self.diff[r][c], div_value) for c in range(cols)]
      for r in range(rows)
    ]
    gradient = WireMatrix2D(
      values=grad_values,
      shape=(rows, cols),
      bitwidth=self.bitwidth,
      name=self.layer_name + "_gradient",
    )
    gradient.valid <<= True
    return gradient

np.random.seed(42)

bitwidth = 16

print("Building PyrtlTraining...")
pyrtl_training = PyrtlTraining(bitwidth=bitwidth)
print("PyrtlTraining built.")

#print("Generating Verilog...")
#with open("pyrtl_training.v", "w") as f:
#  pyrtl.output_to_verilog(f)
#print("Verilog generation done.")

mnist_test_data_file = "./mnist_test_data.npz"
# Load MNIST test data.
mnist_test_data = np.load(str(mnist_test_data_file))
test_images = mnist_test_data.get("test_images")
test_labels = mnist_test_data.get("test_labels")

test_image, test_label = test_images[0], test_labels[0]

pyrtl_training.simulate(flat_image=test_image)