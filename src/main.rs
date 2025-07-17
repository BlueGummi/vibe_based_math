use rand::Rng;

const LOOKUP_SIZE: usize = 4096;
const SIGMOID_DOM_MIN: Fixed = -15 * FIXED_ONE;
const SIGMOID_DOM_MAX: Fixed = 15 * FIXED_ONE;

type ActivationFunction = fn(Fixed) -> Fixed;

type Fixed = i32;
const FIXED_SHIFT: u32 = 16;
const FIXED_ONE: Fixed = 1 << FIXED_SHIFT;
const FIXED_ZERO: Fixed = 0;

#[inline]
fn fixed_mul(a: Fixed, b: Fixed) -> Fixed {
    ((a as i64 * b as i64) >> FIXED_SHIFT) as Fixed
}

#[inline]
fn fixed_div(a: Fixed, b: Fixed) -> Fixed {
    (((a as i64) << FIXED_SHIFT) / b as i64) as Fixed
}

fn to_fixed_i(x: i32) -> Fixed {
    x << FIXED_SHIFT
}

fn sigmoid_approx(x: Fixed) -> Fixed {
    let abs_x = if x < 0 { -x } else { x };
    fixed_div(x, FIXED_ONE + abs_x)
}

#[derive(Debug, Clone)]
struct NeuralNetwork {
    inputs: usize,
    hidden_layers: usize,
    hidden: usize,
    outputs: usize,
    activation_hidden: ActivationFunction,
    activation_output: ActivationFunction,
    weights: Vec<Fixed>,
    outputs_buffer: Vec<Fixed>,
    delta: Vec<Fixed>,
    sigmoid_lookup: [Fixed; LOOKUP_SIZE],
    sigmoid_interval: Fixed,
}

impl NeuralNetwork {
    fn new(
        inputs: usize,
        hidden_layers: usize,
        hidden: usize,
        outputs: usize,
    ) -> Option<NeuralNetwork> {
        if hidden_layers > 0 && hidden < 1 {
            return None;
        }
        if inputs < 1 || outputs < 1 {
            return None;
        }

        let hidden_weights = if hidden_layers > 0 {
            (inputs + 1) * hidden + (hidden_layers - 1) * (hidden + 1) * hidden
        } else {
            0
        };

        let output_weights = if hidden_layers > 0 {
            (hidden + 1) * outputs
        } else {
            (inputs + 1) * outputs
        };

        let total_weights = hidden_weights + output_weights;
        let total_neurons = inputs + hidden * hidden_layers + outputs;

        let mut nn = NeuralNetwork {
            inputs,
            hidden_layers,
            hidden,
            outputs,
            activation_hidden: sigmoid,
            activation_output: sigmoid,
            weights: vec![0 * FIXED_ONE; total_weights],
            outputs_buffer: vec![0 * FIXED_ONE; total_neurons],
            delta: vec![0 * FIXED_ONE; total_neurons - inputs],
            sigmoid_lookup: [0 * FIXED_ONE; LOOKUP_SIZE],
            sigmoid_interval: 0 * FIXED_ONE,
        };

        nn.randomize();
        nn.init_sigmoid_lookup();

        Some(nn)
    }

    fn randomize(&mut self) {
        let mut rng = rand::rng();
        for weight in &mut self.weights {
            let rand_u32 = rng.random::<u32>();
            let rand_fixed = (rand_u32 % (FIXED_ONE as u32)) as i32 - (FIXED_ONE / 2);
            *weight = rand_fixed;
        }
    }

    fn init_sigmoid_lookup(&mut self) {
        let range = SIGMOID_DOM_MAX - SIGMOID_DOM_MIN;
        for i in 0..LOOKUP_SIZE {
            let a = range;

            let x: Fixed = SIGMOID_DOM_MIN
                + fixed_mul(
                    a,
                    to_fixed_i(i as i32) / to_fixed_i((LOOKUP_SIZE - 1) as i32),
                );
            self.sigmoid_lookup[i as usize] = sigmoid_approx(x);
        }

        self.sigmoid_interval = fixed_div(to_fixed_i(LOOKUP_SIZE as i32), range);
    }

    fn run(&mut self, inputs: &[Fixed]) -> &[Fixed] {
        // Copy inputs to the beginning of the outputs buffer
        self.outputs_buffer[..self.inputs].copy_from_slice(inputs);

        let mut weight_index = 0;
        let mut output_buffer_position = self.inputs;

        // Bias multiplier in fixed point (-1.0)
        let bias_multiplier = -FIXED_ONE;

        // Special case: no hidden layers (simple input-to-output network)
        if self.hidden_layers == 0 {
            let output_start_index = output_buffer_position;

            for _ in 0..self.outputs {
                // Start with bias weight * bias input
                let mut neuron_sum = fixed_mul(self.weights[weight_index], bias_multiplier);
                weight_index += 1;

                // Sum weighted inputs
                for input_idx in 0..self.inputs {
                    neuron_sum +=
                        fixed_mul(self.weights[weight_index], self.outputs_buffer[input_idx]);
                    weight_index += 1;
                }

                // Apply activation function and store result
                self.outputs_buffer[output_buffer_position] = (self.activation_output)(neuron_sum);
                output_buffer_position += 1;
            }

            return &self.outputs_buffer[output_start_index..output_start_index + self.outputs];
        }

        // Process first hidden layer (inputs -> first hidden)
        for _ in 0..self.hidden {
            let mut neuron_sum = fixed_mul(self.weights[weight_index], bias_multiplier);
            weight_index += 1;

            for input_idx in 0..self.inputs {
                neuron_sum += fixed_mul(self.weights[weight_index], self.outputs_buffer[input_idx]);
                weight_index += 1;
            }

            self.outputs_buffer[output_buffer_position] = (self.activation_hidden)(neuron_sum);
            output_buffer_position += 1;
        }

        // Process additional hidden layers
        let mut current_layer_input_start = self.inputs;
        for _ in 1..self.hidden_layers {
            for _ in 0..self.hidden {
                let mut neuron_sum = fixed_mul(self.weights[weight_index], bias_multiplier);
                weight_index += 1;

                for prev_neuron_idx in 0..self.hidden {
                    neuron_sum += fixed_mul(
                        self.weights[weight_index],
                        self.outputs_buffer[current_layer_input_start + prev_neuron_idx],
                    );
                    weight_index += 1;
                }

                self.outputs_buffer[output_buffer_position] = (self.activation_hidden)(neuron_sum);
                output_buffer_position += 1;
            }
            current_layer_input_start += self.hidden;
        }

        // Process output layer
        let output_start_index = output_buffer_position;
        for _ in 0..self.outputs {
            let mut neuron_sum = fixed_mul(self.weights[weight_index], bias_multiplier);
            weight_index += 1;

            for hidden_neuron_idx in 0..self.hidden {
                neuron_sum += fixed_mul(
                    self.weights[weight_index],
                    self.outputs_buffer[current_layer_input_start + hidden_neuron_idx],
                );
                weight_index += 1;
            }

            self.outputs_buffer[output_buffer_position] = (self.activation_output)(neuron_sum);
            output_buffer_position += 1;
        }

        &self.outputs_buffer[output_start_index..output_start_index + self.outputs]
    }

    fn train(&mut self, inputs: &[Fixed], desired_outputs: &[Fixed], learning_rate: Fixed) {
        // Run forward pass
        self.run(inputs);

        let output_layer_start_idx = self.inputs + self.hidden * self.hidden_layers;
        let output_delta_start_idx = self.hidden * self.hidden_layers;

        // Calculate delta for output layer
        for (output_idx, &target) in desired_outputs.iter().enumerate().take(self.outputs) {
            let neuron_output = self.outputs_buffer[output_layer_start_idx + output_idx];
            let error = target - neuron_output;

            self.delta[output_delta_start_idx + output_idx] = {
                let one_minus_o = FIXED_ONE - neuron_output;
                fixed_mul(error, fixed_mul(neuron_output, one_minus_o))
            };
        }

        // Backpropagate through hidden layers
        for layer_idx in (0..self.hidden_layers).rev() {
            let current_layer_start = layer_idx * self.hidden;
            let next_layer_start = (layer_idx + 1) * self.hidden;

            for neuron_idx in 0..self.hidden {
                let neuron_output =
                    self.outputs_buffer[self.inputs + current_layer_start + neuron_idx];
                let mut error_sum: Fixed = 0;

                let next_layer_size = if layer_idx == self.hidden_layers - 1 {
                    self.outputs
                } else {
                    self.hidden
                };

                for next_layer_neuron in 0..next_layer_size {
                    let next_layer_delta = self.delta[next_layer_start + next_layer_neuron];

                    let weight_index = if layer_idx == self.hidden_layers - 1 {
                        // Weights between last hidden layer and output layer
                        (self.inputs + 1) * self.hidden
                            + (self.hidden_layers - 1) * (self.hidden + 1) * self.hidden
                            + next_layer_neuron * (self.hidden + 1)
                            + (neuron_idx + 1)
                    } else {
                        // Weights between hidden layers
                        (self.inputs + 1) * self.hidden
                            + layer_idx * (self.hidden + 1) * self.hidden
                            + next_layer_neuron * (self.hidden + 1)
                            + (neuron_idx + 1)
                    };

                    let connecting_weight = self.weights[weight_index];
                    error_sum += fixed_mul(next_layer_delta, connecting_weight);
                }

                // delta = neuron_output * (1.0 - neuron_output) * error_sum
                let one_minus_o = FIXED_ONE - neuron_output;
                self.delta[current_layer_start + neuron_idx] =
                    fixed_mul(fixed_mul(neuron_output, one_minus_o), error_sum);
            }
        }

        // Update weights between last hidden layer and output layer (or input-output if no hidden)
        let mut weight_index = if self.hidden_layers > 0 {
            (self.inputs + 1) * self.hidden
                + (self.hidden_layers - 1) * (self.hidden + 1) * self.hidden
        } else {
            0
        };

        let last_hidden_layer_output_start = if self.hidden_layers > 0 {
            self.inputs + self.hidden * (self.hidden_layers - 1)
        } else {
            0
        };

        for output_neuron_idx in 0..self.outputs {
            let delta = self.delta[self.hidden * self.hidden_layers + output_neuron_idx];

            // Update bias weight: weights[weight_index] += delta * learning_rate * -1.0
            self.weights[weight_index] += fixed_mul(fixed_mul(delta, learning_rate), -FIXED_ONE);
            weight_index += 1;

            let preceding_layer_size = if self.hidden_layers > 0 {
                self.hidden
            } else {
                self.inputs
            };

            for preceding_neuron_idx in 0..preceding_layer_size {
                let preceding_neuron_output =
                    self.outputs_buffer[last_hidden_layer_output_start + preceding_neuron_idx];

                self.weights[weight_index] +=
                    fixed_mul(fixed_mul(delta, learning_rate), preceding_neuron_output);
                weight_index += 1;
            }
        }

        // Update weights between hidden layers (and input to first hidden layer)
        for layer_idx in (0..self.hidden_layers).rev() {
            let preceding_layer_output_start = if layer_idx == 0 {
                0
            } else {
                self.inputs + (layer_idx - 1) * self.hidden
            };

            let layer_weight_start_index = if layer_idx == 0 {
                0
            } else {
                (self.inputs + 1) * self.hidden + (layer_idx - 1) * (self.hidden + 1) * self.hidden
            };

            for neuron_idx in 0..self.hidden {
                let delta = self.delta[layer_idx * self.hidden + neuron_idx];

                let weights_per_neuron = if layer_idx == 0 {
                    self.inputs + 1
                } else {
                    self.hidden + 1
                };

                // Update bias weight
                let bias_weight_index = layer_weight_start_index + neuron_idx * weights_per_neuron;
                self.weights[bias_weight_index] +=
                    fixed_mul(fixed_mul(delta, learning_rate), -FIXED_ONE);

                for preceding_neuron_idx in 0..if layer_idx == 0 {
                    self.inputs
                } else {
                    self.hidden
                } {
                    let weight_index = bias_weight_index + preceding_neuron_idx + 1;
                    let preceding_neuron_output =
                        self.outputs_buffer[preceding_layer_output_start + preceding_neuron_idx];

                    self.weights[weight_index] +=
                        fixed_mul(fixed_mul(delta, learning_rate), preceding_neuron_output);
                }
            }
        }
    }
}

fn sigmoid(a: Fixed) -> Fixed {
    let approx = sigmoid_approx(a);
    (approx + FIXED_ONE) / 2
}
fn main() {
    let inputs = [
        [FIXED_ZERO, FIXED_ZERO],
        [FIXED_ZERO, FIXED_ONE],
        [FIXED_ONE, FIXED_ZERO],
        [FIXED_ONE, FIXED_ONE],
    ];
    let expected = [FIXED_ZERO, FIXED_ONE, FIXED_ONE, FIXED_ZERO];

    let mut net = NeuralNetwork::new(2, 1, 4, 1).unwrap(); // small network for XOR
    let learning_rate = FIXED_ONE / 4;

    for _ in 0..10_000 {
        for i in 0..inputs.len() {
            net.train(&inputs[i], &[expected[i]], learning_rate);
        }
    }

    for i in 0..inputs.len() {
        let tsc = unsafe { std::arch::x86_64::_rdtsc() };
        let output = net.run(&inputs[i])[0];
        println!("Took {} clock cycles", unsafe {
            std::arch::x86_64::_rdtsc() - tsc
        });
        let predicted = if output > FIXED_ONE / 2 { 1 } else { 0 };
        println!(
            "Input: {:?}, Output (fixed): {}, Predicted: {}, Expected: {}",
            &inputs[i],
            output,
            predicted,
            expected[i] / FIXED_ONE
        );
    }
}
