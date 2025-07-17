#![allow(unused)]
use rand::Rng;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

const SQRT_TIMES: usize = 100;
const SQUARE_TIMES: usize = 100;
const SQRT_TRAINING_LOOPS: usize = 200000;
const SQUARE_TRAINING_LOOPS: usize = 200000;
const SQRT_LEARNING_RATE: f64 = 4.0;
const SQUARE_LEARNING_RATE: f64 = 6.0;
const TEST_COUNT: usize = 50;
const PROGRESS_BAR_LENGTH: usize = 50;
const LOOKUP_SIZE: usize = 4096;
const SIGMOID_DOM_MIN: f64 = -15.0;
const SIGMOID_DOM_MAX: f64 = 15.0;

const ANSI_RESET: &str = "\x1b[0m";
const ANSI_BOLD: &str = "\x1b[1m";
const ANSI_UNDERLINE: &str = "\x1b[4m";
const ANSI_RED: &str = "\x1b[31m";
const ANSI_GREEN: &str = "\x1b[32m";
const ANSI_YELLOW: &str = "\x1b[33m";
const ANSI_BLUE: &str = "\x1b[34m";
const ANSI_BRIGHT_RED: &str = "\x1b[91m";

type ActivationFunction = fn(f64) -> f64;

#[derive(Debug, Clone)]
struct NeuralNetwork {
    inputs: usize,
    hidden_layers: usize,
    hidden: usize,
    outputs: usize,
    activation_hidden: ActivationFunction,
    activation_output: ActivationFunction,
    weights: Vec<f64>,
    outputs_buffer: Vec<f64>,
    delta: Vec<f64>,
    sigmoid_lookup: [f64; LOOKUP_SIZE],
    sigmoid_interval: f64,
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
            weights: vec![0.0; total_weights],
            outputs_buffer: vec![0.0; total_neurons],
            delta: vec![0.0; total_neurons - inputs],
            sigmoid_lookup: [0.0; LOOKUP_SIZE],
            sigmoid_interval: 0.0,
        };

        nn.randomize();
        nn.init_sigmoid_lookup();

        Some(nn)
    }

    fn randomize(&mut self) {
        let mut rng = rand::rng();
        for weight in &mut self.weights {
            *weight = rng.random::<f64>() - 0.5;
        }
    }

    fn init_sigmoid_lookup(&mut self) {
        let f = (SIGMOID_DOM_MAX - SIGMOID_DOM_MIN) / LOOKUP_SIZE as f64;
        self.sigmoid_interval = LOOKUP_SIZE as f64 / (SIGMOID_DOM_MAX - SIGMOID_DOM_MIN);

        for i in 0..LOOKUP_SIZE {
            let x = SIGMOID_DOM_MIN + f * i as f64;
            self.sigmoid_lookup[i] = 1.0 / (1.0 + (-x).exp());
        }
    }

    fn sigmoid_cached(&self, a: f64) -> f64 {
        if a < SIGMOID_DOM_MIN {
            return self.sigmoid_lookup[0];
        }
        if a >= SIGMOID_DOM_MAX {
            return self.sigmoid_lookup[LOOKUP_SIZE - 1];
        }

        let j = ((a - SIGMOID_DOM_MIN) * self.sigmoid_interval + 0.5) as usize;
        self.sigmoid_lookup[j.min(LOOKUP_SIZE - 1)]
    }

    fn run(&mut self, inputs: &[f64]) -> &[f64] {
        self.outputs_buffer[..self.inputs].copy_from_slice(inputs);

        let mut weight_idx = 0;
        let mut output_idx = self.inputs;

        if self.hidden_layers == 0 {
            let result_idx = output_idx;

            for _ in 0..self.outputs {
                let mut sum = self.weights[weight_idx] * -1.0;
                weight_idx += 1;

                for k in 0..self.inputs {
                    sum += self.weights[weight_idx] * self.outputs_buffer[k];
                    weight_idx += 1;
                }

                self.outputs_buffer[output_idx] = (self.activation_output)(sum);
                output_idx += 1;
            }

            return &self.outputs_buffer[result_idx..result_idx + self.outputs];
        }

        for _ in 0..self.hidden {
            let mut sum = self.weights[weight_idx] * -1.0;
            weight_idx += 1;

            for k in 0..self.inputs {
                sum += self.weights[weight_idx] * self.outputs_buffer[k];
                weight_idx += 1;
            }

            self.outputs_buffer[output_idx] = (self.activation_hidden)(sum);
            output_idx += 1;
        }

        let mut input_idx = self.inputs;
        for _ in 1..self.hidden_layers {
            for _ in 0..self.hidden {
                let mut sum = self.weights[weight_idx] * -1.0;
                weight_idx += 1;

                for k in 0..self.hidden {
                    sum += self.weights[weight_idx] * self.outputs_buffer[input_idx + k];
                    weight_idx += 1;
                }

                self.outputs_buffer[output_idx] = (self.activation_hidden)(sum);
                output_idx += 1;
            }
            input_idx += self.hidden;
        }

        let result_idx = output_idx;
        for _ in 0..self.outputs {
            let mut sum = self.weights[weight_idx] * -1.0;
            weight_idx += 1;

            for k in 0..self.hidden {
                sum += self.weights[weight_idx] * self.outputs_buffer[input_idx + k];
                weight_idx += 1;
            }

            self.outputs_buffer[output_idx] = (self.activation_output)(sum);
            output_idx += 1;
        }

        &self.outputs_buffer[result_idx..result_idx + self.outputs]
    }

    fn train(&mut self, inputs: &[f64], desired_outputs: &[f64], learning_rate: f64) {
        self.run(inputs);

        let output_layer_start_idx = self.inputs + self.hidden * self.hidden_layers;
        let output_delta_start_idx = self.hidden * self.hidden_layers;

        for (output_idx, &target) in desired_outputs.iter().enumerate().take(self.outputs) {
            let neuron_output = self.outputs_buffer[output_layer_start_idx + output_idx];
            let error = target - neuron_output;

            self.delta[output_delta_start_idx + output_idx] =
                if std::ptr::fn_addr_eq(self.activation_output, linear as fn(f64) -> f64) {
                    error
                } else {
                    error * neuron_output * (1.0 - neuron_output)
                };
        }

        // Backpropagate through hidden layers
        for layer_idx in (0..self.hidden_layers).rev() {
            let current_layer_start = layer_idx * self.hidden;
            let next_layer_start = (layer_idx + 1) * self.hidden;

            for neuron_idx in 0..self.hidden {
                let neuron_output =
                    self.outputs_buffer[self.inputs + current_layer_start + neuron_idx];
                let mut error_sum = 0.0;

                let next_layer_size = if layer_idx == self.hidden_layers - 1 {
                    self.outputs // Next layer is output layer
                } else {
                    self.hidden // Next layer is another hidden layer
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
                        // Weights between two hidden layers
                        (self.inputs + 1) * self.hidden
                            + layer_idx * (self.hidden + 1) * self.hidden
                            + next_layer_neuron * (self.hidden + 1)
                            + (neuron_idx + 1)
                    };

                    let connecting_weight = self.weights[weight_index];
                    error_sum += next_layer_delta * connecting_weight;
                }

                self.delta[current_layer_start + neuron_idx] =
                    neuron_output * (1.0 - neuron_output) * error_sum;
            }
        }

        let mut weight_index = if self.hidden_layers > 0 {
            (self.inputs + 1) * self.hidden
                + (self.hidden_layers - 1) * (self.hidden + 1) * self.hidden
        } else {
            0 // No hidden layers, update input-to-output weights directly
        };

        let last_hidden_layer_output_start = if self.hidden_layers > 0 {
            self.inputs + self.hidden * (self.hidden_layers - 1)
        } else {
            0 // No hidden layers, use inputs directly
        };

        for output_neuron_idx in 0..self.outputs {
            let delta = self.delta[self.hidden * self.hidden_layers + output_neuron_idx];

            // Update bias weight
            self.weights[weight_index] += delta * learning_rate * -1.0;
            weight_index += 1;

            // Update weights from last layer neurons
            let preceding_layer_size = if self.hidden_layers > 0 {
                self.hidden
            } else {
                self.inputs
            };

            for preceding_neuron_idx in 0..preceding_layer_size {
                let preceding_neuron_output =
                    self.outputs_buffer[last_hidden_layer_output_start + preceding_neuron_idx];
                self.weights[weight_index] += delta * learning_rate * preceding_neuron_output;
                weight_index += 1;
            }
        }

        // Update weights between hidden layers (and input to first hidden layer)
        for layer_idx in (0..self.hidden_layers).rev() {
            let preceding_layer_output_start = if layer_idx == 0 {
                0 // Input layer
            } else {
                self.inputs + (layer_idx - 1) * self.hidden
            };

            let layer_weight_start_index = if layer_idx == 0 {
                0 // Input-to-first-hidden weights
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
                self.weights[bias_weight_index] += delta * learning_rate * -1.0;

                // Update weights from preceding layer
                for preceding_neuron_idx in 0..if layer_idx == 0 {
                    self.inputs
                } else {
                    self.hidden
                } {
                    let weight_index = bias_weight_index + preceding_neuron_idx + 1;
                    let preceding_neuron_output =
                        self.outputs_buffer[preceding_layer_output_start + preceding_neuron_idx];
                    self.weights[weight_index] += delta * learning_rate * preceding_neuron_output;
                }
            }
        }
    }

    fn average_from(&mut self, networks: &[NeuralNetwork]) {
        if networks.is_empty() {
            return;
        }

        let num_networks = networks.len() as f64;
        for i in 0..self.weights.len() {
            self.weights[i] = networks.iter().map(|net| net.weights[i]).sum::<f64>() / num_networks;
        }
    }
}

fn sigmoid(a: f64) -> f64 {
    if a < -45.0 {
        0.0
    } else if a > 45.0 {
        1.0
    } else {
        1.0 / (1.0 + (-a).exp())
    }
}

fn linear(a: f64) -> f64 {
    a
}

fn threshold(a: f64) -> f64 {
    if a > 0.0 {
        1.0
    } else {
        0.0
    }
}

fn print_loading_bar(progress: f64) {
    let filled_length = (progress * PROGRESS_BAR_LENGTH as f64) as usize;
    print!("[");
    for i in 0..PROGRESS_BAR_LENGTH {
        if i < filled_length {
            print!("#");
        } else {
            print!(" ");
        }
    }
    print!("] {:.0}%\r", progress * 100.0);
    std::io::stdout().flush().unwrap();
}

fn abs_diff(a: f64, b: f64) -> f64 {
    (a - b).abs()
}

fn print_timestamp() {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let local = chrono::DateTime::from_timestamp(now as i64, 0);
    println!("[{}]: ", local.unwrap().format("%Y-%m-%d %H:%M:%S"));
}

fn print_info(msg: &str) {
    println!("[{}INFO{}]: {}", ANSI_YELLOW, ANSI_RESET, msg);
}

fn print_data(msg: &str) {
    println!("[{}DATA{}]: {}", ANSI_BRIGHT_RED, ANSI_RESET, msg);
}

use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::{
    sync::{Arc, Mutex},
    thread,
};

fn main() {
    let rng = Arc::new(Mutex::new(rand::rng()));
    let mut test = vec![0.0; TEST_COUNT];
    for test in test.iter_mut().take(TEST_COUNT) {
        *test = rng.lock().unwrap().random::<f64>() * 10.0;
    }

    let sqrt_input: Vec<f64> = (0..SQRT_TIMES).map(|i| i as f64 / 10.0).collect();
    let sqrt_output: Vec<f64> = sqrt_input
        .iter()
        .map(|&x| x.sqrt() * (1.0 / SQRT_TIMES as f64))
        .collect();

    let square_input: Vec<f64> = (0..SQUARE_TIMES)
        .map(|i| (i as f64 / SQUARE_TIMES as f64) * 10.0)
        .collect();
    let square_output: Vec<f64> = square_input
        .iter()
        .map(|&x| x.powi(2) * (1.0 / SQUARE_TIMES as f64))
        .collect();

    let net1 = Arc::new(Mutex::new(NeuralNetwork::new(1, 1, 13, 1).unwrap()));
    let net2 = Arc::new(Mutex::new(NeuralNetwork::new(1, 1, 13, 1).unwrap()));
    let num_threads = num_cpus::get();

    let pb = ProgressBar::new(
        ((SQUARE_TRAINING_LOOPS / num_threads + SQRT_TRAINING_LOOPS / num_threads) * num_threads)
            as u64,
    );
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} Iters")
            .unwrap()
            .progress_chars("█░"),
    );

    let threads: Vec<_> = (0..num_threads)
        .map(|_| {
            let net1_clone = Arc::clone(&net1);
            let net2_clone = Arc::clone(&net2);
            let sqrt_input = sqrt_input.clone();
            let sqrt_output = sqrt_output.clone();
            let square_input = square_input.clone();
            let square_output = square_output.clone();
            let pb = pb.clone();

            thread::spawn(move || {
                let mut net1_local = net1_clone.lock().unwrap().clone();
                let mut net2_local = net2_clone.lock().unwrap().clone();

                for _ in 0..(SQRT_TRAINING_LOOPS / num_threads) {
                    for j in 0..SQRT_TIMES {
                        net1_local.train(&[sqrt_input[j]], &[sqrt_output[j]], SQRT_LEARNING_RATE);
                    }
                    pb.inc(1);
                }

                for _ in 0..(SQUARE_TRAINING_LOOPS / num_threads) {
                    for j in 0..SQUARE_TIMES {
                        net2_local.train(
                            &[square_input[j]],
                            &[square_output[j]],
                            SQUARE_LEARNING_RATE,
                        );
                    }
                    pb.inc(1);
                }

                (net1_local, net2_local)
            })
        })
        .collect();

    let mut nets1_results = vec![];
    let mut nets2_results = vec![];

    for handle in threads {
        let (net1_trained, net2_trained) = handle.join().unwrap();
        nets1_results.push(net1_trained);
        nets2_results.push(net2_trained);
    }

    pb.finish_with_message("Training Complete!");

    let mut final_net1 = net1.lock().unwrap();
    let mut final_net2 = net2.lock().unwrap();
    final_net1.average_from(&nets1_results);
    final_net2.average_from(&nets2_results);

    loop {
        println!("\nChoose an option:");
        println!("1. Estimate square root");
        println!("2. Estimate square");
        print!("3. Exit\n{} ", ">".green());
        std::io::stdout().flush().unwrap();
        let mut choice = String::new();
        std::io::stdin()
            .read_line(&mut choice)
            .expect("Failed to read line");
        let choice = choice.trim();

        match choice {
            "1" => {
                print!("{} ", "->".green());
                std::io::stdout().flush().unwrap();
                let mut input = String::new();
                std::io::stdin()
                    .read_line(&mut input)
                    .expect("Failed to read line");
                if let Ok(num) = input.trim().parse::<f64>() {
                    let est_sqrt = final_net1.run(&[num])[0] * SQRT_TIMES as f64;
                    let actual_sqrt = num.sqrt();
                    let diff_sqrt =
                        abs_diff(est_sqrt, actual_sqrt) / ((est_sqrt + actual_sqrt) / 2.) * 100.0;

                    let diff_sqrt_colored = if diff_sqrt < 5.0 {
                        format!("{:.6}%", diff_sqrt).green()
                    } else if diff_sqrt < 15.0 {
                        format!("{:.6}%", diff_sqrt).yellow()
                    } else {
                        format!("{:.6}%", diff_sqrt).red()
                    };

                    println!(
                        "Estimated SQRT for {:<7.6} is {:<7.6} => Real is {:7.6} | off by {}",
                        num, est_sqrt, actual_sqrt, diff_sqrt_colored
                    );
                } else {
                    println!("Invalid number entered");
                }
            }
            "2" => {
                print!("{} ", "->".green());
                std::io::stdout().flush().unwrap();
                let mut input = String::new();
                std::io::stdin()
                    .read_line(&mut input)
                    .expect("Failed to read line");
                if let Ok(num) = input.trim().parse::<f64>() {
                    use std::arch::x86_64::_rdtsc;

                    let tsc = unsafe { _rdtsc() };
                    let est_square = final_net2.run(&[num])[0];
                    println!("Running took {} clock cycles", unsafe { _rdtsc() - tsc });

                    let est_square = est_square * SQUARE_TIMES as f64;

                    let actual_square = num * num;
                    let diff_square = abs_diff(est_square, actual_square)
                        / ((est_square + actual_square) / 2.)
                        * 100.0;

                    let diff_square_colored = if diff_square < 5.0 {
                        format!("{:.6}%", diff_square).green()
                    } else if diff_square < 15.0 {
                        format!("{:.6}%", diff_square).yellow()
                    } else {
                        format!("{:.6}%", diff_square).red()
                    };

                    println!(
                        "Estimated SQUARE for {:<7.6} is {:<7.6} => Real is {:7.6} | off by {}",
                        num, est_square, actual_square, diff_square_colored
                    );
                } else {
                    println!("Invalid number entered");
                }
            }
            "3" => break,
            _ => println!("Invalid option, please choose 1, 2, or 3"),
        }
    }
}
