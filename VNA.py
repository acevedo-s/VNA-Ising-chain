from gettext import translation
import tensorflow as tf
import numpy as np
import os

class Annealing:
  """
  Inputs:
  - memory_units: length of state vector
  - activation_function:
  - N_sites: number of spins in the chain
  - N_samples: number of samples for training or measuring
  - Jz: array of magnetic couplings 
  - BC: boundary conditions: 1 = PBC, 0 = OBC
  """
  def __init__(self,memory_units,activation_function,N_sites,N_samples,Jz,BC):
    self.memory_units = memory_units
    self.activation_function = activation_function
    self.N_sites = N_sites
    self.N_samples = N_samples 
    self.Jz = Jz 
    self.BC = BC
    self.set_BC() # sets the appropiate functions for the chosen BC
    self.rnn = self.get_rnn() # initializes an RNN

  def set_BC(self):
    """
    The functions 'self.samples_energy' and 'self.f_exact' are used below. They are defined here 
    differently if the user chooses periodic or open boundary conditions.
    The function 'self.exact_correlations'
    is defined here only for PBC
    """
    if (self.BC == 0):
      self.samples_energy = self.samples_energy_OBC
      self.f_exact = self.f_exact_OBC
      self.exact_correlations = self.exact_correlations_OBC
      self.s_exact = self.s_exact_OBC
    elif (self.BC == 1):
      self.samples_energy = self.samples_energy_PBC
      self.f_exact = self.f_exact_PBC
      self.exact_correlations = self.exact_correlations_PBC
      self.s_exact = self.s_exact_PBC

    return

  def f_exact_OBC(self, beta, L):
    """
    Exact free energy per site of the 1D ferromagnetic Ising chain with
    open boundary conditions and all couplings equal to 1
    """
    return (-1/beta) * (np.log(2) + (L-1) * np.log(2 * np.cosh(beta))) / L

  def f_exact_PBC(self, beta, L):
    """
    Exact free energy per site of the 1D ferromagnetic Ising chain with
    periodic boundary conditions and all couplings equal to 1
    """
    return (-1/beta) * (np.log(2) + np.log(np.cosh(beta)) + (1/L) * np.log(1 + np.tanh(beta)**L))

  def s_exact_OBC(self, beta, L):
    """
    Exact entropy per site of the 1D ferromagnetic Ising chain with
    open boundary conditions and all couplings equal to 1
    """
    return (- beta * (L-1) * np.tanh(beta) + np.log(2) + (L-1) * np.log(2 * np.cosh(beta))) / L

  def s_exact_PBC(self, beta, L):
    """
    Exact entropy per site of the 1D ferromagnetic Ising chain with
    periodic boundary conditions and all couplings equal to 1
    """
    partial_beta_log_Z = np.tanh(beta) + 1/(np.sinh(beta) * np.cosh(beta)) * \
                     (1/(1+np.cosh(beta)/np.sinh(beta)))**L 
    log_Z = np.log(2) + np.log(np.cosh(beta)) + (1/L) * np.log(1 + (np.tanh(beta))**L)
    return  - beta * partial_beta_log_Z + log_Z
  
  def get_rnn(self):
    inputs = tf.keras.Input(shape=(1,2),
                            tensor=tf.zeros(shape=(1, 2)))
    x, _ = tf.keras.layers.GRUCell(units=self.memory_units,
                                   activation=self.activation_function) \
                                  (inputs, states=tf.zeros(shape=(1,self.memory_units)))
    outputs = tf.keras.layers.Dense(
                       2, activation=tf.keras.activations.softmax)(x)
    rnn = tf.keras.Model(inputs=inputs, outputs=outputs, name='rnn')
    return rnn

  @tf.function
  def sample_rnn(self):
    """
    Returns: 
    - system_sample: (N_sites,)
    - conditionals: (N_sites,2)
    """

    conditionals = tf.TensorArray(tf.float32, size=self.N_sites)
    system_sample = tf.TensorArray(tf.int64, size=self.N_sites)

    # Initially, the recurrent cell is fed with null vectors
    rnn_output, rnn_state = self.rnn.layers[1](
                            inputs=tf.zeros(shape=(1, 1, 2)),
                            states=tf.zeros(shape=(1,self.memory_units)))
    # The dense layer is fed with the state
    output = self.rnn.layers[2](rnn_output)[0]
    conditionals = conditionals.write(0, output)
    # Binary variable drawn from the previous output probabilities
    spin_sample = tf.random.categorical(tf.math.log(output), 1)
    system_sample = system_sample.write(0, spin_sample[0][0])

    # Iterate
    for i in tf.range(1, self.N_sites):
        rnn_output, rnn_state = self.rnn.layers[1](
                                tf.one_hot(spin_sample, 2),
                                rnn_state)                     # Feed the RNN cell
        output = self.rnn.layers[2](rnn_output)[0]             # Feed the Dense layer
        conditionals = conditionals.write(i, output)           # Collect probabilities
        spin_sample = tf.random.categorical(
                      tf.math.log(output), 1)                  # Sample contitional
        system_sample = system_sample.write(i, spin_sample[0][0])

    return system_sample.stack(), conditionals.stack()

  @tf.function
  def sample_probability(self, sample, conditionals):
    """
    To get the (normalized) sample probability one must multiply all the conditional probabilities, 
    which are the final output of the RNN

    Inputs: 
    - sample: (N_sites,) a sample drawn from the RNN
    - conditionals: (N_sites,1,2) the conditional probabilities used to draw the sample 
    """
    sample_conditionals = tf.TensorArray(tf.float32, size=tf.shape(sample)[0])
    for i in tf.range(tf.shape(sample)[0]):
        sample_conditionals = sample_conditionals.write(i,
                              tf.tensordot(conditionals[i][0],
                              tf.one_hot(sample[i], 2), axes=[[0], [0]]))
    return tf.math.reduce_prod(sample_conditionals.stack())

  @tf.function
  def draw_samples(self):
    """
    This function draws 'self.N_samples' of length 
    'self.N_sites' from the 'self.rnn'

    Returns:
    - samples: (self.N_samples,self.N_sites)
    - logprobs: (self.N_samples,)
    """
    samples = tf.TensorArray(tf.int64, size=self.N_samples,
                             element_shape=(self.N_sites,))
    probs = tf.TensorArray(tf.float32, size=self.N_samples)
    
    for i in tf.range(self.N_samples):
      system_sample, conditionals = self.sample_rnn()        # Draw single sample and conditionals
      samples = samples.write(i, system_sample)              # Collect sample
      probs = probs.write(i, 
      self.sample_probability(system_sample, conditionals))  # Sample probability
    return samples.stack(), tf.math.log(probs.stack())

  @tf.function
  def samples_energy_OBC(self,samples):
    """ 
    Given the 1D Ising chain with open boundary conditions (OBC),
    obtains the energy of set of samples in parallel!

    Inputs:
    - samples: (self.N_samples,self.N_sites)

    Returns: 
    - The local energies that correspond to the samples
    Note: the energies are returned as if each spin was +1 or -1, not 0 or 1.
    """
    numsamples = tf.shape(samples)[0]
    system_size = tf.shape(samples)[1]
    energies = tf.zeros((numsamples), dtype=np.float32)

    # There are 'system_size'-1 interactions in the chain with OBC
    for i in tf.range(system_size - 1):
      values = samples[:, i] + samples[:, i+1]
      valuesC = tf.math.cos(
                tf.cast(values, dtype=tf.float32) * tf.constant(np.pi))
      energies += (-1) * tf.math.multiply(tf.cast(valuesC,
                         dtype=tf.float32), self.Jz[i])

    return energies

  @tf.function
  def samples_energy_PBC(self,samples):
    """ 
    Given the 1D Ising chain with periodic boundary conditions (PBC),
    obtains the energy of set of samples in parallel!

    Inputs:
    - samples: (self.N_samples,self.N_sites)

    Returns: 
    - The local energies that correspond to the samples
    Note: the energies are returned as if each spin was +1 or -1, not 0 or 1.
    """
    numsamples = tf.shape(samples)[0]
    system_size = tf.shape(samples)[1]
    energies = tf.zeros((numsamples), dtype=np.float32)

    # There are 'system_size' interactions in the chain with PBC
    for i in tf.range(system_size):
      values = samples[:, i] + samples[:, (i+1) % system_size]
      valuesC = tf.math.cos(
                tf.cast(values, dtype=tf.float32) * tf.constant(np.pi))
      energies += (-1) * tf.math.multiply(tf.cast(valuesC,
                         dtype=tf.float32), self.Jz[i])

    return energies

  def train_step(self,optimizer,T):
    """
    The train step consists in:
    - Draw a set of 'self.N_samples' samples from the RNN
    - Compute an estimation of the free energy at temperature T
    - Compute gradients using automatic differentiation (AD)
    - Update the network trainable parameters

    Returns:
    - Free energy estimation over the given set of samples

    For AD bibliography: 

    - https://www.tensorflow.org/guide/autodiff
    - https://github.com/acevedo-s/AD_example
    - https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
    - https://stackoverflow.com/questions/33727935/how-to-use-stop-gradient-in-tensorflow

    To understand Floc and cost, see Sec. V.B paper VNA
    """
    with tf.GradientTape() as tape:
      samples, logprobs = self.draw_samples()
      Floc = self.samples_energy(samples) + T * logprobs
      cost = tf.reduce_mean(tf.multiply(logprobs, tf.stop_gradient(Floc))) \
             - tf.reduce_mean(logprobs) * tf.reduce_mean(tf.stop_gradient(Floc))  
      grads = tape.gradient(cost, self.rnn.trainable_weights)
      optimizer.apply_gradients(zip(grads, self.rnn.trainable_weights))
    return tf.reduce_mean(Floc)

  def warm_up(self, N_warm_up, lr, flags, modelsfolder, T0):
    """
    Inputs:
    - N_warm_up: number of train_steps for the warm up
    - lr: learning rate
    - flags
    - modelsfolder: path to save the trained model
    - T0: target temperature

    Outputs: 
    - F_estimator_t: (N_warm_up,) A list with the convergence 
      of F to the equilibrium estimated value 
    """
    # Preparation
    print('starting warm up:')
    F_estimator_t = []
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    Train_step = tf.function(self.train_step) # Creates train graph
    
    # Warm up
    for i in range(N_warm_up):
      F_estimator = Train_step(optimizer,T0)
      tf.print('iteration', i + 1, '| f relative error: ', self.f_relative_error(F_estimator,T0))
      F_estimator_t.append((F_estimator/self.N_sites).numpy())

    # Saving
    if flags['save_weights']:
      os.makedirs(modelsfolder, exist_ok = True)
      self.rnn.save_weights(modelsfolder + 'T{:.2f}'.format(T0) + '_warmup.h5')

    return F_estimator_t

  def f_relative_error(self, F_estimator, T):
    """
    Inputs:
    - Estimation of the free energy computed in a set of samples
    - Temperature
    
    Returns:
    - Relative error to the exact solution
    """
    return (F_estimator / self.N_sites - self.f_exact(1./T,self.N_sites)) / abs(self.f_exact(1./T,self.N_sites))
  
  def annealing(self, N_train, lr_annealing, flags, modelsfolder,
                N_warm_up, lr_warm_up, T0, final_T, delta_T):
    """
    Inputs:

    - N_train: number of train steps for each temperature
    - lr_annealing: learning rate for the annealing
    - flags: dictionary with flags. 
      keys used: 'save_weights', 'load_weights' 
    - modelsfolder: folder where models are saved
    - N_warm_up: number of train steps for the warm up
    - lr_warm_up: learning rate for the warm up
    - T0: initial temperature to start warmp up or to load trained weights
    - final_T: final temperature
    - delta_T: temperature step 
    """
    T = T0 # variable temperature
    # Choice: load_weights at T0 or warm up to T0
    if (flags['load_weights']):
      self.load_weights(modelsfolder,T0)
      T -= delta_T
    else:
      self.F_estimator_t = self.warm_up(N_warm_up, lr_warm_up,
                                        flags, modelsfolder, T0)

    # Preparation
    print('starting annealing:')
    optimizer = tf.keras.optimizers.Adam(
                learning_rate = lr_annealing)
    Train_step = tf.function(self.train_step) # Creates train graph

    # Annealing
    while (round(T - final_T, 5) >= 0):
      print(f'T={T : .2f}')
      for i in range(N_train):
        F_estimator = Train_step(optimizer,T)
        print(f'iteration {i + 1} | relative error: {self.f_relative_error(F_estimator,T) : .6f}')
      # Saving
      if flags['save_weights']: 
        os.makedirs(modelsfolder, exist_ok = True)
        self.rnn.save_weights(modelsfolder + 'T{:.2f}'.format(T) + '.h5')
      T -= delta_T

    return

  def load_weights(self, modelsfolder, T):
    """
    Loads the weights saved in 'modelsfolder'
    belonging to a RNN trained at temperature T
    """
    self.rnn.load_weights(modelsfolder + 'T{:.2f}'.format(T) + '.h5')
    return

  def measure_f_m_s(self, T):
    """
    Inputs:
    - Temperature 

    Note: 'self.draw_samples' uses 'self.N_samples' and 'self.rnn'

    Returns:  measurements, a list of observables measured at temperature T.
    The measurement is  computed in a generated set of samples.
    The list contains:
    - average free energy estimation per site
    - associated standard deviation
    - magnetization per site
    - associated standard deviation
    - entropy per site
    - associated standard deviation
    """
    measurements = []
    samples, logprobs = self.draw_samples()
    # Free energy
    mean_f = tf.reduce_mean(self.samples_energy(samples)
                            + T * logprobs) / self.N_sites
    std_f = tf.math.reduce_std((self.samples_energy(samples)
                            + T * logprobs) ) / self.N_sites
    measurements.append(mean_f)
    measurements.append(std_f)
    # Energy
    mean_e = tf.reduce_mean(self.samples_energy(samples)) / self.N_sites
    std_e = tf.math.reduce_std(self.samples_energy(samples)) / self.N_sites
    measurements.append(mean_e)
    measurements.append(std_e)
    # Entropy
    mean_s =  - tf.reduce_mean(logprobs) / self.N_sites
    std_s = - tf.math.reduce_std(logprobs) / self.N_sites
    measurements.append(mean_s)
    measurements.append(std_s)
    ## Magnetization
    # samples = 2*tf.cast(samples, dtype = tf.float32)-1 # normalization
    # magnetizations = tf.reduce_sum(samples, axis = 1) / self.N_sites
    # mean_m = tf.reduce_mean(magnetizations) 
    # std_m = tf.math.reduce_std(magnetizations)
    # measurements.append(mean_m)
    # measurements.append(std_m)

    return measurements

  def measurements_f_m_s(self, T0, T_final, delta_T, modelsfolder):
    """
    Inputs:
    - T0: initial temperature
    - T_final: final temperature
    - delta_T: temperature step
    - modelsfolder: folder where models are saved

    Returns:  (6,N_measurements), list of observables vs. T, containing
    - means_f: mean values of the free energy per site (f)
    - stds_f: corresponding standard deviations
    - means_e: mean values of the energy per site (e)
    - stds_e: corresponding standard deviations
    - means_s: mean values of the entropy per site (s)
    - stds_s: corresponding standard deviations

    Note: N_measurements is the number of temperatures given T0, T_final and delta_T, defined below
    """
    T = T0
    N_measurements = int((T0 - T_final) / delta_T) + 1
    means_f = tf.TensorArray(tf.float32, size = N_measurements)
    stds_f = tf.TensorArray(tf.float32, size = N_measurements)
    means_e = tf.TensorArray(tf.float32, size = N_measurements)
    stds_e = tf.TensorArray(tf.float32, size = N_measurements)
    means_s = tf.TensorArray(tf.float32, size = N_measurements)
    stds_s = tf.TensorArray(tf.float32, size = N_measurements)

    for i in tf.range(N_measurements):
      self.load_weights(modelsfolder,T)
      measurements = self.measure_f_m_s(T)
      means_f = means_f.write(i, measurements[0])
      stds_f = stds_f.write(i, measurements[1])
      means_e = means_e.write(i, measurements[2])
      stds_e = stds_e.write(i, measurements[3])
      means_s = means_s.write(i, measurements[4])
      stds_s = stds_s.write(i, measurements[5])
      epsilon = (measurements[0] - self.f_exact(1/T,self.N_sites)) / self.f_exact(1/T,self.N_sites)
      tf.print('T: %.2f | mean f: %.2f | relative error: %1.2E' % (T, measurements[0], epsilon))
      T -= delta_T
    
    return [means_f.stack(), stds_f.stack(), means_e.stack(),
            stds_e.stack(), means_s.stack(), stds_s.stack()]

  @tf.function
  def correlations(self, modelsfolder, T):
    """
    Inputs:
    - modelsfolder: path to trained models
    - T: Temperature

    Returns:
    - (self.N_sites/2,) tensor of non-connected correlations <sigma_i sigma_0>
    with normalized spins sigma =  +- 1, for PBC
    - (self.N_sites,) tensor of non-connected correlations <sigma_i sigma_0>
    with normalized spins sigma =  +- 1, for OBC
    """
    if (self.BC == 1):
      N_correlations = int(self.N_sites / 2) # half a chain
    elif (self.BC ==0):
      N_correlations = self.N_sites # entire chain

    self.load_weights(modelsfolder, T)
    samples, _ = self.draw_samples()
    samples = tf.cast(2 * samples - 1, dtype = tf.int64)
    average_correlations = tf.zeros(shape = 
                           (N_correlations,), dtype = tf.int64)
    
    for i in tf.range(self.N_samples):
      sample_correlations = tf.TensorArray(tf.int64,
                            size = N_correlations)
      for j in tf.range(N_correlations):
        sample_correlations = sample_correlations.write(j,
                              samples[i][j] * samples[i][0])
      average_correlations += sample_correlations.stack()
    
    return tf.cast(average_correlations, dtype = tf.float32) / self.N_samples

  def exact_correlations_PBC(self,T):
    """
    Inputs:
    - T: Temperature
    Returns:
    - (self.N_sites / 2) np array of correlations <sigma_i sigma_0>

    Note: spins are +-1 here.
    """
    def exact_correlations_PBC_ij(T,i,j):
      """
      Exact value of <sigma_i sigma_j> with PBC and no external field,
      as in Subir Sachdev book of quantum phase transitions, chapter 2. 
      
      Inputs:
      - T: Temperature
      - i: index in the chain
      - j: index in the chain (j>i)
      """
      L = self.N_sites
      eps_1 = 2 * np.cosh(1/T)
      eps_2 = 2 * np.sinh(1/T)

      return (eps_1**(L-j+i) * eps_2**(j-i) + eps_2**(L-j+i) * eps_1**(j-i)) / (eps_1**L + eps_2**L)

    correlations = []

    for j in range(int(self.N_sites / 2)):
      correlations.append(exact_correlations_PBC_ij(T,0,j))

    return np.array(correlations)
  
  def exact_correlations_OBC(self,T):
    """
    Inputs:
    - T: Temperature
    Returns:
    - (self.N_sites) np array of correlations <sigma_i sigma_0>

    Note: spins are +-1 here.
    """
    def exact_correlations_OBC_ij(T,i,j):
      """
      Exact value of <sigma_i sigma_j> with OBC and no external field,
      computed using a transfer matrix approach.
      
      Inputs:
      - T: Temperature
      - i: index in the chain
      - j: index in the chain. (j>i)
      """
      L = self.N_sites

      return np.exp(L/T) * np.cosh(1/T)**(i-j+L) * (np.exp(1/T)*np.cosh(1/T))**(-L) * np.sinh(1/T)**(-i+j)

    correlations = []

    for j in range(int(self.N_sites)):
      correlations.append(exact_correlations_OBC_ij(T,0,j))

    return np.array(correlations)

  ############################## END OF CODE ##############################

  ### Uncompleted things (do not use them):
  
  @tf.function
  def periodic_symmetric_sample_probability(self, sample, conditionals):
    """
    Inputs:
    - sample: (N_sites,) a sample drawn from the RNN
    - conditionals: (N_sites,1,2) the conditional probabilities used to draw the sample 

    Returns:
    - symmetrized sample probability
    """
    symmetrized_sample_probability = tf.zeros(shape = ())
    for i in tf.range(self.N_sites):
      pass

    return
  
  @tf.function
  def periodic_symmetric_sample(self):
    """
    This function samples from the rnn and applies a
    random translation to the configuration. 
    No translation is a possibility, given by translation_index = 0
    Returns: 
    - system_sample: (N_sites,)
    - conditionals: (N_sites,2)
    """
    sample, conditionals = self.sample_rnn()
    translation_index = tf.random.uniform(shape=(), minval=0, 
                             maxval=self.N_sites + 1, dtype=tf.int32)
    transformed_sample = tf.TensorArray(tf.float32, size = self.N_sites)

    for i in tf.range():
      transformed_sample = transformed_sample.write(i,
                           sample[(i + translation_index) % self.N_sites ])
    
    transformed_conditionals = 0

    return transformed_sample.stack(), transformed_conditionals

  def parity_symmetric_sample_rnn(self):
    """
    This function samples from the rnn and then flips the sample with probability 1/2.
    The 'conditionals' vector of shape (self.N_sites,2) **does change**
    Returns: 
    - system_sample: (self.N_sites,)
    - conditionals: (self.N_sites,2)
    """
    sample, conditionals = self.sample_rnn()
    flip = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32)

    new_conditionals = 0 # falta hacer
    if (flip):
      return (-1) * sample, new_conditionals
    else:
      return sample, conditionals
