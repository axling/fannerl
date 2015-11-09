#include <ei.h>

#include <unistd.h>

#ifdef __APPLE__
#include <sys/uio.h>
#else
#include <sys/io.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "doublefann.h"
#include "uthash.h"
 
#define BUF_SIZE 128 
 
typedef unsigned char byte;

int read_cmd(byte **buf, int *size);
int write_cmd(ei_x_buff* x);
int read_exact(byte *buf, int len);
int write_exact(byte *buf, int len);

int do_fann_create_standard(byte *buf, int * index, ei_x_buff * result);
int do_fann_create_from_file(byte *buf, int * index, ei_x_buff * result);
int do_fann_copy(byte *buf, int * index, ei_x_buff * result);
int do_fann_destroy(byte *buf, int * index, ei_x_buff * result);
int do_fann_train(byte *buf, int * index, ei_x_buff * result);
int do_fann_run(byte *buf, int * index, ei_x_buff * result);
int do_fann_train_on_file(byte *buf, int * index, ei_x_buff * result);
int do_fann_get_params(byte*buf, int * index, ei_x_buff * result);
int do_fann_read_train_from_file(byte*buf, int * index, ei_x_buff * result);
int do_fann_train_on_data(byte*buf, int * index, ei_x_buff * result);
int do_fann_train_epoch(byte*buf, int * index, ei_x_buff * result);
int do_fann_test_data(byte*buf, int * index, ei_x_buff * result);
int do_fann_test(byte*buf, int * index, ei_x_buff * result);
int do_fann_save_to_file(byte*buf, int * index, ei_x_buff * result);
int do_fann_shuffle_train(byte*buf, int * index, ei_x_buff * result);
int do_fann_subset_train_data(byte*buf, int * index, ei_x_buff * result);
int do_fann_randomize_weights(byte*buf, int * index, ei_x_buff * result);
int do_fann_init_weights(byte*buf, int * index, ei_x_buff * result);
int do_fann_reset_mse(byte*buf, int * index, ei_x_buff * result);

int get_tuple_double_data(byte * buf, int * index, double * inputs,
			  unsigned int num_inputs);
double get_double(byte * buf, int * index);
int get_fann_ptr(byte * buf, int * index, struct fann** fann);
int get_fann_train_ptr(byte * buf, int * index, struct fann_train_data** fann);

int traverse_create_options(byte * buf, int * index, struct fann ** network);
int get_activation_function(char * activation_function);


struct ann_map {
  int key;
  struct fann * ann;
  UT_hash_handle hh; // makes this structure hashable
};

struct train_map {
  int key;
  struct fann_train_data * train;
  UT_hash_handle hh; // makes this structure hashable
};

struct ann_map * ann_map = NULL;
struct train_map * train_map = NULL;

int ann_ctr=0;
int train_ctr=0;

void add_ann(int ann_num, struct fann * ann) {
  struct ann_map * s;
  HASH_FIND_INT(ann_map, &ann_num, s);
  if(s == NULL) {
    s = malloc(sizeof(struct ann_map));
    s->key = ann_num;
    s->ann = ann;
    HASH_ADD_INT(ann_map, key, s);
  }  
}

struct fann * find_ann(int ann_num) {
  struct ann_map * s;
  HASH_FIND_INT(ann_map, &ann_num, s);
  return s->ann;
}

void delete_ann(int ann_num) {
  struct ann_map * s;
  HASH_FIND_INT(ann_map, &ann_num, s);
  if(s != NULL) {
    HASH_DEL(ann_map, s);
    free(s);
  }
}

unsigned int num_ann() {
  return HASH_COUNT(ann_map);
}

void add_train(int train_num, struct fann_train_data * train) {
  struct train_map * s;
  HASH_FIND_INT(train_map, &train_num, s);
  if(s == NULL) {
    s = malloc(sizeof(struct train_map));
    s->key = train_num;
    s->train = train;
    HASH_ADD_INT(train_map, key, s);
  }  
}

struct train_map * find_train(int train_num) {
  struct train_map * s;
  HASH_FIND_INT(train_map, &train_num, s);
  return s;
}

void delete_train(int train_num) {
  struct train_map * s;
  HASH_FIND_INT(train_map, &train_num, s);
  if(s != NULL) {
    HASH_DEL(train_map, s);
    free(s);
  }
}

unsigned int num_train() {
  return HASH_COUNT(train_map);
}

/*-----------------------------------------------------------------
 * API functions
 *----------------------------------------------------------------*/

/*-----------------------------------------------------------------
 * MAIN
 *----------------------------------------------------------------*/
int main() {
  byte*     buf;
  int       size = BUF_SIZE;
  char      command[MAXATOMLEN];
  int       index, version, arity;

  ei_x_buff result;
 
  #ifdef _WIN32
  /* Attention Windows programmers: you need to explicitly set
   * mode of stdin/stdout to binary or else the port program won't work
   */
  setmode(fileno(stdout), O_BINARY);
  setmode(fileno(stdin), O_BINARY);
  #endif
 
  if ((buf = (byte *) malloc(size)) == NULL) 
    return -1;
     
  while (read_cmd(&buf, &size) > 0) {
    /* Reset the index, so that ei functions can decode terms from the 
     * beginning of the buffer */
    index = 0;
    /* Ensure that we are receiving the binary term by reading and
     * stripping the version byte */
    if (ei_decode_version((const char *)buf, &index, &version)) {
      return 199;
    }
    /* Our marshalling spec is that we are expecting a tuple
       {Command, Ref, Arg} or {create_standard, Arg} */
    if (ei_decode_tuple_header((const char *)buf, &index, &arity)) return 2;
     
    if (arity > 3 || arity < 2) return 3;
     
    if (ei_decode_atom((const char *)buf, &index, command)) return 4;
    // Prepare the output buffer that will hold {ok, Result} or {error, Reason}
    if (ei_x_new_with_version(&result) || ei_x_encode_tuple_header(&result, 2))
      return 5;
    if(!strcmp("create_standard", command)) {
      
      if(do_fann_create_standard(buf, &index, &result) != 1) return 9;
      
    } else if(!strcmp("create_from_file", command)) {
      
      if(do_fann_create_from_file(buf, &index, &result) != 1) return 23;
      
    } else if(!strcmp("copy", command)) {
      
      if(do_fann_copy(buf, &index, &result) != 1) return 26;
      
    } else if(!strcmp("destroy", command)) {

      if(do_fann_destroy(buf, &index, &result) != 1) return 10;

    } else if(!strcmp("train", command)) {

      if(do_fann_train(buf, &index, &result) != 1) return 13;

    } else if(!strcmp("run", command)) {

      if(do_fann_run(buf, &index, &result) != 1) return 14;

    } else if(!strcmp("train_on_file", command)) {

      if(do_fann_train_on_file(buf, &index, &result) != 1) return 15;

    } else if(!strcmp("get_params", command)) {

      if(do_fann_get_params(buf, &index, &result) != 1) return 16;

    } else if(!strcmp("read_train_from_file", command)) {

      if(do_fann_read_train_from_file(buf, &index, &result) != 1) return 17;

    } else if(!strcmp("train_on_data", command)) {

      if(do_fann_train_on_data(buf, &index, &result) != 1) return 18;

    } else if(!strcmp("train_epoch", command)) {

      if(do_fann_train_epoch(buf, &index, &result) != 1) return 19;

    } else if(!strcmp("test_data", command)) {

      if(do_fann_test_data(buf, &index, &result) != 1) return 20;

    } else if(!strcmp("test", command)) {

      if(do_fann_test(buf, &index, &result) != 1) return 21;

    } else if(!strcmp("save_to_file", command)) {

      if(do_fann_save_to_file(buf, &index, &result) != 1) return 22;

    } else if(!strcmp("shuffle_train", command)) {

      if(do_fann_shuffle_train(buf, &index, &result) != 1) return 24;

    } else if(!strcmp("subset_train_data", command)) {

      if(do_fann_subset_train_data(buf, &index, &result) != 1) return 25;

    } else if(!strcmp("randomize_weights", command)) {

      if(do_fann_randomize_weights(buf, &index, &result) != 1) return 29;

    } else if(!strcmp("init_weights", command)) {

      if(do_fann_init_weights(buf, &index, &result) != 1) return 30;

    } else if(!strcmp("reset_mse", command)) {

      if(do_fann_reset_mse(buf, &index, &result) != 1) return 31;

    } else {
      if (ei_x_encode_atom(&result, "error") ||
	  ei_x_encode_atom(&result, "unsupported_command")) 
        return 99;
    }
    write_cmd(&result);
 
    ei_x_free(&result);
  }
  return 0;
}

/*-----------------------------------------------------------------
 * API functions
 *----------------------------------------------------------------*/
int do_fann_create_standard(byte *buf, int * index, ei_x_buff * result) {
  int arity, size;
  char type[MAXATOMLEN];
  // decode {Network, Type, ConnRate, Options}
  if(ei_decode_tuple_header((const char*)buf, index, &size)) return -1;
  if(ei_decode_tuple_header((const char*)buf, index, &arity)) return -1;
  unsigned int layers[arity];
  unsigned long layerNum = 0;
  struct fann * network;
  // Here we dynamically create an array to use in fann_create_standard
  // arity is the same as num_layers
  for(int i = 0; i < arity; ++i) {
    if(ei_decode_ulong((const char *)buf, index, &layerNum)) return -1;
    layers[i] = (unsigned int)layerNum;
  }
  // Decode type
  if(ei_decode_atom((const char *)buf, index, type)) return -1;
  if(!strcmp("standard", type)) {
    network = fann_create_standard_array(arity, layers);
    ei_skip_term((const char*)buf, index);
  } else if(!strcmp("sparse", type)) {
    double temp_conn_rate;
    if(ei_decode_double((const char *)buf, index, &temp_conn_rate)) return -1;
    network = fann_create_sparse_array(temp_conn_rate, arity, layers);
  } else if(!strcmp("shortcut", type)) {
    network = fann_create_shortcut_array(arity, layers);
    ei_skip_term((const char*)buf, index);
  } else {
    // standard course of action
    network = fann_create_standard_array(arity, layers);
    ei_skip_term((const char*)buf, index);
  }
  if(traverse_create_options(buf, index, &network) != 1) return -1;
  int hash_key = ann_ctr;
  add_ann(ann_ctr, network);
  ann_ctr += 1;
  if(ei_x_encode_atom(result, "ok") || ei_x_encode_long(result,
							(long)hash_key))
    return -1;
  return 1;
}

int traverse_create_options(byte * buf, int * index, struct fann ** network){
  int arity=0;
  int type, type_size;
  char key[MAXATOMLEN];
  char activation_function[MAXATOMLEN];
  ei_print_term(stderr, (const char *)buf, index);
  if(ei_decode_map_header((const char *)buf, index, &arity)) return -1;
  if(arity == 0) return 1;
  // Go through the map
  for(int i = 0; i < arity; ++i) {
    // keys should be atom
    if(ei_get_type((const char *)buf, index, &type, &type_size)) return -1;
    if(type != ERL_ATOM_EXT && type != ERL_SMALL_ATOM_EXT &&
       type != ERL_ATOM_UTF8_EXT && type != ERL_SMALL_ATOM_UTF8_EXT) {
      ei_skip_term((const char*)buf, index);
      ei_skip_term((const char*)buf, index);
      continue;
    }
    if(ei_decode_atom((const char *)buf, index, key)) return -1;
    
    if(!strcmp("learning_rate", key)) {
      float learning_rate;
      if(ei_get_type((const char *)buf, index, &type, &type_size)) return -1;
      if(type == ERL_INTEGER_EXT) {
	long temp;
	if(ei_decode_long((const char *)buf, index, &temp)) return -1;
	learning_rate = (float)temp;
	fann_set_learning_rate(*network, learning_rate);
      } else if(type == ERL_FLOAT_EXT || type == NEW_FLOAT_EXT) {
	double temp;
	if(ei_decode_double((const char *)buf, index, &temp)) return -1;
	learning_rate = (float)(temp);
	fann_set_learning_rate(*network, learning_rate);
      } else {
	ei_skip_term((const char*)buf, index);
      }
    } else if(!strcmp("activation_func_input", key)) {
      if(ei_get_type((const char *)buf, index, &type, &type_size)) return -1;
      if(type == ERL_ATOM_EXT || type == ERL_SMALL_ATOM_EXT ||
	 type == ERL_ATOM_UTF8_EXT || type == ERL_SMALL_ATOM_UTF8_EXT)  {
	if(ei_decode_atom((const char *)buf, index, activation_function))
	  return -1;
	int act_func = -1;
	if((act_func = get_activation_function(activation_function)) == -1)
	  return -1;
	fann_set_activation_function_layer(*network, act_func, 0);
      } else {
	ei_skip_term((const char*)buf, index);
      }
    } else if(!strcmp("activation_func_hidden", key)) {
      if(ei_get_type((const char *)buf, index, &type, &type_size)) return -1;
      if(type == ERL_ATOM_EXT || type == ERL_SMALL_ATOM_EXT ||
	 type == ERL_ATOM_UTF8_EXT || type == ERL_SMALL_ATOM_UTF8_EXT)  {
	if(ei_decode_atom((const char *)buf, index, activation_function))
	  return -1;
	int act_func = -1;
	if((act_func = get_activation_function(activation_function)) == -1)
	  return -1;
	fann_set_activation_function_hidden(*network, act_func);
      } else {
	ei_skip_term((const char*)buf, index);
      }
    } else if(!strcmp("activation_func_output", key)) {
      if(ei_get_type((const char *)buf, index, &type, &type_size)) return -1;
      if(type == ERL_ATOM_EXT || type == ERL_SMALL_ATOM_EXT ||
	 type == ERL_ATOM_UTF8_EXT || type == ERL_SMALL_ATOM_UTF8_EXT)  {
	if(ei_decode_atom((const char *)buf, index, activation_function))
	  return -1;
	int act_func = -1;
	if((act_func = get_activation_function(activation_function)) == -1)
	  return -1;
	fann_set_activation_function_output(*network, act_func);
      } else {
	ei_skip_term((const char*)buf, index);
      }      
    } else if(!strcmp("activation_func_layer", key)) {
      // Decode {Layer, ActivationFunction}
      int layer_tuple = 0;
      unsigned long layer = 0;
      if(ei_decode_tuple_header((const char*)buf, index, &layer_tuple))
	return -1;

      if(ei_decode_ulong((const char*)buf, index, &layer)) return -1;
      // Sanity check size of layer, must be less than number of layers
      unsigned int num_layers = fann_get_num_layers(*network);
      if( (unsigned int)layer >= num_layers ) return -1;

      //Decode activation function
      if(ei_get_type((const char *)buf, index, &type, &type_size)) return -1;
      if(type == ERL_ATOM_EXT || type == ERL_SMALL_ATOM_EXT ||
	 type == ERL_ATOM_UTF8_EXT || type == ERL_SMALL_ATOM_UTF8_EXT)  {
	if(ei_decode_atom((const char *)buf, index, activation_function))
	  return -1;
	int act_func = -1;
	if((act_func = get_activation_function(activation_function)) == -1)
	  return -1;
	fann_set_activation_function_layer(*network, act_func, layer);
      } else {
	ei_skip_term((const char*)buf, index);
      }      
    } else {
      ei_skip_term((const char*)buf, index);
    }
  }
  return 1;
}

int do_fann_create_from_file(byte *buf, int * index, ei_x_buff * result) {
  struct fann * network = NULL;
  char * filename = NULL;
  int size, type;

  // Decode Filename
  if(ei_get_type((const char *)buf, index, &type, &size)) return -1;
  if(type != ERL_STRING_EXT) return -1;
  filename = malloc((size+1)*sizeof(char));

  if(ei_decode_string((const char *)buf, index, filename)) return -1;

  network = fann_create_from_file(filename);
  if(network == NULL) return -1;

  int hash_key = ann_ctr;
  add_ann(ann_ctr, network);
  ann_ctr += 1;
  if(ei_x_encode_atom(result, "ok") || ei_x_encode_long(result,
							(long)hash_key))
    return -1;
  return 1;

}

int do_fann_copy(byte *buf, int * index, ei_x_buff * result) {
  struct fann * oldNetwork = NULL;
  struct fann * newNetwork = NULL;

  if(get_fann_ptr(buf, index, &oldNetwork) != 1) return -1;
    
  newNetwork = fann_copy(oldNetwork);
  if(newNetwork == NULL) return -1;

  int hash_key = ann_ctr;
  add_ann(ann_ctr, newNetwork);
  ann_ctr += 1;
  if(ei_x_encode_atom(result, "ok") || ei_x_encode_long(result,
							(long)hash_key))
    return -1;
  return 1;

}

int do_fann_destroy(byte *buf, int * index, ei_x_buff * result) {
  struct fann * network = 0;
  // decode Ptr, {} (skip empty tuple)
  // Decode network ptr first
  if(get_fann_ptr(buf, index, &network) != 1) return -1;
  ei_skip_term((const char * )buf, index);
  fann_destroy(network);
  if(ei_x_new_with_version(result) ||
     ei_x_encode_atom_len(result, "ok", 2)) return -1;
  return 1;
}

int do_fann_train(byte *buf, int * index, ei_x_buff * result) {
  unsigned int num_inputs, num_outputs;
  int arity;
  struct fann * network=0;

  // Decode Ptr,{Input, Output}
  // Decode network ptr first
  if(get_fann_ptr(buf, index, &network) != 1) return -1;

  if(ei_decode_tuple_header((const char *)buf, index, &arity)) return -1;

  num_inputs = fann_get_num_input(network);
  num_outputs = fann_get_num_output(network);
  double inputs[num_inputs];
  //Decode the inputs, verify size
  if(get_tuple_double_data(buf, index, inputs, num_inputs) == -1) return -1;
  double outputs[num_outputs];
  if(get_tuple_double_data(buf, index, outputs, num_outputs) == -1) return -1;

  fann_train(network, (fann_type*)inputs, (fann_type *) outputs);

  if(ei_x_new_with_version(result) ||
     ei_x_encode_atom_len(result, "ok", 2)) return -1;
  return 1;
}

int do_fann_train_on_file(byte *buf, int * index, ei_x_buff * result) {
  int arity, type, size;
  struct fann * network = 0;
  char * filename = 0;
  unsigned long max_epochs = 0;
  double desired_error = 0.0;

  // Decode Ptr,( Filename, MaxEpochs, DesiredError}
  // Decode network ptr
  if(get_fann_ptr(buf, index, &network) != 1) return -1;
  
  if(ei_decode_tuple_header((const char *)buf, index, &arity)) return -1;
  
  if(ei_get_type((const char *)buf, index, &type, &size)) return -1;
  if(type != ERL_STRING_EXT) return -1;
  filename = malloc((size+1)*sizeof(char));
  // Decode filename
  if(ei_decode_string((const char *)buf, index, filename)) return -1;

  //Decode MaxEpochs
  if(ei_decode_ulong((const char *)buf, index, &max_epochs)) return -1;
  
  //Decode DesiredError
  if(ei_decode_double((const char *)buf, index, &desired_error)) return -1;

  //Call FANN API
  fann_train_on_file(network, filename, (unsigned int)max_epochs, 0,
		     (float)desired_error);

  free(filename);

  if(ei_x_new_with_version(result) ||
     ei_x_encode_atom_len(result, "ok", 2)) return -1;
  return 1;
}

int do_fann_run(byte *buf, int * index, ei_x_buff * result) {
  unsigned int num_inputs, num_outputs;
  int arity;
  struct fann * network = 0;

  //Decode Ptr, {Input}
  // Decode network ptr first
  if(get_fann_ptr(buf, index, &network) != 1) return -1;
  
  if(ei_decode_tuple_header((const char *)buf, index, &arity)) return -1;

  num_inputs = fann_get_num_input(network);
  num_outputs = fann_get_num_output(network);
  
  double inputs[num_inputs];
  //Decode the inputs, verify size
  if(get_tuple_double_data(buf, index,
			   inputs, num_inputs) == -1) return -1;
  double * outputs;
  outputs = fann_run(network, inputs);
  
  if(ei_x_new_with_version(result) ||
     ei_x_encode_tuple_header(result, num_outputs)) return -1;
  
  for(int i=0; i < num_outputs; ++i) {
    ei_x_encode_double(result, outputs[i]);
  }
  return 1;
}

int do_fann_get_params(byte*buf, int * index, ei_x_buff * result)  {
  struct fann * network = 0;
  float learning_rate, learning_momentum, mse, connection_rate;
  enum fann_train_enum train_alg;
  enum fann_errorfunc_enum error_func;
  enum fann_nettype_enum network_type;
  unsigned int bit_fail;
  unsigned int num_input, num_output, total_neurons;
  unsigned int total_connections;
  unsigned int num_layers;
  
  //Decode Ptr, {}
  // Decode network ptr first
  if(get_fann_ptr(buf, index, &network) != 1) return -1;
  ei_skip_term((const char*)buf, index);
  
  // fetch params
  learning_rate = fann_get_learning_rate(network);
  learning_momentum= fann_get_learning_momentum(network);
  train_alg = fann_get_training_algorithm(network);
  mse = fann_get_MSE(network);
  bit_fail = fann_get_bit_fail(network);
  error_func = fann_get_train_error_function(network);
  network_type = fann_get_network_type(network);

  num_input = fann_get_num_input(network);
  num_output = fann_get_num_output(network);
  total_neurons = fann_get_total_neurons(network);
  total_connections = fann_get_total_connections(network);

  connection_rate = fann_get_connection_rate(network);

  num_layers = fann_get_num_layers(network);

  // encode to map
  if(ei_x_new_with_version(result)) return -1;
  ei_x_encode_map_header(result, 10);
  ei_x_encode_atom(result, "learning_rate");
  ei_x_encode_double(result, (double)learning_rate);
  ei_x_encode_atom(result, "learning_momentum");
  ei_x_encode_double(result, (double)learning_momentum);
  ei_x_encode_atom(result, "training_algorithm");
  if(train_alg == FANN_TRAIN_INCREMENTAL) {
    ei_x_encode_atom(result, "fann_train_incremental");
  } else if(train_alg == FANN_TRAIN_BATCH) {
    ei_x_encode_atom(result, "fann_train_batch");
  } else if(train_alg == FANN_TRAIN_RPROP) {
    ei_x_encode_atom(result, "fann_train_rprop");
  } else if(train_alg == FANN_TRAIN_QUICKPROP) {
    ei_x_encode_atom(result, "fann_train_quickprop");
  } else {
    ei_x_encode_atom(result, "unknown_training_algorithm");
  }
  ei_x_encode_atom(result, "mean_square_error");
  ei_x_encode_double(result, (double)mse);
  ei_x_encode_atom(result, "bit_fail");
  ei_x_encode_ulong(result, (unsigned long)bit_fail);

  ei_x_encode_atom(result, "train_error_function");
  if(error_func == FANN_ERRORFUNC_LINEAR) {
    ei_x_encode_atom(result, "fann_errorfunc_linear");
  } else if(error_func == FANN_ERRORFUNC_TANH) {
    ei_x_encode_atom(result, "fann_errorfunc_tanh");
  } else {
    ei_x_encode_atom(result, "unknown_error_function");
  }
  
  ei_x_encode_atom(result, "num_input");
  ei_x_encode_ulong(result, (unsigned long)num_input);
  ei_x_encode_atom(result, "num_output");
  ei_x_encode_ulong(result, (unsigned long)num_output);
  ei_x_encode_atom(result, "total_neurons");
  ei_x_encode_ulong(result, (unsigned long)total_neurons);
  ei_x_encode_atom(result, "total_connections");
  ei_x_encode_ulong(result, (unsigned long)total_connections);

  ei_x_encode_atom(result, "network_type");
  if(error_func == FANN_NETTYPE_LAYER) {
    ei_x_encode_atom(result, "fann_nettype_layer");
  } else if(error_func == FANN_NETTYPE_SHORTCUT) {
    ei_x_encode_atom(result, "fann_nettype_shortcut");
  } else {
    ei_x_encode_atom(result, "unknown_network_type");
  }
  ei_x_encode_atom(result, "connection_rate");
  ei_x_encode_double(result, (double)connection_rate);
  
  ei_x_encode_atom(result, "num_layers");
  ei_x_encode_ulong(result, (unsigned int)num_layers);
  return 1;
}

int do_fann_read_train_from_file(byte*buf, int * index, ei_x_buff * result)  {
  int type, size;
  char * filename = 0;
  struct fann_train_data * training_data;
  if(ei_get_type((const char*)buf, index, &type, &size)) return -1;
  if(type != ERL_STRING_EXT) return -1;
  //Alloc memory for string
  filename = malloc((size +1) * sizeof(char));
  if(ei_decode_string((const char*)buf, index, filename)) return -1;
  training_data = fann_read_train_from_file(filename);
  free(filename);
  if(ei_x_new_with_version(result)) return -1;
  if(ei_x_encode_tuple_header(result, 2)) return -1;
  if(ei_x_encode_atom(result, "ok") ||
     ei_x_encode_long(result, (long)training_data))
    return -1;
  return 1;
}

int do_fann_train_on_data(byte*buf, int * index, ei_x_buff * result) {
  int arityRefs, arityArgs, type, type_size;
  struct fann * network;
  struct fann_train_data * train_data;
  unsigned long max_epochs;
  long long_value;
  double desired_error;
  // Decode {NetworkRef, TrainRef}, {MaxEpochs, DesiredError}
  if(ei_decode_tuple_header((const char *)buf, index, &arityRefs)) return -1;

  get_fann_ptr(buf, index, &network);
  get_fann_train_ptr(buf, index, &train_data);
  
  if(ei_decode_tuple_header((const char *)buf, index, &arityArgs)) return -1;
  
  if(ei_decode_ulong((const char*)buf, index, &max_epochs)) return -1;
  
  ei_get_type((const char *)buf, index, &type, &type_size);
  if(type == ERL_SMALL_INTEGER_EXT || type == ERL_INTEGER_EXT) {
    if(ei_decode_long((const char *)buf, index, &long_value)) return -1;
    desired_error = (double)long_value;
  } else if(type == ERL_FLOAT_EXT) {
    if(ei_decode_double((const char *)buf, index, &desired_error)) return -1;
  } else {
    return -1;
  }
  fann_train_on_data(network, train_data, max_epochs, 0, desired_error);
  
  if(ei_x_new_with_version(result) ||
     ei_x_encode_atom_len(result, "ok", 2)) return -1;  
  return 1;
}

int do_fann_train_epoch(byte*buf, int * index, ei_x_buff * result) {
  int arityRefs;
  struct fann * network;
  struct fann_train_data * train_data;
  // Decode {NetworkRef, TrainRef}, {MaxEpochs, DesiredError}
  if(ei_decode_tuple_header((const char *)buf, index, &arityRefs)) return -1;

  get_fann_ptr(buf, index, &network);
  get_fann_train_ptr(buf, index, &train_data);

  fann_train_epoch(network, train_data);

  if(ei_x_new_with_version(result) ||
     ei_x_encode_atom_len(result, "ok", 2)) return -1;  
  return 1;
}

int do_fann_test_data(byte*buf, int * index, ei_x_buff * result) {
  int arityRefs;
  float MSE;
  struct fann * network;
  struct fann_train_data * train_data;
  // Decode {NetworkRef, TrainRef}, {}
  if(ei_decode_tuple_header((const char *)buf, index, &arityRefs)) return -1;

  get_fann_ptr(buf, index, &network);
  get_fann_train_ptr(buf, index, &train_data);

  MSE = fann_test_data(network, train_data);
  
  if(ei_x_new_with_version(result)) return -1;
  if(ei_x_encode_tuple_header(result, 2)) return -1;
  if(ei_x_encode_atom(result, "ok") ||
     ei_x_encode_double(result, (double)MSE))
    return -1;
  return 1;
}

int do_fann_test(byte*buf, int * index, ei_x_buff * result) {
  unsigned int num_inputs, num_outputs;
  int arity;
  struct fann * network = 0;

  //Decode Ptr, {Input, DesiredOutput}
  // Decode network ptr first
  if(get_fann_ptr(buf, index, &network) != 1) return -1;
  
  if(ei_decode_tuple_header((const char *)buf, index, &arity)) return -1;

  num_inputs = fann_get_num_input(network);
  num_outputs = fann_get_num_output(network);
  
  double input[num_inputs];
  //Decode the inputs, verify size
  if(get_tuple_double_data(buf, index,
			   input, num_inputs) == -1) return -1;
  double desired_output[num_outputs];
  //Decode the desired_output, verify size
  if(get_tuple_double_data(buf, index,
			   desired_output, num_outputs) == -1) return -1;
  double * output;
  output = fann_test(network, input, desired_output);
  
  if(ei_x_new_with_version(result) ||
     ei_x_encode_tuple_header(result, num_outputs)) return -1;
  
  for(int i=0; i < num_outputs; ++i) {
    ei_x_encode_double(result, output[i]);
  }
  return 1;
}

int do_fann_save_to_file(byte*buf, int * index, ei_x_buff * result) {
  int arity, type, size;
  struct fann * network;
  char * filename = NULL;
  // Decode Ptr, {Filename}
  // Decode network ptr
  if(get_fann_ptr(buf, index, &network) != 1) return -1;
  
  if(ei_decode_tuple_header((const char *)buf, index, &arity)) return -1;
  
  if(ei_get_type((const char *)buf, index, &type, &size)) return -1;
  if(type != ERL_STRING_EXT) return -1;
  filename = malloc((size+1)*sizeof(char));
  // Decode filename
  if(ei_decode_string((const char *)buf, index, filename)) return -1;

  if(fann_save(network, filename) == -1) {
    free(filename);
    return -1;
  }

  free(filename);
    
  if(ei_x_new_with_version(result) ||
     ei_x_encode_atom_len(result, "ok", 2)) return -1;

  return 1;
}

int do_fann_shuffle_train(byte*buf, int * index, ei_x_buff * result) {
  struct fann_train_data * train;
  // Decode trainPtr, {}
  if(get_fann_train_ptr(buf, index, &train) != 1) return -1;

  // Skip the {} part
  ei_skip_term((const char*)buf, index);

  fann_shuffle_train_data(train);
  
  if(ei_x_new_with_version(result) ||
     ei_x_encode_atom_len(result, "ok", 2)) return -1;
  
  return 1;
}

int do_fann_subset_train_data(byte*buf, int * index, ei_x_buff * result) {
  struct fann_train_data * train;
  struct fann_train_data * copy;
  unsigned long pos, length;
  int arity;
  // Decode trainPtr, {Pos, Length}
  if(get_fann_train_ptr(buf, index, &train) != 1) return -1;

  if(ei_decode_tuple_header((const char *)buf, index, &arity)) return -1;
  
  if(ei_decode_ulong((const char *)buf, index, &pos)) return -1;
  if(ei_decode_ulong((const char *)buf, index, &length)) return -1;

  copy = fann_subset_train_data(train, pos, length);

  if(ei_x_new_with_version(result)) return -1;
  if(ei_x_encode_tuple_header(result, 2)) return -1;
  if(ei_x_encode_atom(result, "ok") ||
     ei_x_encode_long(result, (long)copy))
    return -1;
  return 1;
}

int do_fann_randomize_weights(byte*buf, int * index, ei_x_buff * result) {
  fann_type min_weight, max_weight;
  struct fann * network = 0;
  int arity;
    
  // Decode Ptr, {MinWeight, MaxWeight}
  // Decode network ptr
  if(get_fann_ptr(buf, index, &network) != 1) return -1;
  
  if(ei_decode_tuple_header((const char *)buf, index, &arity)) return -1;
    
  //Decode MinWeight
  min_weight = get_double(buf, index);
  max_weight = get_double(buf, index);
  
  fann_randomize_weights(network, min_weight, max_weight);
    
  if(ei_x_new_with_version(result) ||
     ei_x_encode_atom_len(result, "ok", 2)) return -1;

  return 1;
}

int do_fann_init_weights(byte*buf, int * index, ei_x_buff * result) {
  struct fann * network;
  struct fann_train_data * train_data;
  int arity;

  // Decode {NetworkRef, TrainRef}, {}
  if(ei_decode_tuple_header((const char *)buf, index, &arity)) return -1;

  get_fann_ptr(buf, index, &network);
  get_fann_train_ptr(buf, index, &train_data);
  
  fann_init_weights(network, train_data);
    
  if(ei_x_new_with_version(result) ||
     ei_x_encode_atom_len(result, "ok", 2)) return -1;

  return 1;
}

int do_fann_reset_mse(byte*buf, int * index, ei_x_buff * result) {
  struct fann * network = 0;
  //Decode Ptr, {}
  // Decode network ptr first
  if(get_fann_ptr(buf, index, &network) != 1) return -1;
  
  fann_reset_MSE(network);

  if(ei_x_new_with_version(result) ||
     ei_x_encode_atom_len(result, "ok", 2)) return -1;

  return 1;
}

/*-----------------------------------------------------------------
 * Util functions
 *----------------------------------------------------------------*/
int get_tuple_double_data(byte * buf, int * index, double * data,
			  unsigned int size) {
  int arity;

  if(ei_decode_tuple_header((const char *)buf, index, &arity)) return -1;
  if(arity != size) return -1;
  
  for(int i = 0; i < arity; ++i) {
    data[i] = get_double(buf, index);
  }
  return 1;
}

double get_double(byte*buf, int * index) {
  int type, type_size;
  double double_value;
  long long_value;
  ei_get_type((const char *)buf, index, &type, &type_size);
    if(type == ERL_SMALL_INTEGER_EXT || type == ERL_INTEGER_EXT) {
      if(ei_decode_long((const char *)buf, index, &long_value)) return -1;
      return (double)long_value;
    } else {
      if(ei_decode_double((const char *)buf, index, &double_value)) return -1;
      return double_value;
    }
    
}

int get_fann_ptr(byte * buf, int * index, struct fann ** fann) {
  // Decode the network ptr
  unsigned long hash_key = 0;
  if(ei_decode_ulong((const char *)buf, index, &hash_key)) return -1;
  *fann = find_ann((int)hash_key);
  return 1;
}
int get_fann_train_ptr(byte * buf, int * index,
		       struct fann_train_data ** fann_train_data) {
  // Decode the training data ptr
  unsigned long ptr = 0;
  if(ei_decode_ulong((const char *)buf, index, &ptr)) return -1;
  *fann_train_data = (struct fann_train_data *)ptr;
  return 1;
}

int get_activation_function(char * activation_function) {

  if(!strcmp("fann_linear", activation_function)) {
    return FANN_LINEAR;
  } else if(!strcmp("fann_threshold", activation_function)) {
    return FANN_THRESHOLD;
  } else if(!strcmp("fann_threshold_symmetric", activation_function)) {
    return FANN_THRESHOLD_SYMMETRIC;
  } else if(!strcmp("fann_sigmoid", activation_function)) {
    return FANN_SIGMOID;
  } else if(!strcmp("fann_sigmoid_symmetric", activation_function)) {
    return FANN_SIGMOID_SYMMETRIC;
  } else if(!strcmp("fann_sigmoid_stepwise", activation_function)) {
    return FANN_SIGMOID_STEPWISE;
  } else if(!strcmp("fann_gaussian", activation_function)) {
    return FANN_GAUSSIAN;
  } else if(!strcmp("fann_gaussian_symmetric", activation_function)) {
    return FANN_GAUSSIAN_SYMMETRIC;
  } else if(!strcmp("fann_elliot", activation_function)) {
    return FANN_ELLIOT;
  } else if(!strcmp("fann_elliot_symmetric", activation_function)) {
    return FANN_ELLIOT_SYMMETRIC;
  } else if(!strcmp("fann_linear_piece", activation_function)) {
    return FANN_LINEAR_PIECE;
  } else if(!strcmp("fann_linear_piece_symmetric", activation_function)) {
    return FANN_LINEAR_PIECE_SYMMETRIC;
  } else if(!strcmp("fann_sin_symmetric", activation_function)) {
    return FANN_SIN_SYMMETRIC;
  } else if(!strcmp("fann_cos_symmetric", activation_function)) {
    return FANN_COS_SYMMETRIC;
  } else if(!strcmp("fann_sin", activation_function)) {
    return FANN_SIN;
  } else if(!strcmp("fann_cos", activation_function)) {
    return FANN_COS;
  } else {
    return -1;
  }
}
/*-----------------------------------------------------------------
 * Data marshalling functions
 *----------------------------------------------------------------*/
int read_cmd(byte **buf, int *size)
{
  int len;
 
  if (read_exact(*buf, 2) != 2)
    return(-1);
  len = ((*buf)[0] << 8) | (*buf)[1];
  
  if (len > *size) {
    byte* tmp = (byte *) realloc(*buf, len);
    if (tmp == NULL)
      return -1;
    else
      *buf = tmp;
    *size = len;
  }
  return read_exact(*buf, len);
}
 
int write_cmd(ei_x_buff *buff)
{
  byte li;
 
  li = (buff->index >> 8) & 0xff; 
  write_exact(&li, 1);
  li = buff->index & 0xff;
  write_exact(&li, 1);
 
  return write_exact((byte*)buff->buff, buff->index);
}
 
int read_exact(byte *buf, int len)
{
  int i, got=0;
 
  do {
    if ((i = read(3, buf+got, len-got)) <= 0)
      return i;
    got += i;
  } while (got<len);
 
  return len;
}
 
int write_exact(byte *buf, int len)
{
  int i, wrote = 0;
 
  do {
    if ((i = write(4, buf+wrote, len-wrote)) <= 0)
      return i;
    wrote += i;
  } while (wrote<len);
 
  return len;
}

