# fannerl

This is an attempt to write erlang bindings to the [FANN](http://leenissen.dk/), Fast Artificial Neural Networks, library written in C. The interfacing towards FANN is done by a port driver. 

You can watch the documentation at <http://axling.github.io/fannerl/>.

# Plan of FANN support

## Creation/Execution

| fann function               | supported by         | fannerl verson  | Comment  |
|-----------------------------|----------------------|-----------------|--------- | 
| fann_create_standard        | fannerl:create       | 1.0.0           |          |
| fann_create_standard_array  | fannerl:create       | 1.0.0           | Implicitly covered in fannerl:create   |
| fann_create_sparse          | fannerl:create       | 1.0,0           | Implicitly covered in fannerl:create   |
| fann_create_sparse_array    | fannerl:create       | 1.0.0           | Implicitly covered in fannerl:create   |
| fann_create_shortcut        | fannerl:create       | 1.0.0           | Implicitly covered in fannerl:create   |
| fann_create_shortcut_array  | fannerl:create       | 1.0.0           | Implicitly covered in fannerl:create   |
| fann_destroy                | fannerl:destroy      | 1.0.0           |    |
| fann_copy                   | fannerl:copy         | 1.0.0           |    |
| fann_run                    | fannerl:run          | 1.0.0           |    |
| fann_randomize_weights      | fannerl:randomize_weights  | 1.0.0     |    |
| fann_init_weights           | fannerl:init_weights | 1.0.0           |    |
| fann_print_connections      | -                    | 1.2.0           | Will be covered in future release |
| fann_print_parameters       | -                    | 1.2.0           | Will be covered in future release |
| fann_get_num_input          | fannerl:get_params   | 1.0.0           |    |
| fann_get_num_output         | fannerl:get_params   | 1.0.0           |    |	
| fann_get_total_neurons      | fannerl:get_params   | 1.0.0           |    |
| fann_get_total_connections  | fannerl:get_params   | 1.0.0           |    |
| fann_get_network_type       | fannerl:get_params   | 1.0.0           |    |
| fann_get_connection_rate    | fannerl:get_params   | 1.0.0           |    |
| fann_get_num_layers         | fannerl:get_params   | 1.0.0           |    |
| fann_get_layer_array        | fannerl:get_params   | 1.0.0           |    |
| fann_get_bias_array         | fannerl:get_params   | 1.0.0           |    |
| fann_get_connection_array   | fannerl:get_params   | 1.0.0           |    |
| fann_set_weight_array       | fannerl:set_weights  | 1.0.0           |    |
| fann_set_weight             | fannerl:set_weight   | 1.0.0           |    |
| fann_get_weights            | -                    | -               | Will not cover    |
| fann_set_weights            | -                    | -               | Will not cover    |
| fann_set_user_data          | -                    | -               | Will not cover    |
| fann_get_user_data          | -                    | -               | Will not cover    |
| fann_disable_seed_rand      | -                    | Later release   | Supported in fann 2.3.0  |
| fann_enable_seed_rand       | -                    | Later release   | Supported in fann 2.3.0  |
| fann_get_decimal_point      | -                    | 1.X.0           | Will be covered in future release  |
| fann_get_multiplier         | -                    | 1.X.0           | Will be covered in future release  |

## Training

| fann function               | supported by         | fannerl verson  | Comment  |
|-----------------------------|----------------------|-----------------|--------- | 
| fann_train                  | fannerl:train        | 1.0.0           |          |
| fann_test                   | fannerl:test         | 1.0.0           |          |
| fann_get_MSE                | fannerl:get_params   | 1.0.0           |          |
| fann_get_bit_fail           | fannerl:get_params   | 1.0.0           |          |
| fann_reset_MSE              | fannerl:reset_mse    | 1.0.0           |          |
| fann_train_on_data          | fannerl:train_on_data| 1.0.0           |          |
| fann_train_on_file          | fannerl:train_on_file| 1.0.0           |          |
| fann_train_epoch            | fannerl:train_epoch  | 1.0.0           |          |
| fann_test_data              | fannerl:test_data    | 1.0.0           |          |
| fann_read_train_from_file   | fannerl:read_train_from_file | 1.0.0   |          |
| fann_create_train           | -                    | -               | Will not cover   |
| fann_create_train_pointer_array | -                | Later Release   | Supported in fann 2.3.0  |
| fann_create_train_array     | -                    | Later Release   | Supported in fann 2.3.0  |
| fann_create_train_from_callback | -                | -               | Will not cover  
| fann_destroy_train          | fannerl:destroy_train| 1.0.0           |          |
| fann_get_train_input        | -                    | 1.1.0           |          |
| fann_get_train_output       | -                    | 1.1.0           |          |
| fann_shuffle_train_data     | fannerl:shuffle_train| 1.0.0           |          |
| fann_get_min_train_input    |                      | 1.0.0           | Supported in fann 2.3.0  |
| fann_get_max_train_input    |                      | 1.0.0           | Supported in fann 2.3.0  |
| fann_get_min_train_output   |                      | 1.0.0           | Supported in fann 2.3.0  |
| fann_get_max_train_output   |                      | 1.0.0           | Supported in fann 2.3.0  |
| fann_scale_train            | fannerl:scale_train  | 1.0.0           |          |
| fann_descale_train          | fannerl:descale_train| 1.0.0           |          |
| fann_set_input_scaling_params| -                   | 1.0.0           | Will be covered by fannerl:set_scaling_params |
| fann_set_output_scaling_params| -                  | 1.0.0           | Will be covered by fannerl:set_scaling_params |
| fann_set_scaling_params     | fannerl:set_scaling_params | 1.0.0     |          |
| fann_clear_scaling_params   | fannerl:clear_scaling_params| 1.0.0    |          |
| fann_scale_input            | -                    | 1.1.0           |          |
| fann_scale_output           | -                    | 1.1.0           |          |
| fann_descale_input          | -                    | 1.1.0           |          |
| fann_descale_output         | -                    | 1.1.0           |          |
| fann_scale_input_train_data | -                    | -               | Use scale_train instead   |
| fann_scale_output_train_data| -                    | -               | Use scale_train instead   |
| fann_scale_train_data       | -                    | -               | Use scale_train instead   |
| fann_merge_train_data       | fannerl:merge_train_data| 1.0.0        |    |
| fann_duplicate_train_data   | fannerl:duplicate_train_data | 1.0.0   |    |
| fann_subset_train_data      | fannerl:subset_train_data | 1.0.0      |    |
| fann_length_train_data      | fannerl:get_train_params  | 1.0.0           |    |
| fann_num_input_train_data   | fannerl:get_train_params  | 1.0.0           |    |
| fann_num_output_train_data  | fannerl:get_train_params  | 1.0.0           |    |
| fann_save_train             | fannerl:save_train   | 1.0.0           |    |
| fann_save_train_to_fixed    | -                    | -               | Will be covered in future release |
| fann_get_training_algorithm | fannerl:get_params   | 1.0.0           |    |
| fann_set_training_algorithm | JTBD                 | 1.0.0           |    |
| fann_get_learning_rate      | fannerl:get_params   | 1.0.0           |    |
| fann_set_learning_rate      | JTBD                 | 1.0.0           |    |
| fann_get_learning_momentum  | fannerl:get_params   | 1.0.0           |    |
| fann_set_learning_momentum  | JTBD                 | 1.0.0           |    |
| fann_get_activation_function      | JTBD - fannerl:get_params   | 1.0.0           |    |
| fann_set_activation_function      | JTBD                        | 1.0.0           |    |
| fann_set_activation_function_layer| JTBD                        | 1.0.0           |    |
| fann_set_activation_function_hidden| JTBD                        | 1.0.0           |    |
| fann_set_activation_function_output| JTBD                        | 1.0.0           |    |
| fann_get_activation_steepness      | JTBD - fannerl:get_params   | 1.0.0           |    |
| fann_set_activation_steepness      | JTBD                        | 1.0.0           |    |
| fann_set_activation_steepness_layer| JTBD                        | 1.0.0           |    |
| fann_set_activation_steepness_hidden| JTBD                        | 1.0.0           |    |
| fann_set_activation_steepness_output| JTBD                        | 1.0.0           |    |
| fann_get_train_error_function     | fannerl:get_params   | 1.0.0           |    |
| fann_set_train_error_function     | JTBD                        | 1.0.0           |    |
| fann_get_train_stop_function      | fannerl:get_params   | 1.0.0           |    |
| fann_set_train_stop_function      | JTBD                        | 1.0.0           |    |
| fann_get_bit_fail_limit           | fannerl:get_params   | 1.0.0           |    |
| fann_set_bit_fail_limit           | JTBD                        | 1.0.0           |    |
| fann_set_callback                 | -                           | 1.0.0           | Will not be supported   |
| fann_get_quickprop_decay          | fannerl:get_params   | 1.0.0           |    |
| fann_set_quickprop_decay          | JTBD                        | 1.0.0           |    |
| fann_get_quickprop_mu             | fannerl:get_params   | 1.0.0           |    |
| fann_set_quickprop_mu             | JTBD                        | 1.0.0           |    |
| fann_get_rprop_increase_factor    | fannerl:get_params   | 1.0.0           |    |
| fann_set_rprop_increase_factor    | JTBD                        | 1.0.0           |    |
| fann_get_rprop_decrease_factor    | fannerl:get_params   | 1.0.0           |    |
| fann_set_rprop_decrease_factor    | JTBD                        | 1.0.0           |    |
| fann_get_rprop_delta_min          | fannerl:get_params   | 1.0.0           |    |
| fann_set_rprop_delta_min          | JTBD                        | 1.0.0           |    |
| fann_get_rprop_delta_max          | fannerl:get_params   | 1.0.0           |    |
| fann_set_rprop_delta_max          | JTBD                        | 1.0.0           |    |
| fann_get_rprop_delta_zero         | fannerl:get_params   | 1.0.0           |    |
| fann_set_rprop_delta_zero         | JTBD                        | 1.0.0           |    |
| fann_get_sarprop_weight_decay_shift| fannerl:get_params   | 1.0.0           |    |
| fann_set_sarprop_weight_decay_shift| JTBD                        | 1.0.0           |    |
| fann_get_sarprop_step_error_threshold_factor | fannerl:get_params   | 1.0.0           |    |
| fann_set_sarprop_step_error_threshold_factor | JTBD                        | 1.0.0           |    |
| fann_get_sarprop_step_error_shift | fannerl:get_params   | 1.0.0           |    |
| fann_set_sarprop_step_error_shift | JTBD                        | 1.0.0           |    |
| fann_get_sarprop_temperature      | fannerl:get_params   | 1.0.0           |    |
| fann_set_sarprop_temperature      | JTBD                        | 1.0.0           |    |

## File Input/Output
| fann function                     | supported by                | fannerl verson  | Comment  |
|-----------------------------------|-----------------------------|-----------------|--------- | 
| fann_create_from_file             | fannerl:create_from_file    | 1.0.0           |    |
| fann_save                         | fannerl:save                | 1.0.0           |    |
| fann_save_to_fixed                | -                           | 1.X.0           | Will be supported in later release |

## Cascade Training
All cascade training functions  are planned for 1.3.0