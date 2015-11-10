# fannerl

This is an attempt to write erlang bindings to the [FANN](http://leenissen.dk/), Fast Artificial Neural Networks, library written in C. The interfacing towards FANN is done by a port driver. 

You can watch the documentation at <http://axling.github.io/fannerl/>.

# Plan of FANN support

## CREATING/EXECUTION


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
| fann_print_connections      | -                    | 1.X.0           | Will be covered in future release |
| fann_print_parameters       | -                    | 1.X.0           | Will be covered in future release |
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
| fann_set_weight_array       | JTBD                 | 1.0.0           |    |
| fann_set_weight             | JTBD                 | 1.0.0           |    |
| fann_get_weights            | -                    | -               | Will not cover    |
| fann_set_weights            | -                    | -               | Will not cover    |
| fann_set_user_data          | -                    | -               | Will not cover    |
| fann_get_user_data          | -                    | -               | Will not cover    |
| fann_disable_seed_rand      | -                    | Later release   | Supported in fann 2.3.0  |
| fann_enable_seed_rand       | -                    | Later release   | Supported in fann 2.3.0  |
| fann_get_decimal_point      | -                    | 1.X.0           | Will be covered in future release  |
| fann_get_multiplier         | -                    | 1.X.0           | Will be covered in future release  |

## TRAINING

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
| fann_get_train_input        | JTBD                 | 1.0.0           |          |
| fann_get_train_output       | JTBD                 | 1.0.0           |          |
| fann_shuffle_train_data     | fannerl:shuffle_train| 1.0.0           |          |
| fann_get_min_train_input    | JTBD                 | 1.0.0           |          |
| fann_get_max_train_input    | JTBD                 | 1.0.0           |          |
| fann_get_min_train_output   | JTBD                 | 1.0.0           |          |
| fann_get_max_train_output   | JTBD                 | 1.0.0           |          |
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
| fann_merge_train_data       | JTBD                 | 1.0.0           |    |
| fann_duplicate_train_data   | JTBD                 | 1.0.0           |    |
| fann_subset_train_data      | fannerl:subset_train_data | 1.0.0      |    |
| fann_length_train_data      | JTBD                 | 1.0.0           |    |
| fann_num_input_train_data   | JTBD                 | 1.0.0           |    |
| fann_num_output_train_data  | JTBD                 | 1.0.0           |    |
| fann_save_train             | JTBD                 | 1.0.0           |    |
| fann_save_train_to_fixed    | -                    | -               | Will be covered in future release |
