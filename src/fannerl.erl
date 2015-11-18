%%% @author Erik Axling <>
%%% @copyright (C) 2014, Erik Axling
%%% @doc
%%%
%%% @end
%%% Created : 24 Nov 2014 by Erik Axling <>

-module(fannerl).

-type network_layers() :: tuple().
-type network_ref()    :: reference().
-type train_ref()      :: reference().
-type options()        :: map().
-type connections()    :: map().
-type activation_function() :: fann_linear              |
			       fann_threshold           |
			       fann_threshold_symmetric |
			       fann_sigmoid             |
			       fann_sigmoid_stepwise    |
			       fann_sigmoid_symmetric   |
			       fann_gaussian            |
			       fann_gaussian_symmetric  |
			       fann_elliot              |
			       fann_elliot_symmetric    |
			       fann_linear_piece        |
			       fann_linear_piece_symmetric |
			       fann_sin_symmetric       |
			       fann_cos_symmetric       |
			       fann_sin                 |
			       fann_cos.


-export([start/0,
	 start_instance/0,
	 stop/0,
	 stop_instance/1,
	 init/1]).

-export([create/1,
	 create/2,
	 create_on/2,
	 create_on/3,
	 create_from_file/1,
	 create_from_file/2,
	 copy/1,
	 copy_on/2,
	 destroy/1,
	 destroy_on/2,
	 get_params/1,
	 get_params_on/2,
	 get_activation_function/3,
	 get_activation_function_on/4,
	 set_activation_function/4,
	 set_activation_function_on/5,
	 set_activation_function_layer/3,
	 set_activation_function_hidden/2,
	 set_activation_function_output/2,
	 set_activation_function_all/2,
	 get_activation_steepness/3,
	 get_activation_steepness_on/4,
	 set_activation_steepness/4,
	 set_activation_steepness_on/5,
	 set_activation_steepness_layer/3,
	 set_activation_steepness_hidden/2,
	 set_activation_steepness_output/2,
	 set_activation_steepness_all/2,
	 set_param/3,
	 set_param_on/4,
	 set_params/2,
	 set_params_on/3,
	 reset_mse/1,
	 reset_mse_on/2
	]).

-export([init_weights/2,
	 init_weights_on/3,
	 randomize_weights/3,
	 randomize_weights_on/4]).

-export([train_epoch/2,
	 train_epoch_on/3,
	 train/3,
	 train_on/4,
	 train_on_data/4,
	 train_on_data_on/5,
	 train_on_file/4,
	 train_on_file_on/5,
	 read_train_from_file/1,
	 read_train_from_file_on/2,
	 merge_train_data/2,
	 merge_train_data_on/3,
	 duplicate_train_data/1,
	 duplicate_train_data_on/2,
	 destroy_train/1,
	 destroy_train_on/2,
	 save_train/2,
	 save_train_on/3,
	 shuffle_train/1,
	 shuffle_train_on/2,
	 subset_train_data/3,
	 subset_train_data_on/4,
	 scale_train/2,
	 scale_train_on/3,
	 descale_train/2,
	 descale_train_on/3,
	 set_scaling_params/6,
	 set_scaling_params_on/7,
	 clear_scaling_params/1,
	 clear_scaling_params_on/2,
	 get_train_params/1,
	 get_train_params_on/2,
	 set_weights/2,
	 set_weights_on/3,
	 set_weight/4,
	 set_weight_on/5
	]).

-export([test/3,
	 test_on/4,
	 test_data/2,
	 test_data_on/3
	]).

-export([save/2,
	 save_on/3]).

-export([run/2,
	 run_on/3]).

%% --------------------------------------------------------------------- %%
%% @doc Start an instance of the port driver that can be used for
%%      creating artificial neural networks using fann. Will return a pid
%%      that could be used for monitoring. The pid will be registered with
%%      the name {@module}.
%% @end
%% --------------------------------------------------------------------- %%
-spec start() -> pid().
start() ->
    proc_lib:start_link(?MODULE, init, [module]).

%% --------------------------------------------------------------------- %%
%% @doc Start an instance of the port driver that can be used for
%%      creating artificial neural networks using fann. Will return a pid
%%      that is needed for all operations. The pid will not be registered
%%      in the name service.
%% @end
%% --------------------------------------------------------------------- %%
-spec start_instance() -> pid().
start_instance() ->
    proc_lib:start_link(?MODULE, init, [pid]).

%% --------------------------------------------------------------------- %%
%% @doc This will stop the process interfacing the FANN library. Any
%%      networks or training datas created that have not been saved
%%      will be lost.
%% @end
%% --------------------------------------------------------------------- %%
-spec stop() -> ok | {error, fannerl_not_started} |
		{error, {instance_exit_abnormal, {reason, term()}}}.
stop() ->
    case whereis(?MODULE) of
	undefined ->
	    {error, fannerl_not_started};
	Pid when is_pid(Pid) ->
	    MonRef = erlang:monitor(process, Pid),
	    Pid ! stop,
	    receive
		{'DOWN', MonRef, process, Pid, normal} ->
		    ok;
		{'DOWN', MonRef, process, Pid, Reason} ->
		    {error, {instance_exit_abnormal, {reason, Reason}}}
	    end
    end.

%% --------------------------------------------------------------------- %%
%% @doc This will stop this instance's process interfacing the FANN
%%      library . Any networks or training datas created that have not
%%      been saved will be lost.
%% @end
%% --------------------------------------------------------------------- %%
-spec stop_instance(Instance :: network_ref()) -> ok | {error, {instance_exit_abnormal, {reason, term()}}}.
stop_instance(Instance) when is_pid(Instance) ->
    MonRef = erlang:monitor(process, Instance),
    Instance ! stop,
    receive
	{'DOWN', MonRef, process, Instance, normal} ->
	    ok;
	{'DOWN', MonRef, process, Instance, Reason} ->
	    {error, {instance_exit_abnormal, {reason, Reason}}}
    end.
    
%% --------------------------------------------------------------------- %%
%% @equiv create_on({@module}, Layers, default_options())
%% @end
%% --------------------------------------------------------------------- %%
-spec create(Layers :: network_layers()) -> network_ref().
create(Layers) when is_tuple(Layers) ->
    create_on(?MODULE, Layers, default_options()).

%% --------------------------------------------------------------------- %%
%% @equiv create_on({@module}, Layers, Options)
%% @end
%% --------------------------------------------------------------------- %%
-spec create(Layers::tuple(), Options::options()) -> network_ref().
create(Layers, Options)
  when is_tuple(Layers),
       is_map(Options) ->
    create_on(?MODULE, Layers, Options).

%% --------------------------------------------------------------------- %%
%% @equiv create_on(Instance, Layers, default_options())
%% @end
%% --------------------------------------------------------------------- %%
-spec create_on(Instance::pid(), Layers::tuple()) -> network_ref().
create_on(Instance, Layers)
  when Instance == ?MODULE; is_pid(Instance),
       is_tuple(Layers) ->
    create_on(Instance, Layers, default_options()).

%% --------------------------------------------------------------------- %%
%% @doc Creates an artificial neural network with any number of layers. 
%% The Layers tuple size describe the number of layers while each position
%% sets the size of the layer. See the FANN documentation of create_standard:
%% [http://libfann.github.io/fann/docs/files/fann-h.html#fann_create_standard]
%% @end
%% --------------------------------------------------------------------- %%
-spec create_on(Instance::pid(), Layers::tuple(), Options::options()) ->
		       network_ref().
create_on(Instance, Layers, Options) 
  when Instance == ?MODULE; is_pid(Instance),
      is_tuple(Layers),
      is_map(Options) ->
    {Type, OptionsWithoutType} = 
	case maps:is_key(type, Options) of
	    true ->
		Type0 = maps:get(type, Options),
		Options0 = maps:remove(type, Options),
		{Type0, Options0};
	    false ->
		{standard, Options}
	end,
    {ConnRate, OptionsWithoutConnRateAndType} = 
	case Type of
	    sparse ->
		ConnRate0 = maps:get(conn_rate, OptionsWithoutType, 0.2),
		Options1 = maps:remove(conn_rate, OptionsWithoutType),
		{ConnRate0, Options1};
	    _Else ->
		{undefined, OptionsWithoutType}
	end,
    call_port(Instance, {create_standard,
			 {Layers, Type,
			  ConnRate, OptionsWithoutConnRateAndType}}).

%% --------------------------------------------------------------------- %%
%% @equiv create_from_file({@module}, FileName)
%% @end
%% --------------------------------------------------------------------- %%
-spec create_from_file(FileName :: string()) -> network_ref().
create_from_file(FileName) when is_list(FileName) ->
    create_from_file(?MODULE, FileName).

%% --------------------------------------------------------------------- %%
%% @doc Creates an artificial neural network from a previously saved file. 
%% See
%% [http://libfann.github.io/fann/docs/files/fann_io-h.html#fann_create_from_file].
%% @end
%% --------------------------------------------------------------------- %%
-spec create_from_file(Instance::pid(), FileName :: string()) -> network_ref().
create_from_file(Instance, FileName)
  when Instance == ?MODULE; is_pid(Instance),
       is_list(FileName) ->
    call_port(Instance, {create_from_file, FileName}).

%% --------------------------------------------------------------------- %%
%% @equiv copy_on({@module}, Network)
%% @end
%% --------------------------------------------------------------------- %%
-spec copy(Network::network_ref()) -> network_ref().
copy(Network) when is_reference(Network) ->
    copy_on(?MODULE, Network).

%% --------------------------------------------------------------------- %%
%% @doc Creates a copy of a fann structure. See
%% [http://libfann.github.io/fann/docs/files/fann-h.html#fann_copy].
%% @end
%% --------------------------------------------------------------------- %%
copy_on(Instance, Network)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Network) ->
    call_port(Instance, {copy, Network, {}}).
    

%% --------------------------------------------------------------------- %%
%% @equiv destroy_on({@module}, Network)
%% @end
%% --------------------------------------------------------------------- %%
-spec destroy(Network::network_ref) -> ok.
destroy(Network) when is_reference(Network) ->
    destroy_on(?MODULE, Network).

%% --------------------------------------------------------------------- %%
%% @doc This will destroy all references to the neural network and free
%%      up all associated memory.
%% See [http://libfann.github.io/fann/docs/files/fann-h.html#fann_destroy].
%% @end
%% --------------------------------------------------------------------- %%
-spec destroy_on(Instance::pid(), Network::network_ref) -> ok.
destroy_on(Instance, Network)
  when is_pid(Instance); Instance == ?MODULE,
       is_reference(Network) ->
    call_port(Instance, {destroy, Network, {}}).

%% --------------------------------------------------------------------- %%
%% @equiv train_epoch_on({@module}, Network, Train)
%% @end
%% --------------------------------------------------------------------- %%
-spec train_epoch(Network::network_ref(), Train::train_ref()) -> ok.
train_epoch(Network, Train)
  when is_reference(Network),
       is_reference(Train) ->
    train_epoch_on(?MODULE, Network, Train).

%% --------------------------------------------------------------------- %%
%% @doc This will train your neural network for one epoch with the given
%%      training data. See [http://libfann.github.io/fann/docs/files/fann_train-h.html#fann_train_epoch]
%% @end
%% --------------------------------------------------------------------- %%
-spec train_epoch_on(Instance::pid, Network::network_ref(),
		     Train::train_ref()) -> ok.
train_epoch_on(Instance, Network, Train)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Network),
       is_reference(Train) ->
    call_port(Instance, {train_epoch, {Network, Train}, {}}).

%% --------------------------------------------------------------------- %%
%% @equiv train_on({@module}, Network, Input, DesiredOutput)
%% @end
%% --------------------------------------------------------------------- %%
-spec train(Network::network_ref(), Input::tuple(), DesiredOutput::tuple()) ->
		   ok.
train(Network, Input, DesiredOutput) 
  when is_reference(Network),
       is_tuple(Input),
       is_tuple(DesiredOutput) ->
    train_on(?MODULE, Network, Input, DesiredOutput).

%% --------------------------------------------------------------------- %%
%% @doc Train one iteration with a set of inputs, and a set of desired outputs.
%% This training is always incremental training (see fann_train_enum),
%% since only one pattern is presented. See [http://libfann.github.io/fann/docs/files/fann_train-h.html#fann_train].
%% @end
%% --------------------------------------------------------------------- %%
-spec train_on(Instance::pid(),Network::network_ref(), Input::tuple(),
	       DesiredOutput::tuple()) -> ok.
train_on(Instance, Network, Input, DesiredOutput)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Network),
       is_tuple(Input),
       is_tuple(DesiredOutput) ->
    call_port(Instance, {train, Network, {Input, DesiredOutput}}).

%% --------------------------------------------------------------------- %%
%% @equiv train_on_data_on({@module}, Network, Train, MaxEpochs, DesiredError)
%% @end
%% --------------------------------------------------------------------- %%
-spec train_on_data(Network::network_ref(), Train::train_ref(),
		    MaxEpochs::non_neg_integer(), DesiredError::number()) ->
			   ok.
train_on_data(Network, Train, MaxEpochs, DesiredError)
  when is_reference(Network),
       is_reference(Train),
       is_integer(MaxEpochs),
       is_number(DesiredError) ->
    train_on_data_on(?MODULE, Network, Train, MaxEpochs, DesiredError).

%% --------------------------------------------------------------------- %%
%% @doc Trains on an entire dataset, for a chosen period of time. See [http://libfann.github.io/fann/docs/files/fann_train-h.html#fann_train_on_data].
%% @end
%% --------------------------------------------------------------------- %%
-spec train_on_data_on(Instance::pid(), Network::network_ref(),
		       Train:: train_ref(), MaxEpochs::non_neg_integer(),
		       DesiredError::number()) -> ok.
train_on_data_on(Instance, Network, Train, MaxEpochs, DesiredError)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Network),
       is_reference(Train),
       is_integer(MaxEpochs),
       is_number(DesiredError) ->
    call_port(?MODULE, {train_on_data, {Network, Train},
			{MaxEpochs, DesiredError}}).

%% --------------------------------------------------------------------- %%
%% @equiv train_on_file_on({@module}, Network, FileName, MaxEpochs,
%%  DesiredError)
%% @end
%% --------------------------------------------------------------------- %%
-spec train_on_file(Network::network_ref(), FileName::string(),
		    MaxEpochs::non_neg_integer(), DesiredError:: number()) ->
			   ok.
train_on_file(Network, FileName, MaxEpochs, DesiredError) 
  when is_reference(Network),
       is_list(FileName),
       is_integer(MaxEpochs), MaxEpochs > 0,
       is_float(DesiredError) ->
    train_on_file_on(?MODULE, Network, FileName, MaxEpochs, DesiredError).

%% --------------------------------------------------------------------- %%
%% @doc Does the same as train_on_data_on/5, but reads the training data directly from a file. See [http://libfann.github.io/fann/docs/files/fann_train-h.html#fann_train_on_file].
%% @end
%% --------------------------------------------------------------------- %%
-spec train_on_file_on(
	Instance::pid(), Network::network_ref(), FileName::string(),
	MaxEpochs::non_neg_integer(), DesiredError:: number()) ->
			      ok.
train_on_file_on(Instance, Network, FileName, MaxEpochs, DesiredError) 
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Network),
       is_list(FileName),
       is_integer(MaxEpochs), MaxEpochs > 0,
       is_float(DesiredError) ->
    call_port(Instance, 
	      {train_on_file, Network, {FileName, MaxEpochs, DesiredError}}).

%% --------------------------------------------------------------------- %%
%% @equiv test_data({@module}, Network, Train)
%% @end
%% --------------------------------------------------------------------- %%
-spec test_data(Network::network_ref(), Train::train_ref()) ->
		       {ok, MeanSquareError::number()}.
test_data(Network, Train)
  when is_reference(Network),
       is_reference(Train) ->
    test_data_on(?MODULE, Network, Train).

%% --------------------------------------------------------------------- %%
%% @doc Test a set of training data and calculates the MSE for the training data. See [http://libfann.github.io/fann/docs/files/fann_train-h.html#fann_test_data].
%% @end
%% --------------------------------------------------------------------- %%
-spec test_data_on(Instance::pid(), Network::network_ref(),
		   Train::train_ref()) ->
			  {ok, MeanSquareError::number()}.
test_data_on(Instance, Network, Train)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Network),
       is_reference(Train) ->
    call_port(Instance, {test_data, {Network, Train}, {}}).

%% --------------------------------------------------------------------- %%
%% @equiv test_on({@module}, Network, Input, DesiredOutput)
%% @end
%% --------------------------------------------------------------------- %%
-spec test(Network::network_ref(), Input::tuple(), DesiredOutput::tuple()) ->
		  Output::tuple().
test(Network, Input, DesiredOutput)
  when is_reference(Network),
       is_tuple(Input),
       is_tuple(DesiredOutput) ->
    test_on(?MODULE, Network, Input, DesiredOutput).

%% --------------------------------------------------------------------- %%
%% @doc Test with a set of inputs, and a set of desired outputs.  This operation updates the mean square error, but does not change the network in any way. See [http://libfann.github.io/fann/docs/files/fann_train-h.html#fann_test].
%% @end
%% --------------------------------------------------------------------- %%
-spec test_on(Instance::pid(), Network::network_ref(),
	      Input::tuple(), DesiredOutput::tuple()) ->
		  Output::tuple().
test_on(Instance, Network, Input, DesiredOutput)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Network),
       is_tuple(Input),
       is_tuple(DesiredOutput) ->
    call_port(Instance, {test, Network, {Input, DesiredOutput}}).

%% --------------------------------------------------------------------- %%
%% @equiv read_train_from_file_on({@module}, Filename)
%% @end
%% --------------------------------------------------------------------- %%
-spec read_train_from_file(Filename::string()) -> train_ref().
read_train_from_file(FileName)
  when is_list(FileName) ->
    read_train_from_file_on(?MODULE, FileName).

%% --------------------------------------------------------------------- %%
%% @doc Reads a file that stores training data. See [http://libfann.github.io/fann/docs/files/fann_train-h.html#fann_read_train_from_file].
%% @end
%% --------------------------------------------------------------------- %%
-spec read_train_from_file_on(Instance::pid(), Filename::string()) ->
				     train_ref().
read_train_from_file_on(Instance, FileName)
  when Instance == ?MODULE; is_pid(Instance),
       is_list(FileName) ->
    call_port(Instance, {read_train_from_file, FileName}).

%% --------------------------------------------------------------------- %%
%% @equiv merge_train_data_on({@module}, Train1, Train2)
%% @end
%% --------------------------------------------------------------------- %%
-spec merge_train_data(Train1::train_ref(), Train2::train_ref()) -> train_ref().
merge_train_data(Train1, Train2)
  when is_reference(Train1),
       is_reference(Train2) ->
    merge_train_data_on(?MODULE, Train1, Train2).

%% --------------------------------------------------------------------- %%
%% @doc Merges the data from data1 and data2 into a new training data.
%% See [http://libfann.github.io/fann/docs/files/fann_train-h.html#fann_merge_train_data].
%% @end
%% --------------------------------------------------------------------- %%
-spec merge_train_data_on(Instance::pid(), Train1::train_ref(),
			  Train2::train_ref()) -> train_ref().
merge_train_data_on(Instance, Train1, Train2)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Train1),
       is_reference(Train2) ->
    call_port(Instance, {merge_train, {trains, Train1, Train2}, {}}).

%% --------------------------------------------------------------------- %%
%% @equiv duplicate_train_data_on({@module}, Train)
%% @end
%% --------------------------------------------------------------------- %%
-spec duplicate_train_data(Train::train_ref()) -> train_ref().
duplicate_train_data(Train)
  when is_reference(Train) ->
    duplicate_train_data_on(?MODULE, Train).

%% --------------------------------------------------------------------- %%
%% @doc Returns an exact copy of the training data.
%% See [http://libfann.github.io/fann/docs/files/fann_train-h.html#fann_duplicate_train_data].
%% @end
%% --------------------------------------------------------------------- %%
-spec duplicate_train_data_on(Instance::pid(), Train::train_ref())
			     -> train_ref().
duplicate_train_data_on(Instance, Train)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Train) ->
    call_port(Instance, {duplicate_train, {train, Train}, {}}).

%% --------------------------------------------------------------------- %%
%% @equiv shuffle_train_on({@module}, Train)
%% @end
%% --------------------------------------------------------------------- %%
-spec shuffle_train(Train::train_ref()) -> ok.
shuffle_train(Train)
  when is_reference(Train) ->
    shuffle_train_on(?MODULE, Train).

%% --------------------------------------------------------------------- %%
%% @doc Shuffles training data, randomizing the order.  This is recommended for incremental training, while it has no influence during batch training. See [http://libfann.github.io/fann/docs/files/fann_train-h.html#fann_shuffle_train_data].
%% @end
%% --------------------------------------------------------------------- %%
-spec shuffle_train_on(Instance::pid(), Train::train_ref()) -> ok.
shuffle_train_on(Instance, Train)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Train) ->
    call_port(Instance, {shuffle_train, {train, Train}, {}}).

%% --------------------------------------------------------------------- %%
%% @equiv run_on({@module}, Network, Input)
%% @end
%% --------------------------------------------------------------------- %%
-spec run(Network::network_ref(), Input::tuple()) -> tuple().
run(Network, Input) 
  when is_reference(Network),
       is_tuple(Input) ->
    run_on(?MODULE, Network, Input).

%% --------------------------------------------------------------------- %%
%% @doc Will run input through the neural network, returning a tuple of outputs, the number of which being equal to the number of neurons in the output layer. See [http://libfann.github.io/fann/docs/files/fann-h.html#fann_run].
%% @end
%% --------------------------------------------------------------------- %%
-spec run_on(Instance::pid(), Network::network_ref(), Input::tuple()) ->
		    tuple().
run_on(Instance, Network, Input) 
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Network),
       is_tuple(Input) ->
    call_port(Instance, {run, Network, {Input}}).

%% --------------------------------------------------------------------- %%
%% @equiv save_on({@module}, FileName)
%% @end
%% --------------------------------------------------------------------- %%
-spec save(Network::network_ref(), FileName::string()) -> ok.
save(Network, FileName)
  when is_reference(Network),
       is_list(FileName) ->
    save_on(?MODULE, Network, FileName).

%% --------------------------------------------------------------------- %%
%% @doc Save the entire network to a configuration file.
%% See [http://libfann.github.io/fann/docs/files/fann_io-h.html#fann_save].
%% @end
%% --------------------------------------------------------------------- %%
-spec save_on(Instance::pid(), Network::network_ref(), FileName::string()) ->
		     ok.
save_on(Instance, Network, FileName)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Network),
       is_list(FileName) ->
    call_port(Instance, {save_to_file, Network, {FileName}}).

%% --------------------------------------------------------------------- %%
%% @equiv subset_train_data_on({@module}, Train, Pos, Length)
%% @end
%% --------------------------------------------------------------------- %%
-spec subset_train_data(Train::train_ref(), Pos::non_neg_integer(),
		       Length::non_neg_integer()) -> train_ref().
subset_train_data(Train, Pos, Length) ->
    subset_train_data_on(?MODULE, Train, Pos, Length).

%% --------------------------------------------------------------------- %%
%% @doc Returns an copy of a subset of the struct fann_train_data, starting
%% at position pos and length elements forward.
%% See [http://libfann.github.io/fann/docs/files/fann_train-h.html#fann_subset_train_data].
%% @end
%% --------------------------------------------------------------------- %%
-spec subset_train_data_on(Instance::pid(), Train::train_ref(),
			   Pos::non_neg_integer(),
			   Length::non_neg_integer()) -> 
				  train_ref().
subset_train_data_on(Instance, Train, Pos, Length) 
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Train),
       is_integer(Pos), Pos >= 0,
       is_integer(Length), Length >= 0 ->
    call_port(Instance, {subset_train_data, {train, Train}, {Pos, Length}}).

%% --------------------------------------------------------------------- %%
%% @equiv get_params_on({@module}, Network)
%% @end
%% --------------------------------------------------------------------- %%
-spec get_params(Network::network_ref()) -> map().
get_params(Network) 
  when is_reference(Network) ->
    get_params_on(?MODULE, Network).

%% --------------------------------------------------------------------- %%
%% @doc Fetch all the parameters associated with the neural network.
%% @end
%% --------------------------------------------------------------------- %%
-spec get_params_on(Instance::pid(), Network::network_ref()) -> map().
get_params_on(Instance, Network)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Network) ->
    call_port(Instance, {get_params, Network, {}}).

%% --------------------------------------------------------------------- %%
%% @equiv get_activation_function({@module}, Network, Layer, Neuron)
%% @end
%% --------------------------------------------------------------------- %%
-spec get_activation_function(Network::network_ref(),
			      Layer::pos_integer(),
			      Neuron::non_neg_integer())
			     ->  atom().
get_activation_function(Network, Layer, Neuron) 
  when is_reference(Network), is_integer(Layer), Layer > 0,
       is_integer(Neuron), Neuron >= 0 ->
    get_activation_function_on(?MODULE, Network, Layer, Neuron).

%% --------------------------------------------------------------------- %%
%% @doc Get the activation function for neuron number neuron in layer number layer, counting the input layer as layer 0.
%% It is not possible to get activation functions for the neurons in the input layer.
%% See [http://libfann.github.io/fann/docs/files/fann_train-h.html#fann_get_activation_function]
%% @end
%% --------------------------------------------------------------------- %%
-spec get_activation_function_on(Instance::pid(), Network::network_ref(),
				 Layer::pos_integer(),
				 Neuron::non_neg_integer())
				-> atom().
get_activation_function_on(Instance, Network, Layer, Neuron)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Network), is_integer(Layer), Layer > 0,
       is_integer(Neuron), Neuron >= 0 ->
    call_port(Instance, {get_activation_function,
			 Network, {Layer, Neuron}}).

%% --------------------------------------------------------------------- %%
%% @equiv set_activation_function_on({@module}, Network, ActivationFunction,
%%                                hidden, all)
%% @end
%% --------------------------------------------------------------------- %%
-spec set_activation_function_hidden(
	Network::network_ref(),
	ActivationFunction::activation_function()) ->
					    ok.
set_activation_function_hidden(Network, ActivationFunction) 
  when is_reference(Network),
       is_atom(ActivationFunction) ->
    set_activation_function_on(?MODULE, Network, ActivationFunction,
			       hidden, all).

%% --------------------------------------------------------------------- %%
%% @equiv set_activation_function_on({@module}, Network, ActivationFunction,
%%                                    output, all)
%% @end
%% --------------------------------------------------------------------- %%
-spec set_activation_function_output(
	Network::network_ref(),
	ActivationFunction::activation_function()) ->
					    ok.
set_activation_function_output(Network, ActivationFunction) 
  when is_reference(Network),
       is_atom(ActivationFunction) ->
    set_activation_function_on(?MODULE, Network, ActivationFunction,
			       output, all).

%% --------------------------------------------------------------------- %%
%% @equiv set_activation_function_layer({@module}, Network, ActivationFunction,
%%                                    Layer, all)
%% @end
%% --------------------------------------------------------------------- %%
-spec set_activation_function_layer(
	Network::network_ref(),
	ActivationFunction::activation_function(),
	Layer::pos_integer()) ->
					    ok.
set_activation_function_layer(Network, ActivationFunction, Layer) 
  when is_reference(Network),
       is_atom(ActivationFunction),
       is_integer(Layer), Layer > 0 ->
    set_activation_function_on(?MODULE, Network, ActivationFunction,
			       Layer, all).

%% --------------------------------------------------------------------- %%
%% @equiv set_activation_function_all({@module}, Network, ActivationFunction,
%%                                    all, all)
%% @end
%% --------------------------------------------------------------------- %%
-spec set_activation_function_all(
	Network::network_ref(),
	ActivationFunction::activation_function()) ->
					 ok.
set_activation_function_all(Network, ActivationFunction) 
  when is_reference(Network),
       is_atom(ActivationFunction) ->
    set_activation_function_on(?MODULE, Network, ActivationFunction, 
			       all, all).

%% --------------------------------------------------------------------- %%
%% @equiv set_activation_function_on({@module}, Network, ActivationFunction,
%%                                Layer, Neuron)
%% @end
%% --------------------------------------------------------------------- %%
-spec set_activation_function(Network::network_ref(),
			      ActivationFunction::activation_function(),
			      Layer::pos_integer() | hidden | output | all ,
			      Neuron::non_neg_integer() | all) ->
				     ok.
set_activation_function(Network, ActivationFunction, Layer, Neuron) 
  when is_reference(Network),
       is_atom(ActivationFunction),
       ((is_integer(Layer) and (Layer > 0)) or
	(Layer==hidden) or (Layer==output) or (Layer==all)),
       ((is_integer(Neuron) and (Neuron >= 0)) or (Neuron==all)) ->
    set_activation_function_on(?MODULE, Network, ActivationFunction,
			       Layer, Neuron).

%% --------------------------------------------------------------------- %%
%% @doc Set the activation function for neuron number neuron in layer number
%% layer, counting the input layer as layer 0.
%% It is not possible to set activation functions for the neurons in the input layer.
%% If you specify Layer as hidden, output the activation function will be set
%% for all neurons. If all is specified then the activation function will be 
%% set for all possible layers.
%% See [http://libfann.github.io/fann/docs/files/fann_train-h.html#fann_get_activation_function]
%% @end
%% --------------------------------------------------------------------- %%
-spec set_activation_function_on(Instance::pid(),
				 Network::network_ref(),
				 ActivationFunction::activation_function(),
				 Layer::pos_integer() | hidden | output | all ,
				 Neuron::non_neg_integer() | all) ->
					ok.
set_activation_function_on(Instance, Network, ActivationFunction,
			   Layer, Neuron) 
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Network),
       is_atom(ActivationFunction),
       ((is_integer(Layer) and (Layer > 0)) or
	(Layer==hidden) or (Layer==output) or (Layer==all)),
       ((is_integer(Neuron) and (Neuron >= 0)) or (Neuron==all)) ->
    call_port(Instance, {set_activation_function, Network,
			 {ActivationFunction, Layer, Neuron}}).

%% --------------------------------------------------------------------- %%
%% @equiv get_activation_steepness({@module}, Network, Layer, Neuron)
%% @end
%% --------------------------------------------------------------------- %%
-spec get_activation_steepness(Network::network_ref(),
			       Layer::pos_integer(),
			       Neuron::non_neg_integer())
			      ->  float().
get_activation_steepness(Network, Layer, Neuron) 
  when is_reference(Network), is_integer(Layer), Layer > 0,
       is_integer(Neuron), Neuron >= 0 ->
    get_activation_steepness_on(?MODULE, Network, Layer, Neuron).

%% --------------------------------------------------------------------- %%
%% @doc Get the activation steepness for neuron number neuron in layer number
%% layer, counting the input layer as layer 0. It is not possible to get
%% activation steepness for the neurons in the input layer. The steepness
%% of an activation function says something about how fast the activation
%% function goes from the minimum to the maximum.  A high value for the
%% activation function will also give a more aggressive training.
%%
%% When training neural networks where the output values should be at the
%% extremes (usually 0 and 1, depending on the activation function),
%%  a steep activation function can be used (e.g.  1.0).
%% The default activation steepness is 0.5.
%% See [http://libfann.github.io/fann/docs/files/fann_train-h.html#fann_get_activation_steepness]
%% @end
%% --------------------------------------------------------------------- %%
-spec get_activation_steepness_on(Instance::pid(), Network::network_ref(),
				  Layer::pos_integer(),
				  Neuron::non_neg_integer())
				 -> float().
get_activation_steepness_on(Instance, Network, Layer, Neuron)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Network), is_integer(Layer), Layer > 0,
       is_integer(Neuron), Neuron >= 0 ->
    call_port(Instance, {get_activation_steepness,
			 Network, {Layer, Neuron}}).

%% --------------------------------------------------------------------- %%
%% @equiv set_activation_steepness_on({@module}, Network, ActivationSteepness,
%%                                hidden, all)
%% @end
%% --------------------------------------------------------------------- %%
-spec set_activation_steepness_hidden(
	Network::network_ref(),
	ActivationSteepness::number()) -> ok.
set_activation_steepness_hidden(Network, ActivationSteepness) 
  when is_reference(Network),
       is_number(ActivationSteepness) ->
    set_activation_steepness_on(?MODULE, Network, ActivationSteepness,
				hidden, all).

%% --------------------------------------------------------------------- %%
%% @equiv set_activation_steepness_on({@module}, Network, ActivationSteepness,
%%                                    output, all)
%% @end
%% --------------------------------------------------------------------- %%
-spec set_activation_steepness_output(
	Network::network_ref(),
	ActivationSteepness::number()) -> ok.
set_activation_steepness_output(Network, ActivationSteepness) 
  when is_reference(Network),
       is_number(ActivationSteepness) ->
    set_activation_steepness_on(?MODULE, Network, ActivationSteepness,
				output, all).

%% --------------------------------------------------------------------- %%
%% @equiv set_activation_steepness_layer({@module}, Network,
%%                                       ActivationSteepness,
%%                                       Layer, all)
%% @end
%% --------------------------------------------------------------------- %%
-spec set_activation_steepness_layer(
	Network::network_ref(),
	ActivationSteepness::number(),
	Layer::pos_integer()) -> ok.
set_activation_steepness_layer(Network, ActivationSteepness, Layer) 
  when is_reference(Network),
       is_number(ActivationSteepness),
       is_integer(Layer), Layer > 0 ->
    set_activation_steepness_on(?MODULE, Network, ActivationSteepness,
				Layer, all).

%% --------------------------------------------------------------------- %%
%% @equiv set_activation_steepness_all({@module}, Network, ActivationSteepness,
%%                                    all, all)
%% @end
%% --------------------------------------------------------------------- %%
-spec set_activation_steepness_all(
	Network::network_ref(),
	ActivationSteepness::number()) -> ok.
set_activation_steepness_all(Network, ActivationSteepness) 
  when is_reference(Network),
       is_number(ActivationSteepness) ->
    set_activation_steepness_on(?MODULE, Network, ActivationSteepness, 
				all, all).

%% --------------------------------------------------------------------- %%
%% @equiv set_activation_steepness_on({@module}, Network, ActivationSteepness,
%%                                    Layer, Neuron)
%% @end
%% --------------------------------------------------------------------- %%
-spec set_activation_steepness(Network::network_ref(),
			       ActivationSteepness::number(),
			       Layer::pos_integer() | hidden | output | all ,
			       Neuron::non_neg_integer() | all) ->
				      ok.
set_activation_steepness(Network, ActivationSteepness, Layer, Neuron) 
  when is_reference(Network),
       is_number(ActivationSteepness),
       ((is_integer(Layer) and (Layer > 0)) or
	(Layer==hidden) or (Layer==output) or (Layer==all)),
       ((is_integer(Neuron) and (Neuron >= 0)) or (Neuron==all)) ->
    set_activation_steepness_on(?MODULE, Network, ActivationSteepness,
				Layer, Neuron).

%% --------------------------------------------------------------------- %%
%% @doc Set the activation steepness for neuron number neuron in layer
%% number layer, counting the input layer as layer 0. It is not possible
%% to set activation steepness for the neurons in the input layer.
%% The steepness of an activation function says something about how fast
%% the activation function goes from the minimum to the maximum.  A high
%% value for the activation function will also give a more aggressive training.
%%
%% When training neural networks where the output values should be at the
%% extremes (usually 0 and 1, depending on the activation function), a steep
%% activation function can be used (e.g.  1.0).
%% The default activation steepness is 0.5.
%%
%% If you specify Layer as hidden, output the activation function will be set
%% for all neurons. If all is specified then the activation function will be 
%% set for all possible layers.
%% See [http://libfann.github.io/fann/docs/files/fann_train-h.html#fann_set_activation_steepness]
%% @end
%% --------------------------------------------------------------------- %%
-spec set_activation_steepness_on(Instance::pid(),
				  Network::network_ref(),
				  ActivationSteepness::number(),
				  Layer::pos_integer() | hidden | output | all ,
				  Neuron::non_neg_integer() | all) ->
					ok.
set_activation_steepness_on(Instance, Network, ActivationSteepness,
			    Layer, Neuron) 
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Network),
       is_number(ActivationSteepness),
       ((is_integer(Layer) and (Layer > 0)) or
	(Layer==hidden) or (Layer==output) or (Layer==all)),
       ((is_integer(Neuron) and (Neuron >= 0)) or (Neuron==all)) ->
    call_port(Instance, {set_activation_steepness, Network,
			 {ActivationSteepness, Layer, Neuron}}).


%% --------------------------------------------------------------------- %%
%% @equiv randomize_weights_on({@module}, Network, MinWeight, MaxWeight)
%% @end
%% --------------------------------------------------------------------- %%
-spec randomize_weights(Network::network_ref(),
			MinWeight::number(), MaxWeight::number()) -> ok.
randomize_weights(Network, MinWeight, MaxWeight)
  when is_reference(Network), is_number(MinWeight), is_number(MaxWeight),
       MinWeight =< MaxWeight ->
    randomize_weights_on(?MODULE, Network, MinWeight, MaxWeight).

%% --------------------------------------------------------------------- %%
%% @doc Give each connection a random weight between min_weight and max_weight.
%% From the beginning the weights are random between -0.1 and 0.1.
%% See [http://libfann.github.io/fann/docs/files/fann-h.html#fann_randomize_weights].
%% @end
%% --------------------------------------------------------------------- %%
-spec randomize_weights_on(Instance::pid(), Network::network_ref(),
			   MinWeight::number(), MaxWeight::number()) -> ok.
randomize_weights_on(Instance, Network, MinWeight, MaxWeight) 
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Network), is_number(MinWeight), is_number(MaxWeight),
       MinWeight =< MaxWeight ->
    call_port(Instance, {randomize_weights, Network, {MinWeight, MaxWeight}}).

%% --------------------------------------------------------------------- %%
%% @equiv init_weights_on({@module}, Network, Train)
%% @end
%% --------------------------------------------------------------------- %%
-spec init_weights(Network::network_ref(), Train::train_ref()) -> ok.
init_weights(Network, Train)
  when is_reference(Network), is_reference(Train) ->
    init_weights_on(?MODULE, Network, Train).

%% --------------------------------------------------------------------- %%
%% @doc Initialize the weights using Widrow + Nguyen’s algorithm.
%%
%% This function behaves similarly to fann_randomize_weights.  It will use
%% the algorithm developed by Derrick Nguyen and Bernard Widrow to set
%% the weights in such a way as to speed up training. 
%% See [http://libfann.github.io/fann/docs/files/fann-h.html#fann_init_weights].
%% @end
%% --------------------------------------------------------------------- %%
-spec init_weights_on(Instance::pid(), Network::network_ref(),
		      Train::train_ref()) -> ok.
init_weights_on(Instance, Network, Train)
  when Instance == ?MODULE; is_pid(Instance), 
       is_reference(Network), is_reference(Train) ->
    call_port(Instance, {init_weights, {Network, Train}, {}}).

%% --------------------------------------------------------------------- %%
%% @equiv reset_mse_on({@module}, Network)
%% @end
%% --------------------------------------------------------------------- %%
-spec reset_mse(Network::network_ref()) -> ok.
reset_mse(Network)
  when is_reference(Network) ->
    reset_mse_on(?MODULE, Network).

%% --------------------------------------------------------------------- %%
%% @doc Resets the mean square error from the network.
%% This function also resets the number of bits that fail. 
%% See http://libfann.github.io/fann/docs/files/fann_train-h.html#fann_reset_MSE
%% @end
%% --------------------------------------------------------------------- %%
-spec reset_mse_on(Instance::pid(), Network::network_ref()) -> ok.
reset_mse_on(Instance, Network)
  when Instance == ?MODULE; is_pid(Instance), 
       is_reference(Network) ->
    call_port(Instance, {reset_mse, Network, {}}).

%% --------------------------------------------------------------------- %%
%% @equiv destroy_train_on({@module}, Train)
%% @end
%% --------------------------------------------------------------------- %%
-spec destroy_train(Train::train_ref()) -> ok.
destroy_train(Train)
  when is_reference(Train) ->
    destroy_train_on(?MODULE, Train).

%% --------------------------------------------------------------------- %%
%% @doc Destructs the training data and properly deallocates all of the
%% associated data.
%% See http://libfann.github.io/fann/docs/files/fann_train-h.html#fann_destroy_train
%% @end
%% --------------------------------------------------------------------- %%
-spec destroy_train_on(Instance::pid(), Train::train_ref()) -> ok.
destroy_train_on(Instance, Train)
  when Instance == ?MODULE; is_pid(Instance), 
       is_reference(Train) ->
    call_port(Instance, {destroy_train, {train, Train}, {}}).

%% --------------------------------------------------------------------- %%
%% @equiv save_train_on({@module}, Train, FileName)
%% @end
%% --------------------------------------------------------------------- %%
-spec save_train(Train::train_ref(), FileName::string()) -> ok.
save_train(Train, FileName)
  when is_reference(Train), is_list(FileName) ->
    save_train_on(?MODULE, Train, FileName).

%% --------------------------------------------------------------------- %%
%% @doc Save the training structure to a file, with the format as specified in fann_read_train_from_file
%% See http://libfann.github.io/fann/docs/files/fann_train-h.html#fann_save_train
%% @end
%% --------------------------------------------------------------------- %%
-spec save_train_on(Instance::pid(), Train::train_ref(), FileName::string())
		   -> ok.
save_train_on(Instance, Train, FileName)
  when Instance == ?MODULE; is_pid(Instance), 
       is_reference(Train), is_list(FileName) ->
    call_port(Instance, {save_train, {train, Train}, {FileName}}).

%% --------------------------------------------------------------------- %%
%% @equiv scale_train_on({@module}, Network, Train)
%% @end
%% --------------------------------------------------------------------- %%
-spec scale_train(Network::network_ref(), Train::train_ref()) -> ok.
scale_train(Network, Train)
  when is_reference(Network), is_reference(Train) ->
    scale_train_on(?MODULE, Network, Train).

%% --------------------------------------------------------------------- %%
%% @doc Scale input and output data based on previously calculated parameters. 
%% See [http://libfann.github.io/fann/docs/files/fann_train-h.html#fann_scale_train].
%% @end
%% --------------------------------------------------------------------- %%
-spec scale_train_on(Instance::pid(), Network::network_ref(),
		     Train::train_ref()) -> ok.
scale_train_on(Instance, Network, Train)
  when Instance == ?MODULE; is_pid(Instance), 
       is_reference(Network), is_reference(Train) ->
    call_port(Instance, {scale_train, {Network, Train}, {}}).

%% --------------------------------------------------------------------- %%
%% @equiv descale_train_on({@module}, Network, Train)
%% @end
%% --------------------------------------------------------------------- %%
-spec descale_train(Network::network_ref(), Train::train_ref()) -> ok.
descale_train(Network, Train)
  when is_reference(Network), is_reference(Train) ->
    descale_train_on(?MODULE, Network, Train).

%% --------------------------------------------------------------------- %%
%% @doc Descale input and output data based on previously calculated parameters. 
%% See [http://libfann.github.io/fann/docs/files/fann_train-h.html#fann_descale_train].
%% @end
%% --------------------------------------------------------------------- %%
-spec descale_train_on(Instance::pid(), Network::network_ref(),
		       Train::train_ref()) -> ok.
descale_train_on(Instance, Network, Train)
  when Instance == ?MODULE; is_pid(Instance), 
       is_reference(Network), is_reference(Train) ->
    call_port(Instance, {descale_train, {Network, Train}, {}}).

%% --------------------------------------------------------------------- %%
%% @equiv set_scaling_params_on({@module}, Network, Train,
%%                           NewInputMin, NewInputMax,
%%                           NewOutputMin, NewOutputMax)
%% @end
%% --------------------------------------------------------------------- %%
-spec set_scaling_params(Network::network_ref(), Train::train_ref(),
			 NewInputMin::number(), NewInputMax::number(),
			 NewOutputMin::number(), NewOutputMax::number()) -> ok.
set_scaling_params(Network, Train, NewInputMin, NewInputMax,
		   NewOutputMin, NewOutputMax)
  when is_reference(Network), is_reference(Train),
       is_number(NewInputMin), is_number(NewInputMax),
       is_number(NewOutputMin), is_number(NewOutputMax)->
    set_scaling_params_on(?MODULE, Network, Train, NewInputMin, NewInputMax,
			  NewOutputMin, NewOutputMax).

%% --------------------------------------------------------------------- %%
%% @doc Calculate input and output scaling parameters for future use based on training data.
%% See [http://libfann.github.io/fann/docs/files/fann_train-h.html#fann_set_scaling_params].
%% @end
%% --------------------------------------------------------------------- %%
-spec set_scaling_params_on(
	Instance::pid(), Network::network_ref(), Train::train_ref(),
	NewInputMin::number(), NewInputMax::number(),
	NewOutputMin::number(), NewOutputMax::number()) -> ok.
set_scaling_params_on(Instance, Network, Train, NewInputMin, NewInputMax,
		      NewOutputMin, NewOutputMax)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Network), is_reference(Train),
       is_number(NewInputMin), is_number(NewInputMax),
       is_number(NewOutputMin), is_number(NewOutputMax)->
    call_port(Instance, {set_scaling_params, {Network, Train}, 
			 {NewInputMin, NewInputMax,
			  NewOutputMin, NewOutputMax}}).

%% --------------------------------------------------------------------- %%
%% @equiv clear_scaling_params({@module}, Network)
%% @end
%% --------------------------------------------------------------------- %%
-spec clear_scaling_params(Network::network_ref()) -> ok.
clear_scaling_params(Network)
  when is_reference(Network) ->
    clear_scaling_params_on(?MODULE, Network).

%% --------------------------------------------------------------------- %%
%% @doc Clears scaling parameters.
%% See [http://libfann.github.io/fann/docs/files/fann_train-h.html#fann_clear_scaling_params].
%% @end
%% --------------------------------------------------------------------- %%
-spec clear_scaling_params_on(
	Instance::pid(), Network::network_ref()) -> ok.
clear_scaling_params_on(Instance, Network)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Network) ->
    call_port(Instance, {clear_scaling_params, Network, {}}).

%% --------------------------------------------------------------------- %%
%% @equiv get_train_params_on({@module}, Train)
%% @end
%% --------------------------------------------------------------------- %%
-spec get_train_params(Train::train_ref()) -> ok.
get_train_params(Train)
  when is_reference(Train) ->
    get_train_params_on(?MODULE, Train).

%% --------------------------------------------------------------------- %%
%% @doc Fetches all the params associated with the training data.
%% @end
%% --------------------------------------------------------------------- %%
-spec get_train_params_on(
	Instance::pid(), Train::train_ref()) -> map().
get_train_params_on(Instance, Train)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Train) ->
    call_port(Instance, {get_train_params, {train, Train}, {}}).

%% --------------------------------------------------------------------- %%
%% @equiv set_weights_on({@module}, Network, Connections)
%% @end
%% --------------------------------------------------------------------- %%
-spec set_weights(Network::network_ref(), Connections::connections()) -> ok.
set_weights(Network, Connections)
  when is_reference(Network),
       is_map(Connections) ->
    set_weights_on(?MODULE, Network, Connections).

%% --------------------------------------------------------------------- %%
%% @doc Set connections in the network.
%% See [http://libfann.github.io/fann/docs/files/fann-h.html#fann_set_weight_array].
%% @end
%% --------------------------------------------------------------------- %%
-spec set_weights_on(Instance::pid(),
		     Network::network_ref(), Connections::connections()) -> ok.
set_weights_on(Instance, Network, Connections)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Network),
       is_map(Connections) ->
    call_port(Instance, {set_weights, Network, {Connections}}).

%% --------------------------------------------------------------------- %%
%% @equiv set_weight_on({@module}, Network, FromNeuron, ToNeuron, Weight)
%% @end
%% --------------------------------------------------------------------- %%
-spec set_weight(Network::network_ref(), FromNeuron::non_neg_integer(),
		 ToNeuron::non_neg_integer(), Weight::number()) -> ok.
set_weight(Network, FromNeuron, ToNeuron, Weight)
  when is_reference(Network),
       is_integer(FromNeuron), FromNeuron >= 0,
       is_integer(ToNeuron), ToNeuron >= 0,
       is_number(Weight) ->
    set_weight_on(?MODULE, Network, FromNeuron, ToNeuron, Weight).

%% --------------------------------------------------------------------- %%
%% @doc Set a connection in the network.
%% See [http://libfann.github.io/fann/docs/files/fann-h.html#fann_set_weight].
%% @end
%% --------------------------------------------------------------------- %%
-spec set_weight_on(Instance::pid(),
		    Network::network_ref(), FromNeuron::non_neg_integer(),
		    ToNeuron::non_neg_integer(), Weight::number()) -> ok.
set_weight_on(Instance, Network, FromNeuron, ToNeuron, Weight)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Network),
       is_integer(FromNeuron), FromNeuron >= 0,
       is_integer(ToNeuron), ToNeuron >= 0,
       is_number(Weight) ->
    call_port(Instance, {set_weight, Network, {FromNeuron, ToNeuron, Weight}}).

%% --------------------------------------------------------------------- %%
%% @equiv set_param_on({@module}, Network, Param, Value)
%% @end
%% --------------------------------------------------------------------- %%
-spec set_param(Network::network_ref(), Param::atom(), Value::term()) -> ok.
set_param(Network, Param, Value)
  when is_reference(Network),
       is_atom(Param) ->
    set_param_on(?MODULE, Network, Param, Value).

%% --------------------------------------------------------------------- %%
%% @doc Sets one param on a given network. Equivalent to set_params_on/3
%% with a map with one key and value.
%% @end
%% --------------------------------------------------------------------- %%
-spec set_param_on(Instance::pid(), Network::network_ref(),
		   Param::atom(), Value::term()) -> ok.
set_param_on(Instance, Network, Param, Value)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Network),
       is_atom(Param) ->
    Map = maps:put(Param, Value, maps:new()),
    set_params_on(Instance, Network, Map).

%% --------------------------------------------------------------------- %%
%% @equiv set_params_on({@module}, Network, ParametersMap)
%% @end
%% --------------------------------------------------------------------- %%
-spec set_params(Network::network_ref(), ParametersMap::map()) -> ok.
set_params(Network, ParametersMap)
  when is_reference(Network),
       is_map(ParametersMap) ->
    set_params_on(?MODULE, Network, ParametersMap).

%% --------------------------------------------------------------------- %%
%% @doc Sets parameters on a given network. 
%% @end
%% --------------------------------------------------------------------- %%
-spec set_params_on(Instance::pid(), Network::network_ref(),
		   ParamertersMap::map()) -> ok.
set_params_on(Instance, Network, ParametersMap)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Network),
       is_map(ParametersMap) ->
    call_port(Instance, {set_params, Network, {ParametersMap}}).

%%****************************************************************%%       
%% Private functions
%%****************************************************************%%       
%% @private
call_port(?MODULE, Msg) ->
    case whereis(?MODULE) of
	Pid when is_pid(Pid) ->
	    call(?MODULE, Msg);
	_ ->
	    erlang:error(not_valid_instance)
    end;
call_port(Instance, Msg) when is_pid(Instance) ->
    case is_process_alive(Instance) of
	true ->
	    call(Instance, Msg);
	false ->
	    erlang:error(process_not_alive)
    end.

%% @private
call(Instance, Msg) ->
    Instance ! {call, self(), Msg},
    receive
	{fannerl_res, {error, Reason}} ->
	    erlang:error(Reason);
	{fannerl_res, Result} ->
	    Result;
	{fannerl_exception, Reason} ->
	    erlang:error(Reason)
    end.

%% @private
init(Type) ->
    PrivDir = code:priv_dir(fannerl),
    Program = filename:join(PrivDir, "fannerl"),
    Success =
	case Type of
	    module ->
		case whereis(?MODULE) of
		    Pid when is_pid(Pid) ->
			false;
		    _ ->
			true = register(?MODULE, self())
		end;
	    _ ->
		true
	end,
    case Success of
	false ->
	    exit(already_registered);
	true ->
	    process_flag(trap_exit, true),
	    Port = open_port({spawn, Program},
			     [{packet, 2}, nouse_stdio,  binary, exit_status]),
	    proc_lib:init_ack(self()),
	    loop(Port, #{networks => dict:new(),
			 trains   => dict:new()})
    end.

%% @private
%% TODO: remove all networks and trains if process is to be shut down.
loop(Port, State) ->
    receive
	{call, Caller, Msg} ->
	    handle_port_call(Port, State, Caller, Msg);
	stop ->
	    destroy_all(Port, State),
	    erlang:port_close(Port),
	    exit(normal)
    end.

%% @private
valid_network(Ref, State) ->
    case dict:is_key(Ref, maps:get(networks, State)) of
	true ->
	    {true, dict:fetch(Ref, maps:get(networks, State))};
	false ->
	    false
    end.

%% @private
valid_train(Ref, State) ->
    case dict:is_key(Ref, maps:get(trains, State)) of
	true ->
	    {true, dict:fetch(Ref, maps:get(trains, State))};
	false ->
	    false
    end.

%% @private
destroy_all(Port, State) ->
    Networks = maps:get(networks, State),
    Trains = maps:get(trains, State),
    ok = destroy_networks(Port, dict:to_list(Networks)),
    ok = destroy_trains(Port, dict:to_list(Trains)).

%% @private
destroy_networks(_Port, []) ->
    ok;
destroy_networks(Port, [{_Key, Network} | Networks]) ->
    Msg = {destroy, Network, {}},
    case do_port_call(Port, Msg) of
	ok ->
	    destroy_networks(Port, Networks);
	{error, Reason} ->
	    io:format("FANNERL: exit when destroying network with reason ~p~n",
		      [Reason]),
	    destroy_networks(Port, Networks)
    end.

%% @private
destroy_trains(_Port, []) ->
    ok;
destroy_trains(Port, [{_Key, Train} | Trains]) ->
    Msg = {destroy_train, {train, Train}, {}},
    case do_port_call(Port, Msg) of
	ok ->
	    destroy_trains(Port, Trains);
	{error, Reason} ->
	    io:format("FANNERL: exit when destroying train with reason ~p~n",
		      [Reason]),
	    destroy_trains(Port, Trains)
    end.


%% @private
convert_message({Cmd, {train, _Ref}, Rest}, TrainPtr) ->
    {Cmd, TrainPtr, Rest};
convert_message({Cmd, _Ref, Rest}, NetworkPtr) ->
    {Cmd, NetworkPtr, Rest}.

convert_message({Cmd, {_Ref, _TrainRef}, Rest}, NetworkPtr, TrainPtr) ->
    {Cmd, {NetworkPtr, TrainPtr}, Rest}.

%% @private
get_train_ref({_Cmd, {_Ref, TrainRef}, _Rest}) ->
    TrainRef.

%% @private
get_ref({_Cmd, {Ref, _TrainRef}, _Rest}) ->
    Ref;
get_ref({_Cmd, Ref, _Rest}) ->
    Ref.

%% @private
handle_port_call(Port, State, Caller, {create_standard, _}=Msg) ->
    call_port_with_msg(Port, State, Caller, Msg, undefined);
handle_port_call(Port, State, Caller, {create_from_file, _}=Msg) ->
    call_port_with_msg(Port, State, Caller, Msg, undefined);
handle_port_call(Port, State, Caller, {read_train_from_file, _}=Msg) ->
    call_port_with_msg(Port, State, Caller, Msg, undefined);
handle_port_call(Port, State, Caller, {_Cmd, {train, TrainRef}, _}=Msg) ->
    case valid_train(TrainRef, State) of
	{true, TrainPtr} ->
	    NewMsg = convert_message(Msg, TrainPtr),
	    call_port_with_msg(Port, State, Caller, NewMsg, TrainRef);	
	false ->
	    io:format("The passed training data ref ~p is not valid~n",
		      [{train,TrainRef}]),
	    Caller ! {fannerl_exception, invalid_training_data},
	    loop(Port, State)
    end;
handle_port_call(Port, State, Caller,
		 {merge_train, {trains, TrainRef1, TrainRef2}, _}) ->
    case valid_train(TrainRef1, State) of
	{true, TrainPtr1} ->
	    case valid_train(TrainRef2, State) of
		{true, TrainPtr2} ->
		    NewMsg= {merge_train, {TrainPtr1, TrainPtr2}, {}},
		    call_port_with_msg(Port, State, Caller, NewMsg, TrainRef1);
		false ->
		    io:format("The passed training data ref ~p is not valid~n",
			      [{train,TrainRef2}]),
		    Caller ! {fannerl_exception, invalid_training_data},
		    loop(Port, State)
	    end;
	false ->
	    io:format("The passed training data ref ~p is not valid~n",
		      [{train,TrainRef1}]),
	    Caller ! {fannerl_exception, invalid_training_data},
	    loop(Port, State)
    end;
handle_port_call(Port, State, Caller, {_Cmd, {Ref, TrainRef}, _}=Msg) ->
    Ref = get_ref(Msg),
    TrainRef = get_train_ref(Msg),
    case valid_network(Ref, State) of
	{true, NetworkPtr} ->
	    case valid_train(TrainRef, State) of
		{true, TrainPtr} ->
		    NewMsg = convert_message(Msg, NetworkPtr, TrainPtr),
		    call_port_with_msg(Port, State, Caller, NewMsg, Ref);
		false ->
		    io:format("The passed training data ref ~p is not valid~n",
			      [{train,Ref}]),
		    Caller ! {fannerl_exception, invalid_training_data},
		    loop(Port, State)
	    end;
	false ->
	    Caller ! {fannerl_exception, invalid_network},
	    loop(Port, State)
    end;
handle_port_call(Port, State, Caller, Msg) ->
    Ref = get_ref(Msg),
    case valid_network(Ref, State) of
	{true, NetworkPtr} ->
	    NewMsg = convert_message(Msg, NetworkPtr),
	    call_port_with_msg(Port, State, Caller, NewMsg, Ref);
	false ->
	    Caller ! {fannerl_exception, invalid_network},
	    loop(Port, State)
    end.



%% @private
call_port_with_msg(Port, State, Caller, Msg, Ref) ->
    Ret =  do_port_call(Port, Msg),
    NewState = handle_return_val(
		 Msg, Ret, Caller, State, Ref),
    loop(Port, NewState).
	
do_port_call(Port, Msg) ->
    erlang:port_command(Port, term_to_binary(Msg)),
    receive
	{Port, {data, Data}} ->
	    binary_to_term(Data);
	{Port, {exit_status, Status}} when Status > 128 ->
	    io:format("Port terminated with signal: ~p~n",[Status-128]),
	    exit({port_terminated, Status});
	{Port, {exit_status, Status}} ->
	    io:format("Port terminated with signal: ~p~n",[Status]),
	    exit({port_terminated, Status});
	{'EXIT', Port, Reason} ->
	    exit(Reason)
    end.

%% @private
handle_return_val({subset_train_data, _, _}, {ok, Ptr}, Caller, State, _Ref) ->
    Ref = make_ref(),
    Caller ! {fannerl_res, Ref},
    State#{trains := dict:store(Ref, Ptr, maps:get(trains, State))};
handle_return_val({read_train_from_file, _}, {ok, Ptr}, Caller, State, _Ref) ->
    Ref = make_ref(),
    Caller ! {fannerl_res, Ref},
    State#{trains := dict:store(Ref, Ptr, maps:get(trains, State))};
handle_return_val({merge_train, _, _}, {ok, Ptr}, Caller, State, _Ref) ->
    Ref = make_ref(),
    Caller ! {fannerl_res, Ref},
    State#{trains := dict:store(Ref, Ptr, maps:get(trains, State))};
handle_return_val({duplicate_train, _, _}, {ok, Ptr}, Caller, State, _Ref) ->
    Ref = make_ref(),
    Caller ! {fannerl_res, Ref},
    State#{trains := dict:store(Ref, Ptr, maps:get(trains, State))};
handle_return_val({create_standard, _}, {ok, Ptr}, Caller, State, _Ref) ->
    %% Hide the ptr by giving a ref to the user instead
    Ref = make_ref(),
    Caller ! {fannerl_res, Ref},
    State#{networks := dict:store(Ref, Ptr, maps:get(networks, State))};
handle_return_val({create_from_file, _}, {ok, Ptr}, Caller, State, _Ref) ->
    %% Hide the ptr by giving a ref to the user instead
    Ref = make_ref(),
    Caller ! {fannerl_res, Ref},
    State#{networks := dict:store(Ref, Ptr, maps:get(networks, State))};
handle_return_val({copy, _, _}, {ok, Ptr}, Caller, State, _Ref) ->
    %% Hide the ptr by giving a ref to the user instead
    NewRef = make_ref(),
    Caller ! {fannerl_res, NewRef},
    State#{networks := dict:store(NewRef, Ptr, maps:get(networks, State))};
handle_return_val({destroy, _, _}, ok, Caller, State, Ref) ->
    Caller ! {fannerl_res, ok},
    Networks = maps:get(networks, State),
    State#{networks := dict:erase(Ref, Networks)};
handle_return_val({destroy_train, _, _}, ok, Caller, State, Ref) ->
    Caller ! {fannerl_res, ok},
    Trains = maps:get(trains, State),
    State#{trains := dict:erase(Ref, Trains)};
handle_return_val(_Msg, Return, Caller, State, _Ref) ->
    Caller ! {fannerl_res, Return},
    State.
    
%% @private
default_options() ->
    #{type=>standard}.
