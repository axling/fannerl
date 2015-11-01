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
	 destroy/1,
	 destroy_on/2,
	 get_params/1,
	 get_params/2
	]).

-export([train_epoch/2,
	 train_epoch_on/3,
	 train/3,
	 train_on/4,
	 train_on_data/4,
	 train_on_data_on/5,
	 train_on_file/4,
	 train_on_file_on/5,
	 read_train_from_file/1,
	 read_train_from_file/2,
	 shuffle_train/1,
	 shuffle_train/2,
	 subset_train_data/3,
	 subset_train_data/4]).

-export([test/3,
	 test/4,
	 test_data/2,
	 test_data_on/3
	]).

-export([save/2,
	 save/3]).

-export([run/2]).

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
%% See the FANN documentation of create_standard:
%% [http://libfann.github.io/fann/docs/files/fann_io-h.html#fann_create_from_file]
%% @end
%% --------------------------------------------------------------------- %%
-spec create_from_file(Instance::pid(), FileName :: string()) -> network_ref().
create_from_file(Instance, FileName)
  when Instance == ?MODULE; is_pid(Instance),
       is_list(FileName) ->
    call_port(Instance, {create_from_file, FileName}).

%% --------------------------------------------------------------------- %%
%% @equiv destroy_on({@module}, Network)
%% @end
%% --------------------------------------------------------------------- %%
-spec destroy(Network::network_ref) -> ok.
destroy(Network) when is_reference(Network) ->
    destroy_on(?MODULE, Network).

%% --------------------------------------------------------------------- %%
%% @doc This will destroy all references to the neural network and free
%%      up all associated memory, see 
%%      [http://libfann.github.io/fann/docs/files/fann-h.html#fann_destroy]
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
-spec test_data(Network::network_ref(), Train::train_ref()) -> ok.
test_data(Network, Train)
  when is_reference(Network),
       is_reference(Train) ->
    test_data_on(?MODULE, Network, Train).

%% --------------------------------------------------------------------- %%
%% @doc Test a set of training data and calculates the MSE for the training data. See [http://libfann.github.io/fann/docs/files/fann_train-h.html#fann_test_data].
%% @end
%% --------------------------------------------------------------------- %%
test_data_on(Instance, Ref, TrainRef)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Ref),
       is_reference(TrainRef) ->
    call_port(Instance, {test_data, {Ref, TrainRef}, {}}).


test(Ref, Input, DesiredOutput)
  when is_reference(Ref),
       is_tuple(Input),
       is_tuple(DesiredOutput) ->
    test(?MODULE, Ref, Input, DesiredOutput).

test(Instance, Ref, Input, DesiredOutput)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Ref),
       is_tuple(Input),
       is_tuple(DesiredOutput) ->
    call_port(Instance, {test, Ref, {Input, DesiredOutput}}).
    
read_train_from_file(FileName)
  when is_list(FileName) ->
    read_train_from_file(?MODULE, FileName).

read_train_from_file(Instance, FileName)
  when Instance == ?MODULE; is_pid(Instance),
       is_list(FileName) ->
    call_port(Instance, {read_train_from_file, FileName}).

shuffle_train(TrainRef)
  when is_reference(TrainRef) ->
    shuffle_train(?MODULE, TrainRef).

shuffle_train(Instance, TrainRef)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(TrainRef) ->
    call_port(Instance, {shuffle_train,{train, TrainRef}, {}}).

run(Ref, Input) 
  when is_reference(Ref),
       is_tuple(Input) ->
    run(?MODULE, Ref, Input).

run(Instance, Ref, Input) 
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Ref),
       is_tuple(Input) ->
    call_port(Instance, {run, Ref, {Input}}).

save(Ref, FileName)
  when is_reference(Ref),
       is_list(FileName) ->
    save(?MODULE, Ref, FileName).

save(Instance, Ref, FileName)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Ref),
       is_list(FileName) ->
    call_port(Instance, {save_to_file, Ref, {FileName}}).

subset_train_data(TrainRef, Pos, Length) ->
    subset_train_data(?MODULE, TrainRef, Pos, Length).

subset_train_data(Instance, TrainRef, Pos, Length) 
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(TrainRef),
       is_integer(Pos), Pos >= 0,
       is_integer(Length), Length >= 0 ->
    call_port(Instance, {subset_train_data, {train, TrainRef}, {Pos, Length}}).
    

get_params(Ref) 
  when is_reference(Ref) ->
    get_params(?MODULE, Ref).

get_params(Instance, Ref)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Ref) ->
    call_port(Instance, {get_params, Ref, {}}).

%%****************************************************************%%       
%% Private functions
%%****************************************************************%%       
%% @private
call_port(?MODULE, Msg) ->
    case whereis(?MODULE) of
	Pid when is_pid(Pid) ->
	    call(?MODULE, Msg);
	_ ->
	    {error, not_valid_instance}
    end;
call_port(Instance, Msg) when is_pid(Instance) ->
    case is_process_alive(Instance) of
	true ->
	    call(Instance, Msg);
	false ->
	    {error, process_not_alive}
    end.

%% @private
call(Instance, Msg) ->
    Instance ! {call, self(), Msg},
    receive
	{fannerl_res, Result} ->
	    Result
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
    TrainRef = get_train_ref(Msg),
    case valid_train(TrainRef, State) of
	{true, TrainPtr} ->
	    NewMsg = convert_message(Msg, TrainPtr),
	    call_port_with_msg(Port, State, Caller, NewMsg, TrainRef);
	false ->
	    io:format("The passed training data ref ~p is not valid~n",
		      [{train,TrainRef}]),
	    Caller ! {fannerl_res, {error, invalid_training_data}},
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
		    Caller ! {fannerl_res, {error, invalid_training_data}},
		    loop(Port, State)
	    end;
	false ->
	    io:format("The passed network ref ~p is not valid~n", [Ref]),
	    Caller ! {fannerl_res, {error, invalid_network}},
	    loop(Port, State)
    end;
handle_port_call(Port, State, Caller, Msg) ->
    Ref = get_ref(Msg),
    case valid_network(Ref, State) of
	{true, NetworkPtr} ->
	    NewMsg = convert_message(Msg, NetworkPtr),
	    call_port_with_msg(Port, State, Caller, NewMsg, Ref);
	false ->
	    io:format("The passed network ref ~p is not valid~n", [Ref]),
	    Caller ! {fannerl_res, {error, invalid_network}},
	    loop(Port, State)
    end.

%% @private
call_port_with_msg(Port, State, Caller, Msg, Ref) ->
    erlang:port_command(Port, term_to_binary(Msg)),
    receive
	{Port, {data, Data}} ->
	    NewState = handle_return_val(
			    Msg, binary_to_term(Data), Caller, State, Ref),
	    loop(Port, NewState);
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
handle_return_val({destroy, _}, ok, Caller, State, Ref) ->
    Caller ! {fannerl_res, ok},
    dict:erase(Ref, State);
handle_return_val(_Msg, Return, Caller, State, _Ref) ->
    Caller ! {fannerl_res, Return},
    State.
    
%% @private
default_options() ->
    #{type=>standard}.
