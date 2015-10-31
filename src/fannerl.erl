%%% @author Erik Axling <>
%%% @copyright (C) 2014, Erik Axling
%%% @doc
%%%
%%% @end
%%% Created : 24 Nov 2014 by Erik Axling <>

-module(fannerl).

-type network_layers() :: tuple().
-type network_ref() :: reference().

-export([start/0,
	 start_instance/0,
	 stop/0,
	 stop_instance/1,
	 init/1]).

-export([create/1,
	 create/2,
	 create/3,
	 destroy/1,
	 destroy/2,
	 get_params/1,
	 get_params/2
	]).

-export([train/2,
	 train/3,
	 train/4,
	 train_on_file/4,
	 train_on_file/5,
	 read_train_from_file/1,
	 read_train_from_file/2,
	 shuffle_train/1,
	 shuffle_train/2,
	 subset_train_data/3,
	 subset_train_data/4]).

-export([test/2,
	 test/3,
	 test/4]).

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
%% @doc Creates an artificial neural network with any number of layers. 
%% The Layers tuple size describe the number of layers while each position
%% sets the size of the layer. See the FANN documentation of create_standard:
%% [http://libfann.github.io/fann/docs/files/fann-h.html#fann_create_standard]
%% @end
%% --------------------------------------------------------------------- %%
-spec create(Layers :: network_layers()) -> network_ref();
	    (FileName :: string()) -> network_ref().
create(Network) when is_tuple(Network) ->
    create(?MODULE, Network, default_options());

%% --------------------------------------------------------------------- %%
%% @doc Creates an artificial neural network from a previously save file. 
%% See the FANN documentation of create_standard:
%% [http://libfann.github.io/fann/docs/files/fann_io-h.html#fann_create_from_file]
%% @end
%% --------------------------------------------------------------------- %%
create(FileName) when is_list(FileName) ->
    create_from_file(?MODULE, FileName).


create(Network, Options)
  when is_tuple(Network),
       is_map(Options) ->
    create(?MODULE, Network, Options);

create(Instance, Network)
  when Instance == ?MODULE; is_pid(Instance),
       is_tuple(Network) ->
    create(Instance, Network, default_options());

create(Instance, FileName)
  when Instance == ?MODULE; is_pid(Instance),
       is_list(FileName) ->
    create_from_file(Instance, FileName).

create(Instance, Network, Options) 
  when Instance == ?MODULE; is_pid(Instance),
      is_tuple(Network),
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
			 {Network, Type,
			  ConnRate, OptionsWithoutConnRateAndType}}).

create_from_file(Instance, FileName)
  when Instance == ?MODULE; is_pid(Instance),
       is_list(FileName) ->
    call_port(Instance, {create_from_file, FileName}).

destroy(Ref) when is_reference(Ref) ->
    destroy(?MODULE, Ref).

destroy(Instance, Ref)
  when is_pid(Instance); Instance == ?MODULE,
       is_reference(Ref) ->
    call_port(Instance, {destroy, Ref, {}}).

train(Ref, TrainRef)
  when is_reference(Ref),
       is_reference(TrainRef) ->
    train(?MODULE, Ref, TrainRef).

train(Instance, Ref, TrainRef)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Ref),
       is_reference(TrainRef) ->
    call_port(Instance, {train_epoch, {Ref, TrainRef}, {}});


train(Ref, Input, DesiredOutput) 
  when is_reference(Ref),
       is_tuple(Input),
       is_tuple(DesiredOutput) ->
    train(?MODULE, Ref, Input, DesiredOutput).

train(Instance, Ref, Input, DesiredOutput)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Ref),
       is_tuple(Input),
       is_tuple(DesiredOutput) ->
    call_port(Instance, {train, Ref, {Input, DesiredOutput}});

train(Ref, TrainRef, MaxEpochs, DesiredError)
  when is_reference(Ref),
       is_reference(TrainRef),
       is_integer(MaxEpochs),
       is_number(DesiredError) ->
    train(?MODULE, Ref, TrainRef, MaxEpochs, DesiredError).

train(Instance, Ref, TrainRef, MaxEpochs, DesiredError)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Ref),
       is_reference(TrainRef),
       is_integer(MaxEpochs),
       is_number(DesiredError) ->
    call_port(?MODULE, {train_on_data, {Ref, TrainRef},
			{MaxEpochs, DesiredError}}).

train_on_file(Ref, FileName, MaxEpochs, DesiredError) 
  when is_reference(Ref),
       is_list(FileName),
       is_integer(MaxEpochs), MaxEpochs > 0,
       is_float(DesiredError) ->
    train_on_file(?MODULE, Ref, FileName, MaxEpochs, DesiredError).

train_on_file(Instance, Ref, FileName, MaxEpochs, DesiredError) 
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Ref),
       is_list(FileName),
       is_integer(MaxEpochs), MaxEpochs > 0,
       is_float(DesiredError) ->
    call_port(Instance, 
	      {train_on_file, Ref, {FileName, MaxEpochs, DesiredError}}).

test(Ref, TrainRef)
  when is_reference(Ref),
       is_reference(TrainRef) ->
    test(?MODULE, Ref, TrainRef).

test(Instance, Ref, TrainRef)
  when Instance == ?MODULE; is_pid(Instance),
       is_reference(Ref),
       is_reference(TrainRef) ->
    call_port(Instance, {test_data, {Ref, TrainRef}, {}});


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

call(Instance, Msg) ->
    Instance ! {call, self(), Msg},
    receive
	{fannerl_res, Result} ->
	    Result
    end.

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

loop(Port, State) ->
    receive
	{call, Caller, Msg} ->
	    handle_port_call(Port, State, Caller, Msg);
	stop ->
	    erlang:port_close(Port),
	    exit(normal)
    end.

valid_network(Ref, State) ->
    case dict:is_key(Ref, maps:get(networks, State)) of
	true ->
	    {true, dict:fetch(Ref, maps:get(networks, State))};
	false ->
	    false
    end.

valid_train(Ref, State) ->
    case dict:is_key(Ref, maps:get(trains, State)) of
	true ->
	    {true, dict:fetch(Ref, maps:get(trains, State))};
	false ->
	    false
    end.

convert_message({Cmd, {train, _Ref}, Rest}, TrainPtr) ->
    {Cmd, TrainPtr, Rest};
convert_message({Cmd, _Ref, Rest}, NetworkPtr) ->
    {Cmd, NetworkPtr, Rest}.

convert_message({Cmd, {_Ref, _TrainRef}, Rest}, NetworkPtr, TrainPtr) ->
    {Cmd, {NetworkPtr, TrainPtr}, Rest}.

get_train_ref({_Cmd, {_Ref, TrainRef}, _Rest}) ->
    TrainRef.

get_ref({_Cmd, {Ref, _TrainRef}, _Rest}) ->
    Ref;
get_ref({_Cmd, Ref, _Rest}) ->
    Ref.

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
    

default_options() ->
    #{type=>standard}.
