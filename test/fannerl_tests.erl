-module(fannerl_tests).

-include_lib("eunit/include/eunit.hrl").

%%-----------------------------------------------------------
%% TEST DESCRIPTION
%%-----------------------------------------------------------
fannerl_start_stop_test_() ->
    {"Tests that fannerl can be started and stopped accordingly",
     [fun fannerl_start_stop/0,
      fun fannerl_multiple_start_stop/0,
      fun fannerl_stop_on_unstarted_error/0
     ]}.

fannerl_create_destroy_test_() ->
    {foreach, 
     fun setup/0,
     fun cleanup/1,
     [
      fun fannerl_create/1,
      fun fannerl_multiple_create/1,
      fun fannerl_create_with_learning_rate/1,
      fun fannerl_create_from_file/1,
      fun fannerl_copy/1,
      fun fannerl_create_and_get_params/1
     ]
    }.

fannerl_save_test_() ->
    {foreach,
     fun setup_for_save/0,
     fun cleanup_for_save/1,
     [
      fun fannerl_save/1,
      fun fannerl_create_save_create/1
     ]
    }.

fannerl_train_test_() ->
    {foreach,
     fun setup/0,
     fun cleanup/1,
     [
      fun fannerl_train_create/1,
      fun fannerl_train_shuffle/1,
      fun fannerl_train_subset/1,
      fun fannerl_train_epoch/1,
      fun fannerl_train_on_data/1,
      fun fannerl_train/1,
      fun fannerl_train_on_file/1
     ]
    }.

fannerl_run_and_test_test_() ->
    {foreach,
     fun setup/0,
     fun cleanup/1,
     [
      fun fannerl_run/1,
      fun fannerl_test/1,
      fun fannerl_test_data/1,
      fun fannerl_randomize_weights/1,
      fun fannerl_init_weights/1,
      fun fannerl_reset_mse/1
     ]
    }.

%%-----------------------------------------------------------
%% FIXTURE FUNCTIONS
%%-----------------------------------------------------------
setup() ->
    _Pid = fannerl:start(),
    Pid2 = fannerl:start_instance(),
    #{instances=>[Pid2]}.

cleanup(#{instances := Pids}) ->
    ok = fannerl:stop(),
    lists:foreach(
      fun(Pid) ->
	      fannerl:stop_instance(Pid)
      end, Pids).

setup_for_save() ->
    FileName = "fannerl_test_saving_network.net",
    _Pid = fannerl:start(),
    R = fannerl:create({3,3,1}),
    %% Check if file already exists, if so we delete it
    ok = check_and_delete_file(FileName),
    #{filename=>FileName, net=>R}.

cleanup_for_save(#{filename:=FileName, net:=R}) ->
    ok = fannerl:destroy(R),
    ok = fannerl:stop(),
    check_and_delete_file(FileName).

check_and_delete_file(FileName) ->
    case check_file_exists(FileName) of
	true ->
	    ok = file:delete(FileName);
	false ->
	    ok
    end.

check_file_exists(FileName) ->
    {ok, Dir} = file:get_cwd(),
    {ok, FileNames} = file:list_dir(Dir),
    lists:member(FileName, FileNames).	

%%-----------------------------------------------------------
%% TESTS
%%-----------------------------------------------------------
fannerl_start_stop() ->
    Pid = fannerl:start(),
    ?assert(is_process_alive(Pid) == true),
    ok = fannerl:stop(),
    ?assert(is_process_alive(Pid) == false).

fannerl_multiple_start_stop() ->
    Pid = fannerl:start(),
    Pid2 = fannerl:start_instance(),
    ?assert(is_process_alive(Pid) == true),
    ?assert(is_process_alive(Pid2) == true),
    ok = fannerl:stop_instance(Pid2),
    ok = fannerl:stop(),
    ?assert(is_process_alive(Pid) == false),
    ?assert(is_process_alive(Pid2) == false).

fannerl_stop_on_unstarted_error() ->
    ?assert(fannerl:stop() == {error, fannerl_not_started}).

fannerl_create(_Map) ->
    R = fannerl:create({2,2,1}),
    ?_assert(ok == fannerl:destroy(R)).
    

fannerl_multiple_create(#{instances := Pids}) ->
    ?_test(
       begin
	   random:seed(now()),
	   Fanns =
	       lists:map(
		 fun(_) ->
			 _R = fannerl:create({random:uniform(100),
					     random:uniform(100),
					     random:uniform(100)})
		 end, lists:seq(1,50)),

	   OtherFanns = 
	       lists:map(
		 fun(Pid) ->
			 LocalFanns =
			     lists:map(
			       fun(_) ->
				       _R = fannerl:create_on(
						   Pid,
						   {random:uniform(100),
						    random:uniform(100),
						    random:uniform(100)})
			       end, lists:seq(1,50)),
			 {Pid, LocalFanns}
		 end, Pids),

	   lists:foreach(
	     fun(Ref) ->
		     ?_assert(ok == fannerl:destroy(Ref))
	     end, Fanns),

	   lists:foreach(
	     fun({Pid, LocalFanns}) ->
		     lists:foreach(
		       fun(Ref) ->
			       ?_assert(ok == fannerl:destroy_on(Pid, Ref))
		       end, LocalFanns)
	     end, OtherFanns)
       end).

fannerl_create_with_learning_rate(_Map) ->
    ?_test(
       begin
	   LearningRate = 0.2,
	   R = fannerl:create({5,7,3}, #{learning_rate => LearningRate}),
	   Map = fannerl:get_params(R),
	   #{learning_rate := LearningRateAfter} = Map,
	   ok = fannerl:destroy(R),
	   ?_assert(LearningRateAfter == LearningRate)
       end).

fannerl_create_and_get_params(_Map) ->
    ?_test(
       begin
	   NumInput = 5,
	   NumOutput = 3,
	   Hidden = 7,
	   Net = {NumInput, Hidden, NumOutput},
	   R = fannerl:create(Net),
	   Map = fannerl:get_params(R),
	   _TotalNeurons = NumInput + NumOutput + Hidden,
	   _TotalConnections = NumInput * Hidden + Hidden*NumOutput,
	   NumLayers = tuple_size(Net),
	   #{learning_rate := _LearningRate,
	     learning_momentum := _LearningMomentum,
	     training_algorithm := _TrainingAlgorithm,
	     mean_square_error := _MeanSquareError,
	     bit_fail := _BitFail,
	     train_error_function := _TrainErrorFunc,
	     network_type := _NetType,
	     num_output := NumOutput,
	     num_input := NumInput,
	     total_neurons := _TotalNeuronsPlusBias,
	     total_connections := _TotalConnectionsPlusBiasConns,
	     connection_rate := 1.0,
	     num_layers := NumLayers,
	     layers := Net,
	     bias := _Bias,
	     connections := Connections
	     } = Map,
	   ?assert(is_map(Connections)),
	   _Val = maps:get({0,Hidden-1}, Connections),
	   ok = fannerl:destroy(R)
       end).

fannerl_create_from_file(_Map) ->
    File = "temp_network_xxr23d",
    check_and_delete_file(File),
    ?_test(
       begin
	   R= fannerl:create({2,2,1}),
	   ok = fannerl:save(R, File),
	   
	   R2 = fannerl:create_from_file(File),
	   ?assert(ok == fannerl:destroy(R)),
	   ?assert(ok == fannerl:destroy(R2)),
	   check_and_delete_file(File)
       end).

fannerl_copy(_Map) ->
    ?_test(
       begin
	   R = fannerl:create({2,2,1}),
	   R2 = fannerl:copy(R),
	   R3 = fannerl:copy(R2),
	   ?assert(R /= R2),
	   ?assert(R2 /= R3),
	   ?assert(ok == fannerl:destroy(R)),
	   ?assert(ok == fannerl:destroy(R2)),
	   ?assert(ok == fannerl:destroy(R3))
       end).

    
fannerl_save(#{net:=R, filename:=FileName}) ->
    ?_test(
       begin
	   ?assert(false == check_file_exists(FileName)),
	   ?assert(ok == fannerl:save(R, FileName)),
	   ?assert(true == check_file_exists(FileName))
       end).

fannerl_create_save_create(#{net:=R, filename:=FileName}) ->
    ?_test(
       begin
	   ?assert(false == check_file_exists(FileName)),
	   ?assert(ok == fannerl:save(R, FileName)),
	   ?assert(true == check_file_exists(FileName)),
	   R2 = fannerl:create_from_file(FileName),
	   ?assert(ok == fannerl:destroy(R2))
       end).


fannerl_train_create(_) ->
    ?_test(
       begin
	   PrivDir = code:priv_dir(fannerl),
	   Filename = filename:join(PrivDir, "xor.data"),
	   _N = fannerl:read_train_from_file(Filename)
       end).

fannerl_train_shuffle(_) ->
    ?_test(
       begin
	   PrivDir = code:priv_dir(fannerl),
	   Filename = filename:join(PrivDir, "xor.data"),
	   N = fannerl:read_train_from_file(Filename),
	   ?assert(ok == fannerl:shuffle_train(N))
       end).

fannerl_train_subset(_) ->
    ?_test(
       begin
	   PrivDir = code:priv_dir(fannerl),
	   Filename = filename:join(PrivDir, "xor.data"),
	   N = fannerl:read_train_from_file(Filename),
	   NewTrain = fannerl:subset_train_data(N, 2, 1),
	   ?assert(N /= NewTrain)
       end).

fannerl_train_epoch(_) ->
    ?_test(
       begin
	   R = fannerl:create({2,2,1}),
	   PrivDir = code:priv_dir(fannerl),
	   Filename = filename:join(PrivDir, "xor.data"),
	   N = fannerl:read_train_from_file(Filename),
	   ok = fannerl:train_epoch(R, N),
	   ?assert(ok == fannerl:destroy(R))
       end).

fannerl_train_on_data(_) ->
    ?_test(
       begin
	   R = fannerl:create({2,2,1}),
	   PrivDir = code:priv_dir(fannerl),
	   Filename = filename:join(PrivDir, "xor.data"),
	   N = fannerl:read_train_from_file(Filename),
	   ok = fannerl:train_on_data(R, N, 5, 0.1),
	   ?assert(ok == fannerl:destroy(R))
       end).

fannerl_train(_) ->
    ?_test(
       begin
	   R = fannerl:create({2,2,1}),
	   ok = fannerl:train(R, {1,1}, {0}),
	   ?assert(ok == fannerl:destroy(R))
       end).

fannerl_train_on_file(_) ->
    ?_test(
       begin
	   R = fannerl:create({2,2,1}),
	   PrivDir = code:priv_dir(fannerl),
	   Filename = filename:join(PrivDir, "xor.data"),
	   ok = fannerl:train_on_file(R, Filename, 5, 0.1),
	   ?assert(ok == fannerl:destroy(R))
       end).

fannerl_run(_) ->
    ?_test(
       begin
	   R = fannerl:create({4,3,2}),
	   {_X, _Y} = fannerl:run(R, {1,1,1,1}),
	   ?_assert(ok == fannerl:destroy(R))
       end).

fannerl_test(_) ->
    ?_test(
       begin
	   R = fannerl:create({4,3,2}),
	   {_X, _Y} = fannerl:test(R, {1,1,1,1}, {1,1}),
	   ?_assert(ok == fannerl:destroy(R))
       end).

fannerl_test_data(_) ->
    ?_test(
       begin
	   R = fannerl:create({2,2,1}),
	   PrivDir = code:priv_dir(fannerl),
	   Filename = filename:join(PrivDir, "xor.data"),
	   N = fannerl:read_train_from_file(Filename),
	   {ok, _} = fannerl:test_data(R, N),
	   ?assert(ok == fannerl:destroy(R))
       end).


fannerl_randomize_weights(_) ->
    ?_test(
       begin
	   R = fannerl:create({2,2,1}),
	   ok = fannerl:randomize_weights(R, -20.0, 20),
	   ?assert(ok == fannerl:destroy(R))
       end).

fannerl_init_weights(_) ->
    ?_test(
       begin
	   R = fannerl:create({2,2,1}),

	   PrivDir = code:priv_dir(fannerl),
	   Filename = filename:join(PrivDir, "xor.data"),
	   N = fannerl:read_train_from_file(Filename),
	   
	   ok = fannerl:init_weights(R, N),
	   
	   ?assert(ok == fannerl:destroy(R))
       end).

fannerl_reset_mse(_) ->
    ?_test(
       begin
	   R = fannerl:create({2,2,1}),
	   %% TODO: test setting, resetting, then getting, no only test that
	   %% nothing crashes
	   ok = fannerl:reset_mse(R), 
	   ?assert(ok == fannerl:destroy(R))
       end).
