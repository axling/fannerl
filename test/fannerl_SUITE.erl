-module(fannerl_SUITE).

-compile(export_all).

-include_lib("common_test/include/ct.hrl").

suite() ->
    [{timetrap,{seconds,10}}].

init_per_suite(Config) ->
    Config.

end_per_suite(_Config) ->
    ok.

%%init_per_group(_GroupName, Config) ->
%%    Config.

%%end_per_group(_GroupName, _Config) ->
%%    ok.

init_per_testcase(_TestCase, Config) ->
    ct:pal("Starting fannerl", []),
    fannerl:start(),
    Config.

end_per_testcase(_TestCase, _Config) ->
    ct:pal("Stopping fannerl", []),
    fannerl:stop(),
    ok.

all() -> 
    [tc_create_and_destroy,
     tc_multiple_create_and_destroy,
     tc_train,
     tc_train_and_run,
     tc_train_on_file,
     tc_create_sparse_and_train,
     tc_create_shortcut_and_train,
     tc_read_train_from_file_and_train,
     tc_read_train_from_file_and_train_one_epoch,
     tc_test
    ].

tc_create_and_destroy(_Config) ->
    {ok, Ref} = fannerl:create({2,2,1}),
    true = is_reference(Ref),
    ok = fannerl:destroy(Ref),
    
    %% create instance and destroy
    P = fannerl:start_instance(),
    true = is_pid(P),
    {ok, Ref2} = fannerl:create(P, {2,2,1}),
    true = is_reference(Ref2),
    ok = fannerl:destroy(P, Ref2),
    fannerl:stop_instance(P).

    


tc_multiple_create_and_destroy(_Config) ->
    random:seed(now()),
    RefList = 
	lists:map(
	  fun(_X) ->
		  {ok, Ref} = fannerl:create(
				{random:uniform(50),
				 random:uniform(50),
				 random:uniform(50)}),
		  true = is_reference(Ref),
		  Ref
	  end, lists:seq(1,100)),
    lists:foreach(
      fun(Ref) ->
	      ok = fannerl:destroy(Ref)
      end, RefList).
		  
	      
tc_train(_Config) ->
    {ok, Ref} = fannerl:create({2,2,1}),
    %% XOR training data, int
    fannerl:train(Ref, {-1, -1}, {-1}),
    fannerl:train(Ref, {-1, 1}, {1}),
    fannerl:train(Ref, {1, -1}, {1}),
    fannerl:train(Ref, {1, 1}, {-1}),
    Map = fannerl:get_params(Ref),
    ct:pal("Params: ~p", [maps:to_list(Map)]),
    
    %% try floating point
    fannerl:train(Ref, {-1.0, -1.0}, {-1.0}),
    fannerl:train(Ref, {-1.0, 1.0}, {1.0}),
    fannerl:train(Ref, {1.0, -1.0}, {1.0}),
    fannerl:train(Ref, {1.0, 1.0}, {-1.0}),

    %% try mixed
    fannerl:train(Ref, {1.0, 1}, {-1.0}),
    fannerl:train(Ref, {-1.0, 1.0}, {1}),
    fannerl:train(Ref, {1, -1.0}, {1.0}),
    fannerl:train(Ref, {1, 1.0}, {-1}),

    fannerl:destroy(Ref),
    ok.

tc_train_and_run(_Config) ->
    {ok, Ref} = fannerl:create({2,2,1}, #{learning_rate => 0.01,
					 randomize_weights => {-0.5, 0.5}}),

    lists:foreach(
      fun(_X) ->
	      %% XOR training data, int
	      fannerl:train(Ref, {-1.0, -1.0}, {-1.0}),
	      fannerl:train(Ref, {-1.0, 1.0}, {1.0}),
	      fannerl:train(Ref, {1.0, -1.0}, {1.0}),
	      fannerl:train(Ref, {1.0, 1.0}, {-1.0})
      end, lists:seq(1,1000)),
    Map = fannerl:get_params(Ref),
    ct:pal("Params run 1: ~p", [maps:to_list(Map)]),
    Out = fannerl:run(Ref, {-1,-1}),
    ct:pal("Output run 1: {-1,-1} -> ~p", [Out]),
    Out2 = fannerl:run(Ref, {-1.0,-1.0}),
    ct:pal("Output2 run 1: {-1.0,-1.0} -> ~p", [Out2]),
    Out3 = fannerl:run(Ref, {-1,-1.0}),
    ct:pal("Output3 run 1: {-1, -1.0} -> ~p", [Out3]),

    lists:foreach(
      fun(_X) ->
	      %% XOR training data, int
	      fannerl:train(Ref, {-1.0, -1.0}, {-1.0}),
	      fannerl:train(Ref, {-1.0, 1.0}, {1.0}),
	      fannerl:train(Ref, {1.0, -1.0}, {1.0}),
	      fannerl:train(Ref, {1.0, 1.0}, {-1.0})
      end, lists:seq(1,1000)),
    
    Map2 = fannerl:get_params(Ref),
    ct:pal("Params run 2: ~p", [maps:to_list(Map2)]),
    SOut = fannerl:run(Ref, {-1,-1}),
    ct:pal("Output run 2: {-1,-1} -> ~p", [SOut]),
    SOut2 = fannerl:run(Ref, {-1.0,-1.0}),
    ct:pal("Output2 run 2: {-1.0,-1.0} -> ~p", [SOut2]),
    SOut3 = fannerl:run(Ref, {-1,-1.0}),
    ct:pal("Output3 run 2: {-1, -1.0} -> ~p", [SOut3]),
    
    fannerl:destroy(Ref),
    ok.
	      
tc_train_on_file(_Config) ->
    {ok, Ref} = fannerl:create(
		  {2,3,1}, 
		  #{activation_func_hidden => fann_sigmoid_symmetric,
		    activation_func_output => fann_sigmoid_symmetric,
		    learning_rate => 0.5}),
    PrivDir = code:priv_dir(fannerl),
    Filename = filename:join(PrivDir, "xor.data"),
    ok = fannerl:train_on_file(Ref, Filename, 100000, 0.001),
    Map = fannerl:get_params(Ref),
    ct:pal("Params: ~p", [maps:to_list(Map)]),
    Output = fannerl:run(Ref, {-1.0,-1.0}),
    io:format("Output after train is -1,-1 -> ~p~n", [Output]),
    Output2 = fannerl:run(Ref, {1.0,1.0}),
    io:format("Output after train is 1,1 -> ~p~n", [Output2]),
    Output3 = fannerl:run(Ref, {1.0,-1.0}),
    io:format("Output after train is 1,-1 -> ~p~n", [Output3]),
    Output4 = fannerl:run(Ref, {-1.0,1.0}),
    io:format("Output after train is -1,1 -> ~p~n", [Output4]),

    #{mean_square_error := Mse,
      learning_rate := 0.5} = Map,
    case Mse > 0.01 of
	true ->
	    %% MSe is just too large, fail
	    ct:fail(mse_too_large);
	false ->
	    ok
    end.
    
tc_create_sparse_and_train(_Config) ->
    {ok, Ref} = fannerl:create(
		  {2,3,1},
		  #{type => sparse,
		    activation_func_hidden => fann_sigmoid_symmetric,
		    activation_func_output => fann_sigmoid_symmetric,
		    conn_rate => 0.3}),
    PrivDir = code:priv_dir(fannerl),
    Filename = filename:join(PrivDir, "xor.data"),
    ok = fannerl:train_on_file(Ref, Filename, 500000, 0.001),
    Output = fannerl:run(Ref, {-1.0,-1.0}),
    io:format("Output after train is -1,-1 -> ~p~n", [Output]),
    Output2 = fannerl:run(Ref, {1.0,1.0}),
    io:format("Output after train is 1,1 -> ~p~n", [Output2]),
    Output3 = fannerl:run(Ref, {1.0,-1.0}),
    io:format("Output after train is 1,-1 -> ~p~n", [Output3]),
    ok.

tc_create_shortcut_and_train(_Config) ->
    {ok, Ref} = fannerl:create(
		  {2,3,1}, #{type => shortcut,
			     learning_rate => 0.01,
			     activation_func_hidden => fann_sigmoid_symmetric,
			     activation_func_output => fann_sigmoid_symmetric}),
    PrivDir = code:priv_dir(fannerl),
    Filename = filename:join(PrivDir, "xor.data"),
    ok = fannerl:train_on_file(Ref, Filename, 500000, 0.001),
    Output = fannerl:run(Ref, {-1.0,-1.0}),
    io:format("Output after train is -1,-1 -> ~p~n", [Output]),
    Output2 = fannerl:run(Ref, {1.0,1.0}),
    io:format("Output after train is 1,1 -> ~p~n", [Output2]),
    Output3 = fannerl:run(Ref, {1.0,-1.0}),
    io:format("Output after train is 1,-1 -> ~p~n", [Output3]),
    ok.

tc_read_train_from_file_and_train(_Config) ->
    PrivDir = code:priv_dir(fannerl),
    Filename = filename:join(PrivDir, "xor.data"),
    {ok, TrainRef} = fannerl:read_train_from_file(Filename),
    
    {ok, Ref} = fannerl:create(
		  {2,3,1}, #{learning_rate => 0.01,
			     activation_func_hidden => fann_sigmoid_symmetric,
			     activation_func_output => fann_sigmoid_symmetric}),
    Map = fannerl:get_params(Ref),
    ct:pal("Params run 1: ~p", [maps:to_list(Map)]),
    ok = fannerl:train(Ref, TrainRef, 10000, 0.1),
    Map2 = fannerl:get_params(Ref),
    ct:pal("Params run 2: ~p", [maps:to_list(Map2)]).

tc_read_train_from_file_and_train_one_epoch(_Config) ->
    PrivDir = code:priv_dir(fannerl),
    Filename = filename:join(PrivDir, "xor.data"),
    {ok, TrainRef} = fannerl:read_train_from_file(Filename),
    
    {ok, Ref} = fannerl:create(
		  {2,3,1}, #{learning_rate => 0.01,
			     activation_func_hidden => fann_sigmoid_symmetric,
			     activation_func_output => fann_sigmoid_symmetric}),
    Map = fannerl:get_params(Ref),
    ct:pal("Params run 1: ~p", [maps:to_list(Map)]),
    ok = fannerl:train(Ref, TrainRef),
    Map2 = fannerl:get_params(Ref),
    ct:pal("Params run 2: ~p", [maps:to_list(Map2)]),
    {ok, Mse} = fannerl:test(Ref, TrainRef),
    ct:pal("MSE = ~p", [Mse]).

tc_test(_Config) ->
    {ok, Ref} = fannerl:create(
		  {2,3,1}, #{learning_rate => 0.01,
			     activation_func_hidden => fann_sigmoid_symmetric,
			     activation_func_output => fann_sigmoid_symmetric}),
    lists:foreach(
      fun(_X) ->
	      %% XOR training data, int
	      fannerl:train(Ref, {-1.0, -1.0}, {-1.0}),
	      fannerl:train(Ref, {-1.0, 1.0}, {1.0}),
	      fannerl:train(Ref, {1.0, -1.0}, {1.0}),
	      fannerl:train(Ref, {1.0, 1.0}, {-1.0})
      end, lists:seq(1,1000)),
    {Out} = fannerl:test(Ref, {-1.0, -1.0}, {-1}),
    ct:pal("OutPut from test: {-1,-1} --> ~p~n", [Out]),
    Map = fannerl:get_params(Ref),
    ct:pal("Params run: ~p", [maps:to_list(Map)]).
