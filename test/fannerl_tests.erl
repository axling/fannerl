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
      fun fannerl_create_with_learning_rate/1
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
      fun fannerl_train_shuffle/1
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
    {ok, R} = fannerl:create({3,3,1}),
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
    {ok, R} = fannerl:create({2,2,1}),
    ?_assert(ok == fannerl:destroy(R)).
    

fannerl_multiple_create(#{instances := Pids}) ->
    ?_test(
       begin
	   random:seed(now()),
	   Fanns =
	       lists:map(
		 fun(_) ->
			 {ok, R} = fannerl:create({random:uniform(100),
						   random:uniform(100),
						   random:uniform(100)}),
			 R
		 end, lists:seq(1,50)),

	   OtherFanns = 
	       lists:map(
		 fun(Pid) ->
			 LocalFanns =
			     lists:map(
			       fun(_) ->
				       {ok, R} = fannerl:create(
						   Pid,
						   {random:uniform(100),
						    random:uniform(100),
						    random:uniform(100)}),
				       R
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
			       ?_assert(ok == fannerl:destroy(Pid, Ref))
		       end, LocalFanns)
	     end, OtherFanns)
       end).

fannerl_create_with_learning_rate(_Map) ->
    ?_test(
       begin
	   LearningRate = 0.2,
	   {ok, R} = fannerl:create({5,7,3}, #{learning_rate => LearningRate}),
	   #{learning_rate := LearningRateAfter} = fannerl:get_params(R),
	   ?_assert(LearningRateAfter == LearningRate),
	   ok = fannerl:destroy(R)
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
	   {ok, R2} = fannerl:create(FileName),
	   ?assert(ok == fannerl:destroy(R2))
       end).


fannerl_train_create(_) ->
    ?_test(
       begin
	   PrivDir = code:priv_dir(fannerl),
	   Filename = filename:join(PrivDir, "xor.data"),
	   {ok, _N} = fannerl:read_train_from_file(Filename)
       end).

fannerl_train_shuffle(_) ->
    ?_test(
       begin
	   PrivDir = code:priv_dir(fannerl),
	   Filename = filename:join(PrivDir, "xor.data"),
	   {ok, N} = fannerl:read_train_from_file(Filename),
	   ?assert(ok == fannerl:shuffle_train(N))
       end).
