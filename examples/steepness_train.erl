%% Fannerl: Erlang Bindings to the Fast Artificial Neural Network Library (fann)
%% Copyright (C) 2015 Erik Axling (erik.axling@gmail.com)

%% This library is free software; you can redistribute it and/or
%% modify it under the terms of the GNU Lesser General Public
%% License as published by the Free Software Foundation; either
%% version 2.1 of the License, or (at your option) any later version.

%% This library is distributed in the hope that it will be useful,
%% but WITHOUT ANY WARRANTY; without even the implied warranty of
%% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
%% Lesser General Public License for more details.

%% You should have received a copy of the GNU Lesser General Public
%% License along with this library; if not, write to the Free Software
%% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

%% The fannerl version of FANNs mushroom.c example
-module(steepness_train).

-export([run/0]).

run() ->
    NumInput = 2,
    NumOutput = 1, 
    NumNeuronsHidden = 3,
    DesiredError = 0.001,
    MaxEpochs = 500000,
    EpochBetweenReports = 1000,
    SteepnessStart = 1,
    SteepnessStep = 0.1,
    SteepnessEnd = 20,

    fannerl:start(),

    Network = fannerl:create({NumInput, NumNeuronsHidden, NumOutput}),
    Train = fannerl:read_train_from_file("../priv/xor.data"),
    
    ok = fannerl:set_activation_function_all(Network, fann_sigmoid_symmetric),
    ok = fannerl:set_param(Network, training_algorithm, fann_train_quickprop),
    
    train_on_steepness(Network, Train, MaxEpochs, EpochBetweenReports,
		      DesiredError, SteepnessStart, SteepnessStep,
		       SteepnessEnd),
    
    ok = fannerl:set_activation_function_all(Network, fann_threshold_symmetric),
    
    %% Difference from C example as we don't have that kind of access
    %% to the training data, not officially anyway
    TrainData = [{{-1,-1}, {-1}},
		 {{-1,1}, {1}},
		 {{1,-1}, {1}},
		 {{1,1}, {-1}}],

    lists:foreach(
      fun({{Input1, Input2}=Input, {Output}}) ->
	      {CalcOut} = fannerl:run(Network, Input),
	      io:format("XOR test (~p, ~p) -> ~p, should be ~p, difference"
			"=~p~n", [Input1, Input2, CalcOut, Output,
				  abs(CalcOut-Output)])
      end, TrainData),
    fannerl:save(Network, "xor_float.net"),
    fannerl:destroy(Network),
    fannerl:destroy_train(Train),
    fannerl:stop().

train_on_steepness(Network, Train, MaxEpochs, EpochBetweenReports,
		   DesiredError, SteepnessStart, SteepnessStep,
		   SteepnessEnd) ->
    case EpochBetweenReports > 0 of
	true ->
	    io:format("Max epochs ~p. Desired error: ~p~n",
		      [MaxEpochs, DesiredError]);
	false->
	    ok
    end,
    
    fannerl:set_activation_steepness_all(Network, SteepnessStart),
    
    do_steepness_train(Network, Train, 1, MaxEpochs, EpochBetweenReports,
		       DesiredError, SteepnessStart, SteepnessStep,
		       SteepnessEnd).
do_steepness_train(_Network, _Train, Epoch, MaxEpochs, _EpochBetweenReports,
		   _DesiredError, _SteepnessStart, _SteepnessStep,
		   _SteepnessEnd) when Epoch > MaxEpochs ->
    ok;
do_steepness_train(Network, Train, Epoch, MaxEpochs, EpochBetweenReports,
		   DesiredError, SteepnessStart, SteepnessStep,
		   SteepnessEnd) ->
    ok = fannerl:train_epoch(Network, Train),
    #{mean_square_error := Mse} = fannerl:get_params(Network),
    
    case (EpochBetweenReports > 0) and
	((Epoch rem EpochBetweenReports) == 0) or
	(Mse < DesiredError) or
	(Epoch == 1) or (Epoch == MaxEpochs) of
	true ->
	    io:format("Epochs       ~p. Current error:  ~p~n",
		      [Epoch, Mse]);
	false ->
	    ok
    end,
    case Mse < DesiredError of
	true ->
	    NewSteepness = SteepnessStart + SteepnessStep,
	    if 
		NewSteepness =< SteepnessEnd ->
		    io:format("Steepness: ~p~n", [NewSteepness]),
		    fannerl:set_activation_steepness_all(Network, NewSteepness),
		    do_steepness_train(Network, Train, Epoch + 1, MaxEpochs,
				       EpochBetweenReports,
				       DesiredError, NewSteepness,
				       SteepnessStep,
				       SteepnessEnd);
		true ->
		    ok
	    end;
	false ->
	    do_steepness_train(Network, Train, Epoch + 1, MaxEpochs,
			       EpochBetweenReports,
			       DesiredError, SteepnessStart,
			       SteepnessStep,
			       SteepnessEnd)
    end.
		    

	    
	
	    
	
