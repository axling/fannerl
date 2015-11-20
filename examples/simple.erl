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
-module(simple).

-export([run/0]).

run() ->
    EpochBetweenReports = 100,
    MaxEpochs = 1000,
    fannerl:start(),
    N = fannerl:create({2,3,1}),
    fannerl:set_activation_function_all(N, fann_sigmoid_symmetric),
    fannerl:set_activation_steepness_all(N, 1),

    lists:foreach(
      fun(X) ->
	      ok = fannerl:train(N, {1.0, 1.0}, {-1.0}),
	      ok = fannerl:train(N, {1.0, -1.0}, {1.0}),
	      ok = fannerl:train(N, {-1.0, 1.0}, {1.0}),
	      ok = fannerl:train(N, {-1.0, -1.0}, {-1.0}),
	      Mse = fannerl:get_param(N, mse),
	      if
		  X rem EpochBetweenReports == 0 ->
		      io:format("It #~p, MSE: ~p~n", [X, Mse]);
		  true ->
		      ok
	      end
      end, lists:seq(1, MaxEpochs)),

    fannerl:reset_mse(N),
    fannerl:set_activation_function_all(N, fann_threshold_symmetric),
    {Out1} = fannerl:test(N, {1, 1}, {-1}),
    io:format("Test 1,1 -> ~p, expected -1, diff: ~p~n",
	      [Out1, abs(-1 - Out1)]),
    {Out2} = fannerl:test(N, {1, -1}, {1}),
    io:format("Test 1,1 -> ~p, expected 1, diff: ~p~n",
	      [Out2, abs(1 - Out2)]),
    {Out3} = fannerl:test(N, {-1, 1}, {1}),
    io:format("Test 1,1 -> ~p, expected 1, diff: ~p~n",
	      [Out3, abs(1 - Out3)]),
    {Out4} = fannerl:test(N, {-1, -1}, {-1}),
    io:format("Test 1,1 -> ~p, expected -1, diff: ~p~n",
	      [Out4, abs(-1 - Out4)]),
    
    fannerl:destroy(N),
    fannerl:stop().
    
